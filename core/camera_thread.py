"""Thread isolada para captura de vídeo, anonimização, inferência YOLO/Autoencoder e registro de eventos.

Pipeline de entrada do Autoencoder (idêntico ao treinamento):
  1. 17 keypoints (x, y) por pessoa rastreada (ByteTrack).
  2. Hip-centering (quadris 11 e 12) + normalização por largura/altura do frame -> 34D.
  3. Interpolação linear de falhas (até 3 frames) + EMA (alpha = 0.7).
  4. Janela de 15 frames -> [P, V, A] -> (1, 15, 102) float32.
  5. MSE de reconstrução + Z-Score robusto (mediana/MAD) para o alerta.

Gravação de eventos (vídeo .mp4 + dados .json, sem foto): um evento começa na primeira anomalia e dura enquanto ela persistir;
fecha após EVENT_POST_SECONDS de normalidade (ou EVENT_MAX_SECONDS). O clipe inclui
EVENT_PRE_SECONDS antes do início, é gravado com o FPS real e vem acompanhado de um JSON
com risco/MSE/z/keypoints por frame. Tudo vai para storage/eventos/AAAA-MM-DD/.
"""
import json
import os
import shutil
import sys
import threading
import time
import traceback
from collections import deque
from datetime import datetime
from pathlib import Path

import cv2
import numpy as np
from PySide6.QtCore import QThread, Signal
from ultralytics import YOLO

import config as _cfg
from core.database import BASE_DIR, EVENTS_DIR, STORAGE_DIR, EventDatabase
from core.frame_buffer import FrameBuffer

RECONNECT_DELAY = getattr(_cfg, "RECONNECT_DELAY", 3)
BUFFER_SECONDS = getattr(_cfg, "BUFFER_SECONDS", 30)
POSE_IMGSZ = getattr(_cfg, "POSE_IMGSZ", 480)
# Z-Score robusto que corresponde a 50% de risco no HUD (3.5 = limiar clássico de Iglewicz-Hoaglin)
ANOMALY_Z_THRESHOLD = float(getattr(_cfg, "ANOMALY_Z_THRESHOLD", 3.5))

# --- Gravação de eventos ---
EVENT_PRE_SECONDS = float(getattr(_cfg, "EVENT_PRE_SECONDS", 5))
EVENT_POST_SECONDS = float(getattr(_cfg, "EVENT_POST_SECONDS", 5))
EVENT_MAX_SECONDS = float(getattr(_cfg, "EVENT_MAX_SECONDS", 60))
RETENTION_PENDING_DAYS = float(getattr(_cfg, "RETENTION_PENDING_DAYS", 14))
RETENTION_FALSE_POSITIVE_DAYS = float(getattr(_cfg, "RETENTION_FALSE_POSITIVE_DAYS", 3))
STORAGE_MAX_GB = float(getattr(_cfg, "STORAGE_MAX_GB", 20))
MIN_FREE_GB = float(getattr(_cfg, "MIN_FREE_GB", 1))
CLEANUP_INTERVAL_SEC = 6 * 3600

DEBUG = bool(os.environ.get("VIGILANTE_DEBUG"))

MODELS_DIR = BASE_DIR / "models"

WINDOW_SIZE = 15          # frames por janela
EMA_ALPHA = 0.7           # S_t = 0.7 * P_t + 0.3 * S_{t-1}
MAX_INTERP_GAP = 3        # máximo de frames faltantes preenchidos por interpolação
MIN_HISTORY = 60          # MSEs mínimos no histórico antes de pontuar (calibração)
HISTORY_SIZE = 300
MAD_FLOOR_RATIO = 0.10    # piso do MAD = 10% da mediana (evita z explodir com cena parada)
STALE_TRACK_FRAMES = 30   # remove tracks ausentes por mais que isso


def anonymize_faces_from_keypoints(frame, keypoints_xy, padding_ratio=0.35):
    """Aplica desfoque Gaussiano na região da cabeça usando keypoints faciais COCO (0 a 4)."""
    anonymized = frame.copy()
    h, w, _ = anonymized.shape

    for person_kpts in keypoints_xy:
        head_pts = person_kpts[:5]
        valid_pts = head_pts[(head_pts[:, 0] > 0) & (head_pts[:, 1] > 0)]
        if len(valid_pts) == 0:
            continue

        x_min, y_min = np.min(valid_pts, axis=0)
        x_max, y_max = np.max(valid_pts, axis=0)

        face_w = max(x_max - x_min, 30.0)
        face_h = max(y_max - y_min, face_w)

        pad_x = face_w * padding_ratio
        pad_y = face_h * (padding_ratio * 1.3)

        x1 = int(max(0, x_min - pad_x))
        y1 = int(max(0, y_min - pad_y))
        x2 = int(min(w, x_max + pad_x))
        y2 = int(min(h, y_max + pad_y))

        if x2 > x1 and y2 > y1:
            roi = anonymized[y1:y2, x1:x2]
            anonymized[y1:y2, x1:x2] = cv2.GaussianBlur(roi, (51, 51), 30)

    return anonymized


class TrackState:
    """Estado temporal de UMA pessoa rastreada: interpolação de falhas, EMA e janela de 15 frames.

    Reproduz o tratamento "pré-janela" do treino de forma online:
      - falha de até 3 frames: espera a pessoa reaparecer e preenche por interpolação linear
        (a janela só avança quando a lacuna é resolvida);
      - falha maior que 3: os frames faltantes entram como zeros (0.0) e o EMA reinicia.
    """

    def __init__(self):
        self.window = deque(maxlen=WINDOW_SIZE)
        self.ema = None
        self.last_raw = None
        self.missing = 0
        self.absent = 0

    def _push(self, vec):
        if self.ema is None:
            self.ema = vec.copy()
        else:
            self.ema = EMA_ALPHA * vec + (1.0 - EMA_ALPHA) * self.ema
        self.window.append(self.ema.copy())

    def push_valid(self, vec):
        if self.missing > 0 and self.last_raw is not None and self.missing <= MAX_INTERP_GAP:
            n = self.missing
            for i in range(1, n + 1):
                interp = self.last_raw + (vec - self.last_raw) * (i / (n + 1))
                self._push(interp.astype(np.float32))
        self.missing = 0
        self.absent = 0
        self.last_raw = vec.copy()
        self._push(vec)

    def push_missing(self):
        self.absent += 1
        if self.last_raw is None:
            return
        self.missing += 1
        if self.missing == MAX_INTERP_GAP + 1:
            # lacuna grande demais: frames faltantes viram 0.0 e o EMA recomeça
            for _ in range(self.missing):
                self.window.append(np.zeros(34, dtype=np.float32))
            self.ema = None
            self.last_raw = None
            self.missing = 0

    def ready(self):
        """Janela cheia, sem lacuna pendente e sem frames zerados (evita falso positivo)."""
        if self.missing > 0 or len(self.window) < WINDOW_SIZE:
            return False
        return all(np.any(row) for row in self.window)


class CameraThread(QThread):
    frame_ready = Signal(object)
    fps_updated = Signal(float)
    status_changed = Signal(str)
    pose_status_changed = Signal(str, str)
    event_saved = Signal(int, str)
    storage_error = Signal(str)

    MAX_READ_FAILURES = 20
    NET_TIMEOUT_MS = 5000

    def __init__(self, source=0, camera_name="Câmera", threshold=0.70, parent=None):
        super().__init__(parent)
        self._source = source
        self._camera_name = str(camera_name) if camera_name else str(source)
        self._running = False
        self.threshold = threshold
        self.buffer = FrameBuffer(BUFFER_SECONDS)
        self.db = EventDatabase()

        self.mse_history = deque(maxlen=HISTORY_SIZE)
        self._tracks = {}
        self._sim_until = 0.0
        self._dbg = 0

        # Gravação de eventos
        self._ev = None
        self._recent_info = deque()
        self._pre_frames = deque()   # (t, jpeg) dos últimos EVENT_PRE_SECONDS
        self._fps_est = 10.0
        self._last_frame_t = None
        self._last_cleanup = 0.0
        self._last_disk_warn = 0.0

        self.yolo = None
        self.autoencoder = None
        self.expected_shape = None
        self._ae_input_size = None
        self._ae_mode = "none"      # "keras" | "none"
        self._ae_note = ""

    def simulate_anomaly(self, duration: float = 3.0):
        """Acionado por Ctrl+T na interface para testar disparo de alerta e gravação."""
        self._sim_until = time.monotonic() + duration

    # ------------------------------------------------------------------ modelos
    def _find_yolo_model(self) -> Path:
        MODELS_DIR.mkdir(parents=True, exist_ok=True)
        preferred = MODELS_DIR / "yolo26n-pose.pt"
        if preferred.exists():
            return preferred

        pt_files = sorted(MODELS_DIR.glob("*.pt"))
        if pt_files:
            return pt_files[0]

        raise FileNotFoundError(
            f"Nenhum modelo .pt encontrado em {MODELS_DIR}. Coloque 'yolo26n-pose.pt' nessa pasta."
        )

    def _load_autoencoder(self):
        """Carrega o .h5 com Keras 3 (backend torch). Nunca derruba a câmera se falhar."""
        h5_files = sorted(MODELS_DIR.glob("*.h5"))
        if not h5_files:
            self._ae_note = "sem autoencoder (.h5)"
            self.pose_status_changed.emit("on", "Somente YOLO (nenhum .h5 em models/)")
            return

        h5 = h5_files[0]
        self.pose_status_changed.emit("loading", f"Carregando {h5.name}...")
        try:
            import keras  # backend = torch (definido em main.py)

            self.autoencoder = keras.models.load_model(str(h5), compile=False)
            self.expected_shape = self.autoencoder.input_shape
            self._ae_input_size = int(np.prod([d for d in self.expected_shape[1:] if d]))
            self._ae_mode = "keras"
            self._ae_note = "YOLO + Autoencoder"
            print(f"[CameraThread] Autoencoder carregado: {h5.name} | input_shape={self.expected_shape}")
            self.pose_status_changed.emit("on", f"YOLO + Autoencoder ({h5.name})")
        except Exception as e:
            traceback.print_exc()
            self.autoencoder = None
            self._ae_mode = "none"
            self._ae_note = "autoencoder falhou (so YOLO)"
            self.pose_status_changed.emit("error", f"Autoencoder não carregou: {e}")

    def _init_models(self):
        if self.yolo is None:
            yolo_path = self._find_yolo_model()
            self.pose_status_changed.emit("loading", f"Carregando {yolo_path.name}...")
            self.yolo = YOLO(str(yolo_path))
            self.pose_status_changed.emit("on", f"Modelo ativo: {yolo_path.name}")

        if self.autoencoder is None and self._ae_mode == "none" and not self._ae_note:
            self._load_autoencoder()

    def _reset_tracking(self):
        """Zera tracks e o estado do ByteTrack (ao conectar/reconectar a câmera)."""
        self._tracks.clear()
        self._recent_info.clear()
        self._pre_frames.clear()
        try:
            predictor = getattr(self.yolo, "predictor", None)
            for tracker in getattr(predictor, "trackers", []) or []:
                tracker.reset()
        except Exception:
            pass

    # ---------------------------------------------------------------- features
    @staticmethod
    def _normalize_keypoints(kpts17, w, h):
        """Hip-centering + normalização pelo frame -> vetor de 34 dimensões."""
        p = kpts17.astype(np.float32).copy()
        hip_x = (p[11, 0] + p[12, 0]) / 2.0
        hip_y = (p[11, 1] + p[12, 1]) / 2.0

        if hip_x > 0 and hip_y > 0:
            p[:, 0] = (p[:, 0] - hip_x) / w
            p[:, 1] = (p[:, 1] - hip_y) / h
        else:
            p[:, 0] /= w
            p[:, 1] /= h
        return p.flatten()

    @staticmethod
    def _build_kinematic_102d(window) -> np.ndarray:
        """[P, V, A] com V[0] = A[0] = 0 -> shape (15, 102)."""
        pos = np.array(window, dtype=np.float32)
        vel = np.zeros_like(pos)
        vel[1:] = pos[1:] - pos[:-1]
        acc = np.zeros_like(vel)
        acc[1:] = vel[1:] - vel[:-1]
        return np.concatenate([pos, vel, acc], axis=-1).astype(np.float32)

    def _run_autoencoder(self, kinematic_102d: np.ndarray) -> float:
        """Retorna o MSE de reconstrução. Lança exceção se o formato for incompatível."""
        import torch

        if kinematic_102d.size != self._ae_input_size:
            raise ValueError(
                f"Entrada {kinematic_102d.shape} ({kinematic_102d.size}) incompatível com o modelo "
                f"{self.expected_shape} ({self._ae_input_size})"
            )

        if len(self.expected_shape) == 3:
            inp = np.expand_dims(kinematic_102d, axis=0)          # (1, 15, 102)
        else:
            inp = kinematic_102d.reshape(1, -1)
        inp = inp.astype(np.float32)

        with torch.no_grad():
            out = self.autoencoder(inp, training=False)
        reconstructed = np.asarray(out.detach().cpu().numpy() if hasattr(out, "detach") else out)
        return float(np.mean(np.square(inp - reconstructed)))

    def _robust_score(self, mse: float):
        """Z-Score robusto (mediana/MAD). Retorna (risco 0..1, z, calibrando)."""
        calibrating = len(self.mse_history) < MIN_HISTORY
        z = 0.0
        if not calibrating:
            hist = np.asarray(self.mse_history, dtype=np.float64)
            median = float(np.median(hist))
            mad = float(np.median(np.abs(hist - median)))
            mad = max(mad, MAD_FLOOR_RATIO * median, 1e-12)
            z = 0.6745 * (mse - median) / mad
        self.mse_history.append(mse)
        risk = float(np.clip(z / (2.0 * ANOMALY_Z_THRESHOLD), 0.0, 1.0))
        return risk, z, calibrating

    # --------------------------------------------------------------------- HUD
    def _draw_hud(self, frame, anomaly_score, is_anomaly, is_test=False,
                  persons=0, calibrating=False, mse=0.0, z=0.0):
        h, w, _ = frame.shape
        color = (0, 0, 255) if is_anomaly else (0, 255, 0)
        if is_test:
            status_text = "STATUS: ANOMALIA (TESTE)"
        elif is_anomaly:
            status_text = "STATUS: ANOMALIA DETECTADA!"
        elif calibrating:
            status_text = "STATUS: CALIBRANDO..."
            color = (0, 200, 255)
        else:
            status_text = "STATUS: NORMAL"

        score_text = f"Risco: {anomaly_score * 100:.1f}%"
        rec = " | REC" if self._ev is not None else ""
        info1 = f"Pessoas: {persons} | {self._ae_note or 'YOLO'}{rec}"
        info2 = f"MSE: {mse:.5f} | z: {z:.1f}"

        overlay = frame.copy()
        cv2.rectangle(overlay, (14, 14), (430, 134), (15, 15, 20), -1)
        cv2.addWeighted(overlay, 0.72, frame, 0.28, 0, frame)
        cv2.rectangle(frame, (14, 14), (430, 134), color, 2)

        cv2.putText(frame, status_text, (26, 46), cv2.FONT_HERSHEY_SIMPLEX, 0.65, color, 2)
        cv2.putText(frame, score_text, (26, 78), cv2.FONT_HERSHEY_SIMPLEX, 0.70, (255, 255, 255), 2)
        cv2.putText(frame, info1, (26, 102), cv2.FONT_HERSHEY_SIMPLEX, 0.50, (200, 200, 200), 1)
        cv2.putText(frame, info2, (26, 124), cv2.FONT_HERSHEY_SIMPLEX, 0.50, (200, 200, 200), 1)

        if is_anomaly:
            cv2.rectangle(frame, (0, 0), (w - 1, h - 1), (0, 0, 255), 8)

    # ------------------------------------------------------- gravação de eventos
    @staticmethod
    def _encode_frame(frame) -> bytes:
        """JPEG em memória: eventos longos ocupam ~20x menos RAM que frames crus."""
        ok, buf = cv2.imencode(".jpg", frame, [cv2.IMWRITE_JPEG_QUALITY, 90])
        return buf.tobytes() if ok else b""

    def _disk_ok(self) -> bool:
        try:
            free_gb = shutil.disk_usage(STORAGE_DIR).free / (1024 ** 3)
        except OSError:
            return True
        if free_gb >= MIN_FREE_GB:
            return True
        now = time.monotonic()
        if now - self._last_disk_warn > 60:
            self._last_disk_warn = now
            msg = f"Pouco espaço em disco ({free_gb:.2f} GB livres). Gravação de eventos pausada."
            print(f"[Armazenamento] {msg}")
            self.storage_error.emit(msg)
        return False

    def _update_recording(self, now, frame, is_anomaly, is_test, score, mse, z, persons, vec):
        info = {
            "t": now,
            "risk": round(float(score), 4),
            "mse": float(mse),
            "z": round(float(z), 3),
            "persons": int(persons),
            "kpts": np.round(vec, 4).tolist() if vec is not None else None,
        }
        self._recent_info.append(info)
        while self._recent_info and now - self._recent_info[0]["t"] > EVENT_PRE_SECONDS:
            self._recent_info.popleft()

        # Pré-roll próprio (não depende do FrameBuffer): JPEGs dos últimos N segundos
        enc = self._encode_frame(frame)
        self._pre_frames.append((now, enc))
        while self._pre_frames and now - self._pre_frames[0][0] > EVENT_PRE_SECONDS:
            self._pre_frames.popleft()

        ev = self._ev
        if ev is None:
            if not is_anomaly or not self._disk_ok():
                return
            pre_frames = [e for _, e in self._pre_frames]  # já inclui o frame atual
            ev = self._ev = {
                "start": now,
                "start_dt": datetime.now(),
                "frames": pre_frames,
                "info": list(self._recent_info),
                "last_anom": now,
                "best_score": float(score),
                "best_mse": float(mse),
                "max_persons": int(persons),
                "is_test": bool(is_test),
            }
            print(f"[EVENTO] Início ({'teste' if is_test else 'anomalia'}) risco={score:.2f} "
                  f"pré-roll={len(pre_frames)} frames")
            return

        ev["frames"].append(enc)
        ev["info"].append(info)
        ev["max_persons"] = max(ev["max_persons"], int(persons))
        if is_test:
            ev["is_test"] = True
        if is_anomaly:
            ev["last_anom"] = now
            if score >= ev["best_score"]:
                ev["best_score"] = float(score)
                ev["best_mse"] = float(mse)

        if (now - ev["last_anom"]) >= EVENT_POST_SECONDS or (now - ev["start"]) >= EVENT_MAX_SECONDS:
            self._finish_event()

    def _finish_event(self):
        """Fecha o evento atual e grava em segundo plano (não trava a câmera)."""
        ev, self._ev = self._ev, None
        if ev is None:
            return
        fps = float(np.clip(self._fps_est, 1.0, 60.0))
        threading.Thread(target=self._write_event, args=(ev, fps), daemon=True).start()

    def _write_event(self, ev, fps):
        try:
            start_dt = ev["start_dt"]
            day_dir = EVENTS_DIR / start_dt.strftime("%Y-%m-%d")
            day_dir.mkdir(parents=True, exist_ok=True)

            prefix = "teste" if ev["is_test"] else "alerta"
            base = day_dir / f"{prefix}_{start_dt.strftime('%H-%M-%S')}"
            n = 1
            while base.with_suffix(".mp4").exists():
                n += 1
                base = day_dir / f"{prefix}_{start_dt.strftime('%H-%M-%S')}_{n}"
            vid_path, json_path = base.with_suffix(".mp4"), base.with_suffix(".json")

            # 1) Vídeo (decodifica os frames JPEG e grava com o FPS real)
            frames = []
            for f in ev["frames"]:
                if isinstance(f, (bytes, bytearray)):
                    f = cv2.imdecode(np.frombuffer(f, np.uint8), cv2.IMREAD_COLOR) if f else None
                if f is not None:
                    frames.append(f)
            if not frames:
                raise IOError("Evento sem frames para gravar")

            fh, fw = frames[0].shape[:2]
            writer = None
            for codec in ("avc1", "mp4v"):
                writer = cv2.VideoWriter(str(vid_path), cv2.VideoWriter_fourcc(*codec), fps, (fw, fh))
                if writer.isOpened():
                    break
                writer.release()
                writer = None
            if writer is None:
                raise IOError("Nenhum codec de vídeo (avc1/mp4v) disponível no OpenCV")
            for f in frames:
                if f.shape[:2] == (fh, fw):
                    writer.write(f)
            writer.release()
            if not vid_path.exists() or vid_path.stat().st_size == 0:
                raise IOError(f"Vídeo não foi gravado: {vid_path}")

            # 2) JSON com os dados por frame (permite reajustar limiar / retreinar)
            t0 = ev["start"]
            duration = max(0.0, ev["last_anom"] - ev["start"])
            data = {
                "camera": self._camera_name,
                "inicio": start_dt.isoformat(timespec="seconds"),
                "teste": ev["is_test"],
                "fps_video": round(fps, 2),
                "duracao_anomalia_s": round(duration, 2),
                "limiar_interface": self.threshold,
                "z_limiar_50pct": ANOMALY_Z_THRESHOLD,
                "frames": [
                    {
                        "t": round(i["t"] - t0, 3),
                        "risco": i["risk"],
                        "mse": i["mse"],
                        "z": i["z"],
                        "pessoas": i["persons"],
                        "keypoints_34d": i["kpts"],
                    }
                    for i in ev["info"]
                ],
            }
            with open(json_path, "w", encoding="utf-8") as fp:
                json.dump(data, fp, ensure_ascii=False)

            # 3) Registro no banco (caminhos relativos à raiz do projeto)
            cam_label = f"{self._camera_name} [TESTE]" if ev["is_test"] else self._camera_name
            event_id = self.db.log_event(
                camera_name=cam_label,
                anomaly_score=ev["best_score"],
                raw_mse=ev["best_mse"],
                video_path=str(vid_path.relative_to(BASE_DIR)),
                duration_sec=duration,
                max_score=ev["best_score"],
                fps=fps,
                persons=ev["max_persons"],
                data_path=str(json_path.relative_to(BASE_DIR)),
                timestamp=start_dt,
            )
            print(f"[EVENTO] #{event_id} salvo em {day_dir} ({vid_path.name}, {len(frames)} frames, {fps:.1f} fps)")
            self.event_saved.emit(event_id, start_dt.strftime("%H:%M:%S"))
        except Exception as e:
            traceback.print_exc()
            self.storage_error.emit(f"Falha ao salvar evento: {e}")

    def _run_cleanup(self):
        def worker():
            try:
                res = self.db.cleanup(RETENTION_PENDING_DAYS, RETENTION_FALSE_POSITIVE_DAYS, STORAGE_MAX_GB)
                print(f"[Retenção] {res['removed']} evento(s) removido(s); eventos ocupam {res['size_gb']} GB")
            except Exception:
                traceback.print_exc()

        threading.Thread(target=worker, daemon=True).start()

    def _maybe_cleanup(self, now):
        if now - self._last_cleanup >= CLEANUP_INTERVAL_SEC:
            self._last_cleanup = now
            self._run_cleanup()

    # ------------------------------------------------------------ processamento
    def _evaluate_track(self, state: TrackState):
        """Retorna (risco, mse, z, calibrando) da janela atual de uma pessoa."""
        kin = self._build_kinematic_102d(state.window)

        if self._ae_mode == "keras" and self.autoencoder is not None:
            try:
                mse = self._run_autoencoder(kin)
                risk, z, calibrating = self._robust_score(mse)
                return risk, mse, z, calibrating
            except Exception:
                traceback.print_exc()
                self._ae_mode = "none"
                self._ae_note = "autoencoder incompativel (so YOLO)"
                self.pose_status_changed.emit("error", "Autoencoder incompatível; usando só velocidade")

        # Fallback sem autoencoder: norma da velocidade do último frame
        vel_norm = float(np.linalg.norm(kin[-1, 34:68]))
        return float(np.clip(vel_norm * 1.5, 0.0, 1.0)), vel_norm, 0.0, False

    def process_frame(self, frame):
        h, w, _ = frame.shape
        display_frame = frame.copy()

        # FPS real (média móvel) usado para gravar os vídeos na velocidade certa
        now_mono = time.monotonic()
        if self._last_frame_t is not None:
            dt = now_mono - self._last_frame_t
            if dt > 0:
                self._fps_est = 0.9 * self._fps_est + 0.1 * (1.0 / dt)
        self._last_frame_t = now_mono

        # ByteTrack mantém o mesmo ID por pessoa entre os frames (persist=True)
        results = self.yolo.track(
            frame, persist=True, tracker="bytetrack.yaml",
            imgsz=POSE_IMGSZ, conf=0.25, iou=0.45, verbose=False,
        )

        detections = []  # (track_id, vetor 34D normalizado)
        for result in results:
            if result.keypoints is None or len(result.keypoints) == 0:
                continue
            kpts = result.keypoints.xy.cpu().numpy()
            if len(kpts) == 0:
                continue

            display_frame = anonymize_faces_from_keypoints(display_frame, kpts)
            display_frame = result.plot(img=display_frame)

            ids = result.boxes.id if result.boxes is not None else None
            if ids is not None:
                ids = ids.int().cpu().tolist()
            else:
                ids = [-(i + 1) for i in range(len(kpts))]  # sem tracker: IDs provisórios

            for tid, person in zip(ids, kpts):
                detections.append((tid, self._normalize_keypoints(person, w, h)))
            break

        anomaly_score, raw_mse, z_best, calibrating = 0.0, 0.0, 0.0, False
        best_vec = None
        seen = set()

        for tid, vec in detections:
            state = self._tracks.get(tid)
            if state is None:
                state = self._tracks[tid] = TrackState()
            seen.add(tid)
            state.push_valid(vec)

            if best_vec is None:
                best_vec = vec
            if state.ready():
                risk, mse, z, calib = self._evaluate_track(state)
                if calib:
                    calibrating = True
                if risk >= anomaly_score:
                    anomaly_score, raw_mse, z_best = risk, mse, z
                    best_vec = state.window[-1]

        for tid in list(self._tracks.keys()):
            if tid in seen:
                continue
            st = self._tracks[tid]
            st.push_missing()
            if st.absent > STALE_TRACK_FRAMES:
                del self._tracks[tid]

        if anomaly_score > 0:
            calibrating = False

        if DEBUG:
            self._dbg += 1
            if self._dbg % 5 == 0:
                print(f"pessoas={len(detections)} mse={raw_mse:.6f} z={z_best:.2f} risco={anomaly_score:.2f} "
                      f"fps={self._fps_est:.1f}")

        is_test = now_mono < self._sim_until
        if is_test:
            anomaly_score = max(anomaly_score, 0.95)

        is_anomaly = anomaly_score > self.threshold
        self._draw_hud(display_frame, anomaly_score, is_anomaly, is_test=is_test,
                       persons=len(detections), calibrating=calibrating, mse=raw_mse, z=z_best)

        self._update_recording(now_mono, display_frame, is_anomaly, is_test, anomaly_score,
                               raw_mse, z_best, len(detections), best_vec)
        self._maybe_cleanup(now_mono)

        return np.ascontiguousarray(display_frame)

    # ------------------------------------------------------------ ciclo da thread
    def request_stop(self):
        """Sinaliza parada sem bloquear a thread principal da interface gráfica."""
        self._running = False

    def stop(self, wait_ms: int = 2000):
        self._running = False
        self.wait(wait_ms)

    def _sleep(self, seconds):
        end = time.monotonic() + seconds
        while self._running and time.monotonic() < end:
            time.sleep(0.05)

    def _open_capture(self):
        if isinstance(self._source, int):
            if sys.platform == "win32":
                cap = cv2.VideoCapture(self._source, cv2.CAP_DSHOW)
            else:
                # Backend padrão do OpenCV no Linux evita travamento de codec V4L2/MJPG
                cap = cv2.VideoCapture(self._source)

            if cap.isOpened():
                cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)
                # Warm-up: descarta frames iniciais vazios enquanto o sensor estabiliza
                for _ in range(5):
                    if not self._running:
                        break
                    ok, frame = cap.read()
                    if ok and frame is not None and frame.size > 0:
                        self.frame_ready.emit(np.ascontiguousarray(frame))
                        break
                    time.sleep(0.05)
            return cap

        params = [
            cv2.CAP_PROP_OPEN_TIMEOUT_MSEC, self.NET_TIMEOUT_MS,
            cv2.CAP_PROP_READ_TIMEOUT_MSEC, self.NET_TIMEOUT_MS,
        ]
        cap = cv2.VideoCapture(self._source, cv2.CAP_FFMPEG, params)
        cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)
        return cap

    def _read_loop(self, cap):
        failures = 0
        frames_count = 0
        last_tick = time.perf_counter()

        while self._running:
            ok, frame = cap.read()
            if not ok or frame is None or frame.size == 0:
                failures += 1
                if failures >= self.MAX_READ_FAILURES:
                    print("[CameraThread] Limite de falhas de leitura atingido, reconectando...")
                    return
                time.sleep(0.02)
                continue
            failures = 0

            if not self._running:
                break

            try:
                annotated_frame = self.process_frame(frame)
            except Exception:
                traceback.print_exc()
                annotated_frame = np.ascontiguousarray(frame)

            self.buffer.add(annotated_frame)
            self.frame_ready.emit(annotated_frame)

            frames_count += 1
            now = time.perf_counter()
            if now - last_tick >= 1.0:
                self.fps_updated.emit(frames_count / (now - last_tick))
                frames_count = 0
                last_tick = now

    def run(self):
        self._running = True

        # Garante as pastas e mostra, sem ambiguidade, ONDE tudo será gravado
        try:
            EVENTS_DIR.mkdir(parents=True, exist_ok=True)
            print(f"[Armazenamento] Banco : {self.db.db_path.resolve()}")
            print(f"[Armazenamento] Eventos: {EVENTS_DIR.resolve()}")
        except OSError as e:
            traceback.print_exc()
            self.storage_error.emit(f"Não foi possível criar {EVENTS_DIR}: {e}")

        try:
            self._init_models()
        except Exception as e:
            traceback.print_exc()
            self.pose_status_changed.emit("error", str(e))
            self.status_changed.emit("Erro ao carregar IA")
            return

        self._last_cleanup = time.monotonic()
        self._run_cleanup()  # limpeza na inicialização

        while self._running:
            self.status_changed.emit("Conectando...")
            cap = self._open_capture()

            if cap.isOpened():
                self._reset_tracking()
                self.status_changed.emit("Câmera conectada com IA ativa")
                self._read_loop(cap)
                cap.release()
                self._finish_event()  # não perde evento em andamento ao desconectar/parar
                if self._running:
                    self.status_changed.emit(f"Conexão perdida. Reconectando em {RECONNECT_DELAY}s...")
            else:
                cap.release()
                self.status_changed.emit(f"Sem conexão com a câmera. Tentando em {RECONNECT_DELAY}s...")

            self._sleep(RECONNECT_DELAY)

        self._finish_event()
