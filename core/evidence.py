"""Anonimização e salvamento de evidências (somente quando há anomalia).

O vídeo ao vivo e o buffer de replay ficam SEM desfoque (só na memória). O desfoque
dos rostos é aplicado aqui, no instante em que snapshot e clipe são gravados:
nenhum pixel de rosto sem borrar chega ao disco.

Tudo roda numa thread própria: a thread de captura apenas entrega os frames e segue.
"""
import logging
import queue
import threading

import cv2
import numpy as np

from core import database

log = logging.getLogger("vigilante")
_FONT = cv2.FONT_HERSHEY_SIMPLEX


# ======================================================================
# Anonimização facial
# ======================================================================
def compute_face_rois(keypoints_xy, shape, padding_ratio=0.35, boxes_xyxy=None):
    """Regiões (x1, y1, x2, y2) da cabeça de TODAS as pessoas, pelos keypoints 0-4 do YOLO.

    `boxes_xyxy` (opcional) é uma rede de segurança: se uma pessoa vem sem nenhum
    keypoint de cabeça válido, usa o topo da caixa dela em vez de deixar o rosto à mostra.
    """
    h, w = shape[:2]
    rois = []

    for i, person_kpts in enumerate(keypoints_xy):
        head_pts = person_kpts[:5]  # 0: nariz, 1-2: olhos, 3-4: orelhas
        valid_pts = head_pts[(head_pts[:, 0] > 0) & (head_pts[:, 1] > 0)]

        if len(valid_pts) == 0:
            # Sem keypoints de cabeça: usa o topo (25%) da caixa da pessoa.
            if boxes_xyxy is not None and i < len(boxes_xyxy):
                bx1, by1, bx2, by2 = boxes_xyxy[i]
                bw, bh = bx2 - bx1, by2 - by1
                x1 = int(max(0, bx1 + bw * 0.2))
                x2 = int(min(w, bx2 - bw * 0.2))
                y1 = int(max(0, by1))
                y2 = int(min(h, by1 + bh * 0.25))
                if x2 > x1 and y2 > y1:
                    rois.append((x1, y1, x2, y2))
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
            rois.append((x1, y1, x2, y2))

    return rois


def blur_rois(frame, rois):
    """Desfoque pesado nas regiões informadas (devolve uma cópia)."""
    anonymized = frame.copy()
    for x1, y1, x2, y2 in rois:
        roi = anonymized[y1:y2, x1:x2]
        anonymized[y1:y2, x1:x2] = cv2.GaussianBlur(roi, (51, 51), 30)
    return anonymized


def anonymize_faces_from_keypoints(frame, keypoints_xy, padding_ratio=0.35, boxes_xyxy=None):
    """Atalho: calcula as regiões de rosto e já aplica o desfoque."""
    return blur_rois(frame, compute_face_rois(keypoints_xy, frame.shape, padding_ratio, boxes_xyxy))


def blocked_frame(shape):
    """Tela preta com aviso, usada quando não dá para garantir a anonimização de um frame."""
    out = np.zeros(shape, dtype=np.uint8)
    h, w = shape[:2]
    scale = max(0.5, min(1.0, w / 900.0))
    for i, text in enumerate(("QUADRO BLOQUEADO", "Privacidade: rostos nao verificados")):
        (tw, th), _ = cv2.getTextSize(text, _FONT, scale, 1)
        cv2.putText(out, text, (max(5, (w - tw) // 2), h // 2 + i * int(th * 2.2)),
                    _FONT, scale, (200, 200, 200), 1, cv2.LINE_AA)
    return out


# ======================================================================
# Gravação em disco + SQLite
# ======================================================================
def _unique_stem(day_dir, stem):
    """Evita sobrescrever arquivos caso dois alertas caiam no mesmo segundo."""
    candidate, n = stem, 1
    while (day_dir / f"{candidate}.jpg").exists() or (day_dir / f"{candidate}.mp4").exists():
        n += 1
        candidate = f"{stem}_{n}"
    return candidate


class EvidenceSaver:
    def __init__(self, on_saved=None, on_error=None):
        """on_saved(evento_id, timestamp) e on_error(mensagem) rodam na thread do saver."""
        self._on_saved = on_saved
        self._on_error = on_error
        self._queue = queue.Queue(maxsize=5)
        self._worker = threading.Thread(target=self._run, name="EvidenceSaver", daemon=True)
        self._worker.start()

    def submit(self, camera_name, score, raw_mse, frames, trigger_index, when, status="PENDENTE"):
        """Enfileira um evento.

        frames: [(timestamp, jpeg, rois)] = pré-evento + anomalia + pós-evento.
        trigger_index: posição do frame que disparou o alerta (usado no snapshot).
        when: datetime do disparo.
        """
        job = {"when": when, "camera": camera_name, "score": score, "mse": raw_mse,
               "frames": frames, "trigger": trigger_index, "status": status}
        try:
            self._queue.put_nowait(job)
        except queue.Full:
            self._fail("Fila de evidências cheia: evento descartado.")

    def close(self, timeout=30):
        """Termina os salvamentos pendentes (evita perder evidência ao fechar o app)."""
        self._queue.put(None)
        self._worker.join(timeout)

    # ------------------------------------------------------------------
    def _fail(self, message):
        log.error(message)
        if self._on_error:
            self._on_error(message)

    def _run(self):
        try:
            database.init_db()
        except Exception as exc:
            self._fail(f"Banco de eventos indisponível: {exc}")
        while True:
            job = self._queue.get()
            if job is None:
                return
            try:
                self._save(job)
            except Exception as exc:
                log.exception("Falha ao salvar evidência")
                self._fail(f"Falha ao salvar evidência: {exc}")

    def _save(self, job):
        frames, when = job["frames"], job["when"]
        if not frames:
            raise ValueError("nenhum frame no evento")

        day_dir = database.STORAGE_DIR / "eventos" / when.strftime("%Y-%m-%d")
        day_dir.mkdir(parents=True, exist_ok=True)
        stem = _unique_stem(day_dir, f"alerta_{when:%H-%M-%S}")
        snap_path, video_path = day_dir / f"{stem}.jpg", day_dir / f"{stem}.mp4"

        # FPS real do trecho (o buffer guarda o ritmo em que a IA conseguiu processar).
        span = frames[-1][0] - frames[0][0]
        fps = min(60.0, max(1.0, (len(frames) - 1) / span)) if span > 0 else 15.0

        writer, snapshot, last = None, None, None
        for i, (_, encoded, rois) in enumerate(frames):
            img = cv2.imdecode(encoded, cv2.IMREAD_COLOR)  # frame ao vivo (sem desfoque)
            if img is None:
                continue
            # ANONIMIZAÇÃO: único ponto em que o frame vai para disco.
            img = blocked_frame(img.shape) if rois is None else blur_rois(img, rois)

            if writer is None:
                h, w = img.shape[:2]
                writer = cv2.VideoWriter(str(video_path), cv2.VideoWriter_fourcc(*"mp4v"),
                                         fps, (w, h))
                if not writer.isOpened():
                    log.warning("VideoWriter mp4v indisponível; evento salvo sem clipe.")
                    writer = False
            if writer:
                writer.write(img)
            if i == job["trigger"]:
                snapshot = img
            last = img
        if writer:
            writer.release()

        snapshot = snapshot if snapshot is not None else last
        if snapshot is None or not cv2.imwrite(str(snap_path), snapshot,
                                               [cv2.IMWRITE_JPEG_QUALITY, 90]):
            raise IOError("não foi possível gravar o snapshot")

        def rel(path):  # caminhos relativos à pasta do projeto, sempre com "/"
            return path.relative_to(database.PROJECT_ROOT).as_posix()

        has_video = bool(writer) and video_path.exists() and video_path.stat().st_size > 0
        stamp = when.isoformat(timespec="seconds")
        evento_id = database.insert_evento(
            timestamp=stamp, camera_name=job["camera"], anomaly_score=job["score"],
            raw_mse=job["mse"], video_path=rel(video_path) if has_video else None,
            snapshot_path=rel(snap_path), validado_status=job["status"])
        if self._on_saved:
            self._on_saved(evento_id, stamp)
