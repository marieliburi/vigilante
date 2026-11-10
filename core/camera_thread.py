"""Thread isolada para captura de vídeo com OpenCV (webcam ou RTSP).

A leitura dos frames roda fora da thread da interface. Se a câmera cair
(comum em Wi-Fi), a thread reconecta sozinha até o usuário parar.
"""
import sys
import time

import cv2
from PySide6.QtCore import QThread, Signal

import config as _cfg
from core.frame_buffer import FrameBuffer

RECONNECT_DELAY = getattr(_cfg, "RECONNECT_DELAY", 3)
BUFFER_SECONDS = getattr(_cfg, "BUFFER_SECONDS", 30)


class CameraThread(QThread):
    frame_ready = Signal(object)    # frame BGR (numpy.ndarray) já processado
    fps_updated = Signal(float)     # FPS medido a cada ~1 segundo
    status_changed = Signal(str)    # mensagens de conexão para a UI

    # Leituras falhas seguidas antes de considerar a conexão perdida.
    MAX_READ_FAILURES = 15
    # Timeouts (ms) para abrir/ler stream de rede, evita travar em câmera offline.
    NET_TIMEOUT_MS = 5000

    def __init__(self, source=0, parent=None):
        """source: int (índice da webcam) ou str (URL RTSP)."""
        super().__init__(parent)
        self._source = source
        self._running = False
        # Últimos N segundos de vídeo (usado pela UI para pausar/voltar/avançar).
        self.buffer = FrameBuffer(BUFFER_SECONDS)

    # ------------------------------------------------------------------
    # Hook de IA
    # ------------------------------------------------------------------
    def process_frame(self, frame):
        """Ponto de extensão para Visão Computacional.

        Recebe o frame BGR bruto e devolve o frame (possivelmente anotado)
        que será exibido. Futuramente: pose (YOLO), detecção de anomalia etc.
        Hoje apenas devolve o frame sem alterações.
        """
        return frame

    # ------------------------------------------------------------------
    # Controle
    # ------------------------------------------------------------------
    def stop(self):
        """Pede para o loop encerrar e aguarda a thread terminar."""
        self._running = False
        self.wait()

    def _sleep(self, seconds):
        """Espera em pequenos passos para responder rápido ao stop()."""
        end = time.monotonic() + seconds
        while self._running and time.monotonic() < end:
            time.sleep(0.1)

    # ------------------------------------------------------------------
    # Captura
    # ------------------------------------------------------------------
    def _open_capture(self):
        if isinstance(self._source, int):
            # Webcam USB. No Windows, CAP_DSHOW abre bem mais rápido.
            if sys.platform == "win32":
                return cv2.VideoCapture(self._source, cv2.CAP_DSHOW)
            return cv2.VideoCapture(self._source)

        # Stream de rede (RTSP) via FFmpeg, com timeouts.
        params = [
            cv2.CAP_PROP_OPEN_TIMEOUT_MSEC, self.NET_TIMEOUT_MS,
            cv2.CAP_PROP_READ_TIMEOUT_MSEC, self.NET_TIMEOUT_MS,
        ]
        cap = cv2.VideoCapture(self._source, cv2.CAP_FFMPEG, params)
        cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)  # menos latência
        return cap

    def _read_loop(self, cap):
        """Lê frames até parar ou perder a conexão."""
        failures = 0
        frames_count = 0
        last_tick = time.perf_counter()

        while self._running:
            ok, frame = cap.read()
            if not ok:
                failures += 1
                if failures >= self.MAX_READ_FAILURES:
                    return  # conexão perdida -> run() reconecta
                time.sleep(0.02)
                continue
            failures = 0

            # >>> Futuro módulo de IA entra aqui <<<
            frame = self.process_frame(frame)
            self.buffer.add(frame)          # guarda para o replay
            self.frame_ready.emit(frame)

            frames_count += 1
            now = time.perf_counter()
            if now - last_tick >= 1.0:
                self.fps_updated.emit(frames_count / (now - last_tick))
                frames_count = 0
                last_tick = now

    def run(self):
        self._running = True
        while self._running:
            self.status_changed.emit("Conectando...")
            cap = self._open_capture()

            if cap.isOpened():
                self.status_changed.emit("Conectado, aguardando imagem...")
                self._read_loop(cap)
                cap.release()
                if self._running:
                    self.status_changed.emit(
                        f"Conexão perdida. Reconectando em {RECONNECT_DELAY}s...")
            else:
                cap.release()
                self.status_changed.emit(
                    f"Sem conexão com a câmera. Nova tentativa em {RECONNECT_DELAY}s...")

            self._sleep(RECONNECT_DELAY)
