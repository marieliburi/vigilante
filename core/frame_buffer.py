"""Buffer circular com os últimos N segundos de vídeo.

Os frames são guardados comprimidos em JPEG para economizar memória
(um frame 1080p cru ocupa ~6 MB; em JPEG, ~150 KB). Isso permite pausar,
voltar e avançar mesmo com uma câmera ao vivo.

Cada frame guarda também as regiões de rosto (`rois`) detectadas pelo YOLO.
Elas ficam só na memória e servem para borrar os rostos no momento em que uma
evidência é gravada em disco. O buffer em si nunca é escrito em disco.

Cada frame recebe um número de sequência (`seq`) crescente, que serve como
posição estável mesmo quando os frames mais antigos são descartados.
"""
import threading
import time
from collections import deque

import cv2


class FrameBuffer:
    def __init__(self, seconds=30, jpeg_quality=80):
        self._seconds = seconds
        self._params = [cv2.IMWRITE_JPEG_QUALITY, jpeg_quality]
        self._frames = deque()   # itens: (timestamp, bytes_jpeg, rois)
        self._first_seq = 0      # seq do frame mais antigo ainda guardado
        self._lock = threading.Lock()

    # --- escrita (thread da câmera) ------------------------------------
    def add(self, frame, rois=None):
        """Guarda o frame. `rois`: lista de (x1,y1,x2,y2) de rostos; [] = sem pessoas;
        None = desconhecido (IA falhou), caso em que a evidência sai bloqueada.
        Devolve o item guardado (timestamp, jpeg, rois) ou None se a codificação falhar."""
        ok, encoded = cv2.imencode(".jpg", frame, self._params)
        if not ok:
            return None
        now = time.monotonic()
        item = (now, encoded, rois)
        with self._lock:
            self._frames.append(item)
            while self._frames and now - self._frames[0][0] > self._seconds:
                self._frames.popleft()
                self._first_seq += 1
        return item

    # --- leitura (thread da interface) ---------------------------------
    def bounds(self):
        """(seq_mais_antigo, seq_mais_novo) ou None se vazio."""
        with self._lock:
            if not self._frames:
                return None
            return self._first_seq, self._first_seq + len(self._frames) - 1

    def timestamp(self, seq):
        with self._lock:
            idx = seq - self._first_seq
            if 0 <= idx < len(self._frames):
                return self._frames[idx][0]
        return None

    def get(self, seq):
        """Devolve (timestamp, frame BGR) ou None se o frame já saiu do buffer."""
        with self._lock:
            idx = seq - self._first_seq
            if not 0 <= idx < len(self._frames):
                return None
            ts, encoded, _ = self._frames[idx]
        return ts, cv2.imdecode(encoded, cv2.IMREAD_COLOR)

    def seq_at(self, ts):
        """seq do frame mais recente com timestamp <= ts (limitado ao mais antigo)."""
        with self._lock:
            if not self._frames:
                return None
            seq = self._first_seq + len(self._frames) - 1
            for item in reversed(self._frames):
                if item[0] <= ts:
                    return seq
                seq -= 1
            return self._first_seq

    def recent(self, seconds):
        """Referências aos frames dos últimos `seconds` s: [(timestamp, jpeg, rois)]."""
        with self._lock:
            if not self._frames:
                return []
            limit = self._frames[-1][0] - seconds
            return [item for item in self._frames if item[0] >= limit]

    def span(self):
        """Duração (s) do vídeo guardado."""
        with self._lock:
            if len(self._frames) < 2:
                return 0.0
            return self._frames[-1][0] - self._frames[0][0]

    def get_recent_frames(self, seconds=6):
        """Retorna lista de frames BGR dos últimos `seconds` segundos para salvar clipe."""
        cutoff = time.monotonic() - seconds
        with self._lock:
            encoded_list = [enc for ts, enc in self._frames if ts >= cutoff]
        frames = []
        for enc in encoded_list:
            img = cv2.imdecode(enc, cv2.IMREAD_COLOR)
            if img is not None:
                frames.append(img)
        return frames
