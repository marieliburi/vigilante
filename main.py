"""Ponto de entrada do Vigilante App.

Execute com:  python main.py
"""
import os
import sys

# ---------------------------------------------------------------------------
# IMPORTANTE: tudo abaixo precisa rodar ANTES de importar cv2, torch, keras...
# ---------------------------------------------------------------------------

# Keras 3 usando PyTorch como backend: o TensorFlow NUNCA é carregado.
# (TensorFlow + PyTorch no mesmo processo causam o crash
#  "LLVM ERROR: inconsistency in registered CommandLine options".)
os.environ["KERAS_BACKEND"] = "torch"
os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "2")

try:
    from config import RTSP_TRANSPORT
except ImportError:
    sys.exit("Arquivo config.py não encontrado. Copie config.example.py para config.py "
             "e preencha com os dados da sua câmera.")

# Força o transporte do RTSP (udp/tcp) no FFmpeg usado pelo OpenCV.
os.environ["OPENCV_FFMPEG_CAPTURE_OPTIONS"] = f"rtsp_transport;{RTSP_TRANSPORT}"

# O ultralytics instala o `opencv-python` completo, cujo cv2 aponta o Qt para os
# plugins dele (erro "xcb"). Removemos o desvio ANTES de importar o Qt.
for _var in ("QT_QPA_PLATFORM_PLUGIN_PATH", "QT_QPA_FONTDIR"):
    if "cv2" in os.environ.get(_var, ""):
        os.environ.pop(_var, None)

# Carrega torch e keras na thread principal, uma única vez e na ordem correta.
import torch  # noqa: E402,F401
import keras  # noqa: E402,F401

from PySide6.QtWidgets import QApplication  # noqa: E402

from ui.main_window import MainWindow  # noqa: E402


def main() -> int:
    # Remove de novo, caso o import do cv2 (via ultralytics) tenha recolocado o desvio.
    for var in ("QT_QPA_PLATFORM_PLUGIN_PATH", "QT_QPA_FONTDIR"):
        if "cv2" in os.environ.get(var, ""):
            os.environ.pop(var, None)

    app = QApplication(sys.argv)
    window = MainWindow()
    window.show()
    return app.exec()


if __name__ == "__main__":
    sys.exit(main())
