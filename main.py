"""Ponto de entrada do Vigilante App.

Execute com:  python main.py
"""
import os
import sys

# IMPORTANTE: precisa ser definido ANTES de qualquer `import cv2`.
try:
    from config import RTSP_TRANSPORT
except ImportError:
    sys.exit("Arquivo config.py não encontrado. Copie config.example.py para config.py "
             "e preencha com os dados da sua câmera.")

# Força o transporte do RTSP (udp/tcp) no FFmpeg usado pelo OpenCV.
os.environ["OPENCV_FFMPEG_CAPTURE_OPTIONS"] = f"rtsp_transport;{RTSP_TRANSPORT}"

from PySide6.QtWidgets import QApplication  # noqa: E402

from ui.main_window import MainWindow  # noqa: E402


def main() -> int:
    app = QApplication(sys.argv)
    window = MainWindow()
    window.show()
    return app.exec()


if __name__ == "__main__":
    sys.exit(main())
