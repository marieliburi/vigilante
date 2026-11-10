"""Configuração local (NÃO versionar: contém senha)."""

CAMERAS = {
    "Câmera Yoosee (principal)": "rtsp://admin:vigilante2026%2A@192.168.0.121:554/onvif1",
    "Câmera Yoosee (sub-stream)": "rtsp://admin:vigilante2026%2A@192.168.0.121:554/onvif2",
    "Webcam local (0)": 0,
}

# "udp" = menor latência | "tcp" = mais estável (use se a imagem tiver borrões/travadas)
RTSP_TRANSPORT = "udp"

# Segundos de espera antes de tentar reconectar quando a câmera cai.
RECONNECT_DELAY = 3
