"""Modelo de configuração. Copie para `config.py` e preencha com seus dados.

`config.py` está no .gitignore para que a senha da câmera NUNCA vá para o Git.
"""

# Nome exibido no seletor -> fonte de vídeo.
#   int  = índice de webcam USB (0, 1, ...)
#   str  = URL RTSP (caracteres especiais na senha devem ser codificados: * -> %2A, @ -> %40)
CAMERAS = {
    "Câmera Yoosee (principal)": "rtsp://admin:SENHA@192.168.0.121:554/onvif1",
    "Câmera Yoosee (sub-stream)": "rtsp://admin:SENHA@192.168.0.121:554/onvif2",
    "Webcam local (0)": 0,
}

# Transporte do RTSP: "udp" (menor latência) ou "tcp" (mais estável no Wi-Fi).
RTSP_TRANSPORT = "udp"

# Segundos de espera antes de tentar reconectar quando a câmera cai.
RECONNECT_DELAY = 3

# Quantos segundos de vídeo ficam guardados na memória para pausar/voltar/avançar.
# (~3 a 6 MB por segundo em 720p/1080p. Reduza se faltar memória.)
BUFFER_SECONDS = 30
