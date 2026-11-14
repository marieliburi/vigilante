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

# --- IA: extração de pose (YOLO + ByteTRACK) ---
POSE_ENABLED = True                         # False = só exibe a câmera, sem IA
POSE_MODEL_PATH = "models/yolo26n-pose.pt"  # relativo à pasta do projeto
POSE_IMGSZ = 640                            # resolução da inferência (320-480 = bem mais rápido em CPU)

# --- Detecção de anomalia (Autoencoder Keras .h5) ---
ANOMALY_MODEL_PATH = "models/autoencoder_shanghaitech_kinematic_102d.h5"
# Limiar de MSE validado por você (OBRIGATÓRIO). No HUD, o limiar corresponde a 50%.
ANOMALY_MSE_THRESHOLD = None
# Backend do Keras 3: "torch" reaproveita o PyTorch já instalado pelo YOLO.
# Se o .h5 não carregar, instale `pip install tensorflow` e use "tensorflow".
ANOMALY_KERAS_BACKEND = "torch"

# --- Evidências (só quando há anomalia; o desfoque dos rostos é aplicado ao gravar) ---
EVENT_PRE_SECONDS = 5     # segundos ANTES do início da anomalia
EVENT_POST_SECONDS = 5    # segundos DEPOIS que a anomalia termina
EVENT_MAX_SECONDS = 60    # duração máxima de um único evento (limita o uso de memória)
