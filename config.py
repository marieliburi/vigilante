"""Configuração local (NÃO versionar: contém senha)."""

CAMERAS = {
    "Webcam local (0)": 0,
    "Câmera Yoosee (principal)": "rtsp://admin:SENHA@192.168.0.121:554/onvif1",
    "Câmera Yoosee (sub-stream)": "rtsp://admin:SENHA@192.168.0.121:554/onvif2",
}

# "udp" = menor latência | "tcp" = mais estável (use se a imagem tiver borrões/travadas)
RTSP_TRANSPORT = "udp"

# Segundos de espera antes de tentar reconectar quando a câmera cai.
RECONNECT_DELAY = 3

# Segundos de vídeo guardados em memória (também é de onde vem o "antes" do evento).
BUFFER_SECONDS = 30

# Resolução da inferência do YOLO (320-480 = mais rápido em CPU; 640 = mais preciso).
POSE_IMGSZ = 480

# Z-Score robusto (mediana/MAD) que equivale a 50% de risco no HUD.
# Menor = mais sensível (mais alertas). Maior = menos alertas.
ANOMALY_Z_THRESHOLD = 1.5

# --- Gravação de eventos (storage/eventos/AAAA-MM-DD/) ---
EVENT_PRE_SECONDS = 5     # segundos ANTES do início da anomalia
EVENT_POST_SECONDS = 5    # segundos DEPOIS que a anomalia termina
EVENT_MAX_SECONDS = 60    # duração máxima de um único evento

# --- Retenção (rodam na inicialização e a cada 6 h). CONFIRMADO nunca é apagado. ---
RETENTION_PENDING_DAYS = 14          # eventos não validados
RETENTION_FALSE_POSITIVE_DAYS = 3    # eventos marcados como falso positivo
STORAGE_MAX_GB = 20                  # limite da pasta de eventos (apaga os mais antigos)
MIN_FREE_GB = 1                      # abaixo disso a gravação pausa e avisa
