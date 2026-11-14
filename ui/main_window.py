"""Janela principal (PySide6): vídeo ao vivo + controles de replay (timeshift).

Modos de exibição:
  LIVE    -> mostra o frame mais recente da câmera.
  PAUSED  -> congela num frame do buffer (a câmera continua gravando no buffer).
  PLAYING -> reproduz o buffer em velocidade normal, com atraso em relação ao vivo.
"""
import time

import cv2
from PySide6.QtCore import Qt, QTimer, Slot
from PySide6.QtGui import QImage, QKeySequence, QPixmap, QShortcut
from PySide6.QtWidgets import (
    QComboBox, QHBoxLayout, QLabel, QMainWindow, QPushButton, QSizePolicy,
    QSlider, QStyle, QVBoxLayout, QWidget,
)

from config import CAMERAS
from core.camera_thread import CameraThread

LIVE, PAUSED, PLAYING = "live", "paused", "playing"
SEEK_SECONDS = 5

STYLE = """
QMainWindow, QWidget { background: #1e1f24; color: #e5e7eb; font-size: 13px; }
QLabel#video { background: #0b0b0d; border-radius: 8px; color: #6b7280; font-size: 16px; }
QLabel#time { color: #9ca3af; font-family: monospace; }
QPushButton { background: #2d2f36; border: 1px solid #3b3e47; border-radius: 6px; padding: 7px 12px; }
QPushButton:hover { background: #383b44; }
QPushButton:disabled { color: #6b7280; background: #25262b; border-color: #2d2f36; }
QPushButton#primary { background: #2563eb; border: none; font-weight: bold; }
QPushButton#primary:hover { background: #1d4ed8; }
QPushButton#primary:disabled { background: #25262b; }
QPushButton#danger { background: #b91c1c; border: none; font-weight: bold; }
QPushButton#danger:hover { background: #991b1b; }
QPushButton#danger:disabled { background: #25262b; }
QPushButton#live { font-weight: bold; }
QComboBox { background: #2d2f36; border: 1px solid #3b3e47; border-radius: 6px; padding: 6px 10px; }
QComboBox QAbstractItemView { background: #2d2f36; selection-background-color: #2563eb; }
QSlider::groove:horizontal { height: 6px; background: #3b3e47; border-radius: 3px; }
QSlider::sub-page:horizontal { background: #2563eb; border-radius: 3px; }
QSlider::handle:horizontal { background: #f9fafb; width: 14px; margin: -5px 0; border-radius: 7px; }
QSlider::handle:horizontal:disabled { background: #6b7280; }
"""


class MainWindow(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("Vigilante App - Monitoramento")
        self.resize(1000, 760)
        self.setStyleSheet(STYLE)

        self._thread = None
        self._buffer = None
        self._mode = LIVE
        self._cur_seq = 0
        self._cur_ts = 0.0
        self._play_base_ts = 0.0
        self._play_base_wall = 0.0
        self._fps = None
        self._conn_msg = ""
        self._current_pixmap = None

        self._build_ui()
        self._build_shortcuts()

        self._tick_timer = QTimer(self)
        self._tick_timer.setInterval(200)
        self._tick_timer.timeout.connect(self._on_tick)

        self._play_timer = QTimer(self)
        self._play_timer.setInterval(33)
        self._play_timer.timeout.connect(self._play_tick)

        self._set_running(False)

    # ==================================================================
    # Construção da interface
    # ==================================================================
    def _tool_button(self, text, icon, tooltip, slot):
        btn = QPushButton(text)
        btn.setIcon(self.style().standardIcon(icon))
        btn.setToolTip(tooltip)
        btn.setFocusPolicy(Qt.NoFocus)
        btn.clicked.connect(slot)
        return btn

    def _build_ui(self):
        self.status_label = QLabel()
        self.status_label.setMinimumWidth(250)

        self.ai_label = QLabel()
        self._set_ai_status("idle", "")

        self.source_combo = QComboBox()
        self.source_combo.setMinimumWidth(200)
        self.source_combo.setFocusPolicy(Qt.NoFocus)
        for name, source in CAMERAS.items():
            self.source_combo.addItem(name, source)

        self.start_btn = QPushButton("Iniciar Câmera")
        self.start_btn.setObjectName("primary")
        self.stop_btn = QPushButton("Parar Câmera")
        self.stop_btn.setObjectName("danger")
        for b, slot in ((self.start_btn, self.start_camera), (self.stop_btn, self.stop_camera)):
            b.setFocusPolicy(Qt.NoFocus)
            b.clicked.connect(slot)

        top = QHBoxLayout()
        top.addWidget(self.status_label)
        top.addWidget(self.ai_label)
        top.addStretch(1)
        top.addWidget(QLabel("Câmera:"))
        top.addWidget(self.source_combo)
        top.addWidget(self.start_btn)
        top.addWidget(self.stop_btn)

        self.video_label = QLabel("Câmera parada")
        self.video_label.setObjectName("video")
        self.video_label.setAlignment(Qt.AlignCenter)
        self.video_label.setMinimumSize(640, 360)
        self.video_label.setSizePolicy(QSizePolicy.Ignored, QSizePolicy.Ignored)

        self.slider = QSlider(Qt.Horizontal)
        self.slider.setFocusPolicy(Qt.NoFocus)
        self.slider.valueChanged.connect(self._on_slider)
        self.time_label = QLabel("--")
        self.time_label.setObjectName("time")
        self.time_label.setMinimumWidth(210)
        self.time_label.setAlignment(Qt.AlignRight | Qt.AlignVCenter)

        timeline = QHBoxLayout()
        timeline.addWidget(self.slider, 1)
        timeline.addWidget(self.time_label)

        S = QStyle
        self.back_btn = self._tool_button(
            f"-{SEEK_SECONDS}s", S.SP_MediaSeekBackward,
            f"Voltar {SEEK_SECONDS}s (Ctrl+←)",
            lambda: self.seek_seconds(-SEEK_SECONDS),
        )
        self.prev_btn = self._tool_button(
            "Quadro", S.SP_MediaSkipBackward,
            "Quadro anterior (←)", lambda: self.step_frames(-1),
        )
        self.pause_btn = self._tool_button(
            "Pausar", S.SP_MediaPause,
            "Pausar / Reproduzir (Espaço)", self.toggle_pause,
        )
        self.next_btn = self._tool_button(
            "Quadro", S.SP_MediaSkipForward,
            "Próximo quadro (→)", lambda: self.step_frames(1),
        )
        self.fwd_btn = self._tool_button(
            f"+{SEEK_SECONDS}s", S.SP_MediaSeekForward,
            f"Avançar {SEEK_SECONDS}s (Ctrl+→)",
            lambda: self.seek_seconds(SEEK_SECONDS),
        )
        self.live_btn = self._tool_button(
            "Ao vivo", S.SP_MediaPlay,
            "Voltar para a imagem ao vivo (End)", self.go_live,
        )
        self.live_btn.setObjectName("live")
        self.pause_btn.setMinimumWidth(110)

        transport = QHBoxLayout()
        transport.addStretch(1)
        for b in (self.back_btn, self.prev_btn, self.pause_btn, self.next_btn,
                  self.fwd_btn, self.live_btn):
            transport.addWidget(b)
        transport.addStretch(1)

        layout = QVBoxLayout()
        layout.setContentsMargins(14, 12, 14, 12)
        layout.setSpacing(10)
        layout.addLayout(top)
        layout.addWidget(self.video_label, 1)
        layout.addLayout(timeline)
        layout.addLayout(transport)

        central = QWidget()
        central.setLayout(layout)
        self.setCentralWidget(central)

    def _build_shortcuts(self):
        keys = (
            ("Space", self.toggle_pause),
            ("Left", lambda: self.step_frames(-1)),
            ("Right", lambda: self.step_frames(1)),
            ("Ctrl+Left", lambda: self.seek_seconds(-SEEK_SECONDS)),
            ("Ctrl+Right", lambda: self.seek_seconds(SEEK_SECONDS)),
            ("End", self.go_live),
            ("Ctrl+T", self.simulate_anomaly),
        )
        for key, fn in keys:
            sc = QShortcut(QKeySequence(key), self)
            sc.activated.connect(fn)

    # ==================================================================
    # Estado da interface
    # ==================================================================
    def _set_running(self, running: bool):
        self.start_btn.setEnabled(not running)
        self.stop_btn.setEnabled(running)
        self.source_combo.setEnabled(not running)
        self._sync_ui()

    def _sync_ui(self):
        running = self._thread is not None
        for w in (self.back_btn, self.prev_btn, self.pause_btn,
                  self.next_btn, self.fwd_btn, self.slider):
            w.setEnabled(running)
        self.live_btn.setEnabled(running and self._mode != LIVE)

        S = QStyle
        if self._mode == PAUSED:
            self.pause_btn.setText("Reproduzir")
            self.pause_btn.setIcon(self.style().standardIcon(S.SP_MediaPlay))
        else:
            self.pause_btn.setText("Pausar")
            self.pause_btn.setIcon(self.style().standardIcon(S.SP_MediaPause))
        self._refresh_status()

    def _set_status(self, text, color):
        self.status_label.setText(text)
        self.status_label.setStyleSheet(
            f"background: {color}; color: white; font-weight: bold; "
            "padding: 7px 14px; border-radius: 6px;")

    def _behind_seconds(self):
        if self._mode == LIVE or self._buffer is None:
            return 0.0
        b = self._buffer.bounds()
        newest_ts = self._buffer.timestamp(b[1]) if b else None
        return max(0.0, newest_ts - self._cur_ts) if newest_ts else 0.0

    def _refresh_status(self):
        if self._thread is None:
            self._set_status("Status: Câmera Parada", "#4b5563")
        elif self._mode == PAUSED:
            self._set_status(f"Status: Pausado (-{self._behind_seconds():.1f}s)", "#b45309")
        elif self._mode == PLAYING:
            self._set_status(
                f"Status: Reproduzindo com atraso de {self._behind_seconds():.1f}s", "#b45309"
            )
        elif self._fps is not None:
            self._set_status(f"Status: Câmera Ativa [{self._fps:.0f} FPS]", "#15803d")
        else:
            self._set_status(f"Status: {self._conn_msg}", "#a16207")

    # ==================================================================
    # Câmera
    # ==================================================================
    @Slot()
    def start_camera(self):
        if self._thread is not None:
            return
        self._thread = CameraThread(
            self.source_combo.currentData(),
            camera_name=self.source_combo.currentText(),
        )
        self._buffer = self._thread.buffer
        self._mode = LIVE
        self._fps = None
        self._conn_msg = "Iniciando..."

        self._thread.frame_ready.connect(self._update_frame)
        self._thread.fps_updated.connect(self._update_fps)
        self._thread.status_changed.connect(self._on_status)
        self._thread.pose_status_changed.connect(self._on_pose_status)
        self._thread.event_saved.connect(self._on_event_saved)
        self._thread.storage_error.connect(self._on_storage_error)
        self._thread.finished.connect(self._on_thread_finished)
        self._thread.start()
        self._tick_timer.start()
        self._set_running(True)

    @Slot()
    def stop_camera(self):
        if self._thread is not None:
            # Não chama self._thread.wait() aqui para não congelar a janela
            self.stop_btn.setEnabled(False)
            self._set_status("Status: Parando câmera...", "#a16207")
            self._thread.request_stop()

    @Slot()
    def _on_thread_finished(self):
        self._tick_timer.stop()
        self._play_timer.stop()
        if self._thread is not None:
            self._thread.deleteLater()
        self._thread = None
        self._buffer = None
        self._mode = LIVE
        self._current_pixmap = None
        self.video_label.clear()
        self.video_label.setText("Câmera parada")
        self.slider.blockSignals(True)
        self.slider.setRange(0, 0)
        self.slider.blockSignals(False)
        self.time_label.setText("--")
        self._set_ai_status("idle", "")
        self._set_running(False)

    # ==================================================================
    # Sinais da thread
    # ==================================================================
    @Slot(object)
    def _update_frame(self, frame):
        if self._mode == LIVE:
            self._show_frame(frame)

    @Slot(float)
    def _update_fps(self, fps: float):
        self._fps = fps
        if self._mode == LIVE:
            self._refresh_status()

    @Slot(str)
    def _on_status(self, message: str):
        self._conn_msg = message
        self._fps = None
        self._refresh_status()

    def _set_ai_status(self, state: str, detail: str):
        texts = {
            "idle": ("IA: —", "#6b7280"),
            "off": ("IA: desativada", "#6b7280"),
            "loading": ("IA: carregando modelo...", "#fbbf24"),
            "on": ("IA: pose ativa", "#4ade80"),
            "error": ("IA: indisponível", "#f87171"),
        }
        text, color = texts.get(state, texts["idle"])
        self.ai_label.setText(text)
        self.ai_label.setStyleSheet(f"color: {color}; font-weight: bold;")
        self.ai_label.setToolTip(detail)

    @Slot(str, str)
    def _on_pose_status(self, state: str, detail: str):
        self._set_ai_status(state, detail)

    @Slot(int, str)
    def _on_event_saved(self, event_id: int, stamp: str):
        self.statusBar().showMessage(
            f"Evento #{event_id} salvo ({stamp}) em storage/eventos/", 10000
        )

    @Slot(str)
    def _on_storage_error(self, message: str):
        self.statusBar().showMessage(f"ERRO ao salvar evidência: {message}", 15000)

    # ==================================================================
    # Exibição
    # ==================================================================
    def _show_frame(self, frame):
        if frame is None or frame.size == 0:
            return
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        h, w, ch = rgb.shape
        image = QImage(rgb.data, w, h, ch * w, QImage.Format_RGB888).copy()
        self._current_pixmap = QPixmap.fromImage(image)
        self._render()

    def _render(self):
        if self._current_pixmap is not None and not self._current_pixmap.isNull():
            self.video_label.setPixmap(
                self._current_pixmap.scaled(
                    self.video_label.size(), Qt.KeepAspectRatio, Qt.FastTransformation
                )
            )

    def resizeEvent(self, event):
        super().resizeEvent(event)
        self._render()

    # ==================================================================
    # Controles de replay
    # ==================================================================
    def _show_seq(self, seq):
        b = self._buffer.bounds() if self._buffer else None
        if b is None:
            return None
        seq = max(b[0], min(seq, b[1]))
        result = self._buffer.get(seq)
        if result is None:
            return None
        ts, frame = result
        self._cur_seq, self._cur_ts = seq, ts
        self._show_frame(frame)
        return ts

    def _freeze(self):
        if self._mode != LIVE:
            return True
        b = self._buffer.bounds() if self._buffer else None
        if b is None:
            return False
        self._mode = PAUSED
        if self._show_seq(b[1]) is None:
            self._mode = LIVE
            return False
        return True

    def _rebase_play(self):
        self._play_base_ts = self._cur_ts
        self._play_base_wall = time.monotonic()

    def _seek_seq(self, seq):
        if self._mode == LIVE:
            self._mode = PAUSED
        self._show_seq(seq)
        if self._mode == PLAYING:
            self._rebase_play()
        self._sync_ui()

    @Slot()
    def toggle_pause(self):
        if self._thread is None:
            return
        if self._mode == LIVE:
            self._freeze()
        elif self._mode == PAUSED:
            self._rebase_play()
            self._mode = PLAYING
            self._play_timer.start()
        else:
            self._mode = PAUSED
            self._play_timer.stop()
        self._sync_ui()

    def step_frames(self, delta):
        if self._thread is None or not self._freeze():
            return
        self._play_timer.stop()
        self._mode = PAUSED
        self._show_seq(self._cur_seq + delta)
        self._sync_ui()

    def seek_seconds(self, seconds):
        if self._thread is None or not self._freeze():
            return
        seq = self._buffer.seq_at(self._cur_ts + seconds)
        if seq is not None:
            self._seek_seq(seq)

    @Slot()
    def go_live(self):
        if self._thread is None:
            return
        self._play_timer.stop()
        self._mode = LIVE
        self._sync_ui()

    @Slot()
    def simulate_anomaly(self):
        if self._thread is not None:
            self._thread.simulate_anomaly(3.0)
            self.statusBar().showMessage("Simulando anomalia (evento de TESTE)...", 4000)

    def _on_slider(self, value):
        if self._thread is not None and self._freeze():
            self._seek_seq(value)

    def _play_tick(self):
        b = self._buffer.bounds() if self._buffer else None
        if b is None:
            return
        target = self._play_base_ts + (time.monotonic() - self._play_base_wall)
        newest_ts = self._buffer.timestamp(b[1])
        if newest_ts is not None and target >= newest_ts:
            self.go_live()
            return
        seq = self._buffer.seq_at(target)
        if seq is not None and seq != self._cur_seq:
            self._show_seq(seq)

    def _on_tick(self):
        b = self._buffer.bounds() if self._buffer else None
        if b is None:
            return
        if not self.slider.isSliderDown():
            self.slider.blockSignals(True)
            self.slider.setRange(b[0], b[1])
            self.slider.setValue(b[1] if self._mode == LIVE else self._cur_seq)
            self.slider.blockSignals(False)

        span = self._buffer.span()
        if self._mode == LIVE:
            self.time_label.setText(f"AO VIVO  |  buffer {span:.0f}s")
        else:
            self.time_label.setText(f"-{self._behind_seconds():.1f}s  |  buffer {span:.0f}s")
            self._refresh_status()

    # ==================================================================
    def closeEvent(self, event):
        self._tick_timer.stop()
        self._play_timer.stop()
        if self._thread is not None:
            try:
                self._thread.finished.disconnect(self._on_thread_finished)
            except RuntimeError:
                pass
            self._thread.stop(1500)
        super().closeEvent(event)