"""Gerenciamento do banco de dados SQLite para registro de eventos e auditoria.

Este módulo é a ÚNICA fonte dos caminhos de armazenamento. O camera_thread importa
STORAGE_DIR / EVENTS_DIR daqui, então vídeos, fotos e JSONs ficam sempre na mesma
pasta `storage/` onde está o vigilante.db:

    <raiz do projeto>/storage/
        vigilante.db
        eventos/AAAA-MM-DD/alerta_HH-MM-SS.{jpg,mp4,json}
"""
import os
import sqlite3
import threading
from contextlib import contextmanager
from datetime import datetime, timedelta
from pathlib import Path

BASE_DIR = Path(__file__).resolve().parent.parent
STORAGE_DIR = BASE_DIR / "storage"
EVENTS_DIR = STORAGE_DIR / "eventos"
DB_PATH = STORAGE_DIR / "vigilante.db"

STATUS_PENDING = "PENDENTE"
STATUS_CONFIRMED = "CONFIRMADO"
STATUS_FALSE_POSITIVE = "FALSO_POSITIVO"
VALID_STATUS = (STATUS_PENDING, STATUS_CONFIRMED, STATUS_FALSE_POSITIVE)

_EXTRA_COLUMNS = [
    ("duration_sec", "REAL"),
    ("max_score", "REAL"),
    ("fps", "REAL"),
    ("persons", "INTEGER"),
    ("data_path", "TEXT"),
]


class EventDatabase:
    def __init__(self, db_path: Path = DB_PATH):
        self.db_path = Path(db_path)
        self.db_path.parent.mkdir(parents=True, exist_ok=True)
        # Os caminhos gravados no banco são relativos à raiz do projeto (pai de storage/)
        self.base_dir = self.db_path.parent.parent
        self.events_dir = self.db_path.parent / "eventos"
        self._lock = threading.Lock()
        self._init_db()

    @contextmanager
    def _conn(self):
        conn = sqlite3.connect(str(self.db_path), timeout=10, check_same_thread=False)
        try:
            yield conn
            conn.commit()
        except Exception:
            conn.rollback()
            raise
        finally:
            conn.close()

    def _init_db(self):
        with self._lock, self._conn() as conn:
            conn.execute("PRAGMA journal_mode=WAL")
            conn.execute("""
                CREATE TABLE IF NOT EXISTS eventos (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    timestamp TEXT NOT NULL,
                    camera_name TEXT NOT NULL,
                    anomaly_score REAL NOT NULL,
                    raw_mse REAL NOT NULL,
                    video_path TEXT,
                    snapshot_path TEXT,
                    validado_status TEXT DEFAULT 'PENDENTE'
                )
            """)
            # Migração: acrescenta colunas novas em bancos já existentes
            existing = {row[1] for row in conn.execute("PRAGMA table_info(eventos)")}
            for name, ddl in _EXTRA_COLUMNS:
                if name not in existing:
                    conn.execute(f"ALTER TABLE eventos ADD COLUMN {name} {ddl}")
            conn.execute("CREATE INDEX IF NOT EXISTS idx_eventos_ts ON eventos(timestamp)")
            conn.execute("CREATE INDEX IF NOT EXISTS idx_eventos_status ON eventos(validado_status)")

    # ------------------------------------------------------------------ escrita
    def log_event(
        self,
        camera_name: str,
        anomaly_score: float,
        raw_mse: float,
        video_path: str = "",
        snapshot_path: str = "",
        duration_sec: float = 0.0,
        max_score: float = 0.0,
        fps: float = 0.0,
        persons: int = 0,
        data_path: str = "",
        timestamp: datetime = None,
    ) -> int:
        ts = (timestamp or datetime.now()).strftime("%Y-%m-%d %H:%M:%S")
        with self._lock, self._conn() as conn:
            cursor = conn.execute(
                """
                INSERT INTO eventos (
                    timestamp, camera_name, anomaly_score, raw_mse, video_path, snapshot_path,
                    duration_sec, max_score, fps, persons, data_path
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                """,
                (ts, camera_name, float(anomaly_score), float(raw_mse), video_path, snapshot_path,
                 float(duration_sec), float(max_score), float(fps), int(persons), data_path),
            )
            return cursor.lastrowid

    def set_status(self, event_id: int, status: str):
        """Marca o evento como PENDENTE, CONFIRMADO ou FALSO_POSITIVO (para a interface usar)."""
        if status not in VALID_STATUS:
            raise ValueError(f"Status inválido: {status}. Use um de {VALID_STATUS}")
        with self._lock, self._conn() as conn:
            conn.execute("UPDATE eventos SET validado_status = ? WHERE id = ?", (status, int(event_id)))

    # ---------------------------------------------------------------- retenção
    def _files_of(self, row):
        """Arquivos de um evento (caminhos relativos à raiz do projeto), só dentro de eventos/."""
        files = []
        events_root = self.events_dir.resolve()
        for rel in row[3:6]:  # video_path, snapshot_path, data_path
            if not rel:
                continue
            p = (self.base_dir / rel).resolve()
            try:
                p.relative_to(events_root)
            except ValueError:
                continue  # nunca apaga nada fora de storage/eventos
            files.append(p)
        return files

    @staticmethod
    def _size(files):
        total = 0
        for f in files:
            try:
                total += f.stat().st_size
            except OSError:
                pass
        return total

    def _delete_event(self, conn, row):
        for f in self._files_of(row):
            try:
                f.unlink()
            except FileNotFoundError:
                pass
            except OSError as e:
                print(f"[Retenção] Não foi possível apagar {f}: {e}")
        conn.execute("DELETE FROM eventos WHERE id = ?", (row[0],))

    def cleanup(self, pending_days: float = 14, false_positive_days: float = 3, max_gb: float = 20.0) -> dict:
        """Apaga eventos antigos. CONFIRMADO nunca é apagado.

        - PENDENTE mais velho que `pending_days`
        - FALSO_POSITIVO mais velho que `false_positive_days`
        - Se a pasta de eventos passar de `max_gb`, apaga os mais antigos não confirmados.
        """
        now = datetime.now()
        removed = 0
        with self._lock, self._conn() as conn:
            rows = conn.execute(
                "SELECT id, timestamp, validado_status, video_path, snapshot_path, data_path "
                "FROM eventos ORDER BY timestamp ASC"
            ).fetchall()

            # Reordena colunas para _files_of (índices 3..5 = caminhos)
            keep = []
            for r in rows:
                try:
                    age = now - datetime.strptime(r[1], "%Y-%m-%d %H:%M:%S")
                except ValueError:
                    keep.append(r)
                    continue
                status = r[2] or STATUS_PENDING
                if status == STATUS_CONFIRMED:
                    keep.append(r)
                elif status == STATUS_FALSE_POSITIVE and age > timedelta(days=false_positive_days):
                    self._delete_event(conn, r)
                    removed += 1
                elif status != STATUS_FALSE_POSITIVE and age > timedelta(days=pending_days):
                    self._delete_event(conn, r)
                    removed += 1
                else:
                    keep.append(r)

            # Limite de espaço: apaga os mais antigos não confirmados
            limit = max_gb * (1024 ** 3)
            sizes = {r[0]: self._size(self._files_of(r)) for r in keep}
            total = sum(sizes.values())
            for r in keep:  # já em ordem do mais antigo para o mais novo
                if total <= limit:
                    break
                if (r[2] or STATUS_PENDING) == STATUS_CONFIRMED:
                    continue
                total -= sizes[r[0]]
                self._delete_event(conn, r)
                removed += 1

        self._remove_empty_day_dirs()
        return {"removed": removed, "size_gb": round(total / (1024 ** 3), 3)}

    def _remove_empty_day_dirs(self):
        if not self.events_dir.exists():
            return
        for d in self.events_dir.iterdir():
            if d.is_dir():
                try:
                    next(d.iterdir())
                except StopIteration:
                    try:
                        d.rmdir()
                    except OSError:
                        pass
