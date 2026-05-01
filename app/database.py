"""
database.py — VisionHub
=======================
Gestiona la persistencia de datos del proyecto en DuckDB.

Uso básico:
    from app.database import DatabaseManager
    db = DatabaseManager()
    db.save_image_annotation(image_id="...", annotation=image_annotation_obj)
"""

from __future__ import annotations

import os
from datetime import datetime, timezone
from pathlib import Path

import duckdb
from dotenv import load_dotenv

from app.annotation import BoundingBox, ImageAnnotation, LabelClass  # noqa: F401

# ---------------------------------------------------------------------------
# Carga de variables de entorno
# ---------------------------------------------------------------------------

# Busca el .env desde la raíz del proyecto (un nivel por encima de /app)
_ENV_PATH = Path(__file__).resolve().parent.parent / ".env"
load_dotenv(dotenv_path=_ENV_PATH)


def _get_db_path() -> str:
    """Devuelve la ruta del archivo .db desde la variable de entorno DB_PATH.

    Raises:
        EnvironmentError: Si DB_PATH no está definida en el entorno.
    """
    db_path = os.getenv("DB_PATH")
    if not db_path:
        raise EnvironmentError(
            "La variable de entorno DB_PATH no está definida. "
            "Añádela en el archivo .env o en el entorno del sistema."
        )
    # Asegura que el directorio destino exista
    Path(db_path).parent.mkdir(parents=True, exist_ok=True)
    return db_path


# ---------------------------------------------------------------------------
# SQL de creación de tablas
# ---------------------------------------------------------------------------

_DDL_PROJECTS = """
CREATE TABLE IF NOT EXISTS projects (
    id          VARCHAR PRIMARY KEY,
    name        VARCHAR NOT NULL,
    description VARCHAR,
    base_path   VARCHAR NOT NULL,
    created_at  TIMESTAMP NOT NULL,
    updated_at  TIMESTAMP NOT NULL
);
"""

_DDL_CLASSES = """
CREATE TABLE IF NOT EXISTS classes (
    id          VARCHAR PRIMARY KEY,
    project_id  VARCHAR NOT NULL REFERENCES projects(id),
    name        VARCHAR NOT NULL,
    color       VARCHAR NOT NULL,
    yolo_index  INTEGER NOT NULL
);
"""

_DDL_IMAGES = """
CREATE TABLE IF NOT EXISTS images (
    id          VARCHAR PRIMARY KEY,
    project_id  VARCHAR NOT NULL REFERENCES projects(id),
    file_path   VARCHAR NOT NULL,
    split       VARCHAR DEFAULT 'train',
    width       INTEGER,
    height      INTEGER,
    status      VARCHAR DEFAULT 'unannotated',
    created_at  TIMESTAMP NOT NULL
);
"""

_DDL_ANNOTATIONS = """
CREATE TABLE IF NOT EXISTS annotations (
    id          VARCHAR PRIMARY KEY,
    image_id    VARCHAR NOT NULL REFERENCES images(id),
    class_id    VARCHAR NOT NULL REFERENCES classes(id),
    x_center    DOUBLE NOT NULL,
    y_center    DOUBLE NOT NULL,
    width       DOUBLE NOT NULL,
    height      DOUBLE NOT NULL,
    format      VARCHAR NOT NULL DEFAULT 'yolo_normalized',
    created_at  TIMESTAMP NOT NULL,
    updated_at  TIMESTAMP NOT NULL
);
"""

_DDL_TRAININGS = """
CREATE TABLE IF NOT EXISTS trainings (
    id               VARCHAR PRIMARY KEY,
    project_id       VARCHAR NOT NULL REFERENCES projects(id),
    model_name       VARCHAR NOT NULL,
    dataset_yaml_path VARCHAR,
    epochs           INTEGER,
    image_size       INTEGER,
    batch_size       INTEGER,
    status           VARCHAR DEFAULT 'pending',
    metrics_path     VARCHAR,
    weights_path     VARCHAR,
    started_at       TIMESTAMP,
    finished_at      TIMESTAMP
);
"""


# ---------------------------------------------------------------------------
# DatabaseManager
# ---------------------------------------------------------------------------

class DatabaseManager:
    """Gestiona la conexión y las operaciones CRUD sobre la base de datos DuckDB.

    Attributes:
        db_path (str): Ruta al archivo .db de DuckDB.
        _conn (duckdb.DuckDBPyConnection | None): Conexión activa (lazy).
    """

    def __init__(self, db_path: str | None = None) -> None:
        """Inicializa el manager.

        Args:
            db_path: Ruta explícita al archivo .db. Si es None, se usa DB_PATH
                     del entorno (recomendado).
        """
        self.db_path: str = db_path if db_path is not None else _get_db_path()
        self._conn: duckdb.DuckDBPyConnection | None = None

    # ------------------------------------------------------------------
    # Gestión de la conexión
    # ------------------------------------------------------------------

    @property
    def conn(self) -> duckdb.DuckDBPyConnection:
        """Devuelve la conexión activa, abriéndola si es necesario (lazy init)."""
        if self._conn is None:
            self._conn = duckdb.connect(self.db_path)
        return self._conn

    def close(self) -> None:
        """Cierra la conexión a la base de datos."""
        if self._conn is not None:
            self._conn.close()
            self._conn = None

    def __enter__(self) -> "DatabaseManager":
        """Soporte para uso como context manager (with DatabaseManager() as db:)."""
        return self

    def __exit__(self, exc_type, exc_val, exc_tb) -> None:
        self.close()

    # ------------------------------------------------------------------
    # Inicialización del esquema
    # ------------------------------------------------------------------

    def initialize_tables(self) -> None:
        """Crea todas las tablas del esquema si aún no existen.

        Es seguro llamarlo múltiples veces (idempotente).
        El orden de creación respeta las dependencias de claves foráneas.
        """
        for ddl in (
            _DDL_PROJECTS,
            _DDL_CLASSES,
            _DDL_IMAGES,
            _DDL_ANNOTATIONS,
            _DDL_TRAININGS,
        ):
            self.conn.execute(ddl)

    # ------------------------------------------------------------------
    # Operaciones de escritura — ImageAnnotation
    # ------------------------------------------------------------------

    def save_image_annotation(
        self,
        image_id: str,
        annotation: ImageAnnotation,
        *,
        replace: bool = False,
    ) -> int:
        """Persiste todos los BoundingBox de un ImageAnnotation en la BD.

        Cada BoundingBox se guarda como una fila en la tabla ``annotations``.
        Las coordenadas se normalizan al formato YOLO (x_center, y_center,
        width, height relativos al tamaño de la imagen) antes de guardar.

        Args:
            image_id:   ID de la fila correspondiente en la tabla ``images``.
            annotation: Objeto ``ImageAnnotation`` con los bounding boxes.
            replace:    Si es True, elimina las anotaciones previas de esa imagen
                        antes de insertar las nuevas.

        Returns:
            Número de anotaciones insertadas.

        Raises:
            ValueError: Si la imagen no tiene dimensiones válidas (≤ 0).
        """
        if annotation.image_width <= 0 or annotation.image_height <= 0:
            raise ValueError(
                f"Las dimensiones de la imagen deben ser positivas; "
                f"se recibió {annotation.image_width}×{annotation.image_height}."
            )

        if replace:
            # Parámetro ? para evitar inyecciones SQL
            self.conn.execute(
                "DELETE FROM annotations WHERE image_id = ?",
                (image_id,),
            )

        now = datetime.now(timezone.utc)
        inserted = 0

        for box in annotation.boxes:
            # Calcula coordenadas normalizadas (formato YOLO)
            x_center = (box.x + box.width / 2) / annotation.image_width
            y_center = (box.y + box.height / 2) / annotation.image_height
            w_norm = box.width / annotation.image_width
            h_norm = box.height / annotation.image_height

            # La clase de la BD se referencia por su ID (UUID).
            # LabelClass.class_id es el índice YOLO (int), pero para la FK
            # necesitamos el UUID de la fila en 'classes'. Aquí usamos el
            # campo `box.label_class.class_id` directamente como placeholder;
            # el integrador debe garantizar que coincide con un ID en classes.
            class_db_id = str(box.label_class.class_id)

            # Si no es un reemplazo total, hacemos un "upsert" eliminando primero el ID específico
            if not replace:
                self.conn.execute("DELETE FROM annotations WHERE id = ?", (box.id,))

            self.conn.execute(
                """
                INSERT INTO annotations
                    (id, image_id, class_id, x_center, y_center,
                     width, height, format, created_at, updated_at)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    box.id,          # id          — UUID generado en BoundingBox
                    image_id,        # image_id    — FK a images
                    class_db_id,     # class_id    — FK a classes
                    x_center,        # x_center    — normalizado
                    y_center,        # y_center    — normalizado
                    w_norm,          # width       — normalizado
                    h_norm,          # height      — normalizado
                    "yolo_normalized",  # format
                    now,             # created_at
                    now,             # updated_at
                ),
            )
            inserted += 1

        return inserted

    # ------------------------------------------------------------------
    # Helpers de consulta (lectura)
    # ------------------------------------------------------------------

    def get_annotations_for_image(self, image_id: str) -> list[dict]:
        """Devuelve todas las anotaciones de una imagen como lista de dicts.

        Args:
            image_id: ID de la imagen a consultar.

        Returns:
            Lista de dicts con las columnas de la tabla annotations.
        """
        rows = self.conn.execute(
            "SELECT * FROM annotations WHERE image_id = ?",
            (image_id,),
        ).fetchall()
        columns = [
            "id", "image_id", "class_id", "x_center", "y_center",
            "width", "height", "format", "created_at", "updated_at",
        ]
        return [dict(zip(columns, row)) for row in rows]

    def get_images_for_project(self, project_id: str) -> list[dict]:
        """Devuelve todas las imágenes de un proyecto como lista de dicts.

        Args:
            project_id: ID del proyecto a consultar.

        Returns:
            Lista de dicts con las columnas de la tabla images.
        """
        rows = self.conn.execute(
            "SELECT * FROM images WHERE project_id = ?",
            (project_id,),
        ).fetchall()
        columns = [
            "id", "project_id", "file_path", "split",
            "width", "height", "status", "created_at",
        ]
        return [dict(zip(columns, row)) for row in rows]
