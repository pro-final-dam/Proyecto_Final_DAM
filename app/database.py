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

import json
import os
import uuid
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
    # Resuelve siempre como ruta absoluta para evitar ambigüedad según CWD
    abs_path = Path(db_path)
    if not abs_path.is_absolute():
        abs_path = (_ENV_PATH.parent / db_path).resolve()
    abs_path.parent.mkdir(parents=True, exist_ok=True)
    return str(abs_path)


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
    id               VARCHAR PRIMARY KEY,
    project_id       VARCHAR NOT NULL REFERENCES projects(id),
    file_path        VARCHAR NOT NULL,
    split            VARCHAR DEFAULT 'unassigned',
    width            INTEGER,
    height           INTEGER,
    status           VARCHAR DEFAULT 'unannotated',
    annotation_count INTEGER DEFAULT 0,
    created_at       TIMESTAMP NOT NULL,
    updated_at       TIMESTAMP NOT NULL,
    UNIQUE (project_id, file_path)
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
    run_dir          VARCHAR,
    config_path      VARCHAR,
    config_json      VARCHAR,
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
        self._migrate()

    def _migrate(self) -> None:
        """Añade columnas nuevas y corrige constraints en tablas existentes."""
        existing_cols = {
            row[0]
            for row in self.conn.execute(
                "SELECT column_name FROM information_schema.columns WHERE table_name = 'images'"
            ).fetchall()
        }
        if 'annotation_count' not in existing_cols:
            self.conn.execute(
                "ALTER TABLE images ADD COLUMN annotation_count INTEGER DEFAULT 0"
            )
        if 'updated_at' not in existing_cols:
            self.conn.execute(
                "ALTER TABLE images ADD COLUMN updated_at TIMESTAMP"
            )

        training_cols = {
            row[0]
            for row in self.conn.execute(
                "SELECT column_name FROM information_schema.columns WHERE table_name = 'trainings'"
            ).fetchall()
        }
        for col_name in ("run_dir", "config_path", "config_json"):
            if col_name not in training_cols:
                self.conn.execute(f"ALTER TABLE trainings ADD COLUMN {col_name} VARCHAR")

        # Eliminar el UNIQUE(file_path) incorrecto si existe — la misma imagen
        # puede pertenecer a varios proyectos, el constraint correcto es (project_id, file_path)
        try:
            self.conn.execute("ALTER TABLE images DROP CONSTRAINT images_file_path_key")
        except Exception:
            pass

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
            "SELECT * FROM images WHERE project_id = ? AND status <> 'deleted'",
            (project_id,),
        ).fetchall()
        columns = [
            "id", "project_id", "file_path", "split",
            "width", "height", "status", "annotation_count",
            "created_at", "updated_at",
        ]
        return [dict(zip(columns, row)) for row in rows]

    # ------------------------------------------------------------------
    # Proyectos
    # ------------------------------------------------------------------

    def create_project(self, name: str, base_path: str, description: str = "") -> str:
        """Crea un proyecto nuevo y devuelve su ID."""
        import uuid as _uuid
        project_id = str(_uuid.uuid4())
        now = datetime.now(timezone.utc)
        self.conn.execute(
            """
            INSERT INTO projects (id, name, description, base_path, created_at, updated_at)
            VALUES (?, ?, ?, ?, ?, ?)
            """,
            (project_id, name, description, base_path, now, now),
        )
        return project_id

    def rename_project(self, project_id: str, new_name: str) -> None:
        """Renombra un proyecto existente."""
        now = datetime.now(timezone.utc)
        self.conn.execute(
            "UPDATE projects SET name = ?, updated_at = ? WHERE id = ?",
            (new_name, now, project_id),
        )

    def create_training_run(
        self,
        *,
        training_id: str,
        project_id: str,
        model_name: str,
        dataset_yaml_path: str,
        epochs: int,
        image_size: int,
        batch_size: int,
        status: str = "running",
        run_dir: str = "",
        config_path: str = "",
        config_json: str = "",
    ) -> None:
        """Crea un registro de entrenamiento en la tabla trainings."""
        now = datetime.now(timezone.utc)
        self.conn.execute(
            """
            INSERT INTO trainings
                (id, project_id, model_name, dataset_yaml_path, epochs, image_size,
                 batch_size, status, run_dir, config_path, config_json, started_at)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """,
            (
                training_id,
                project_id,
                model_name,
                dataset_yaml_path,
                epochs,
                image_size,
                batch_size,
                status,
                run_dir,
                config_path,
                config_json,
                now,
            ),
        )

    def update_training_run(
        self,
        *,
        training_id: str,
        status: str,
        metrics_path: str = "",
        weights_path: str = "",
        run_dir: str = "",
        config_path: str = "",
        config_json: str = "",
    ) -> None:
        """Actualiza el estado final de un entrenamiento."""
        now = datetime.now(timezone.utc)
        self.conn.execute(
            """
            UPDATE trainings
            SET status = ?,
                metrics_path = ?,
                weights_path = ?,
                run_dir = COALESCE(NULLIF(?, ''), run_dir),
                config_path = COALESCE(NULLIF(?, ''), config_path),
                config_json = COALESCE(NULLIF(?, ''), config_json),
                finished_at = ?
            WHERE id = ?
            """,
            (status, metrics_path, weights_path, run_dir, config_path, config_json, now, training_id),
        )

    def delete_training_run(self, training_id: str, project_id: str) -> int:
        """Elimina una version de entrenamiento del proyecto."""
        row = self.conn.execute(
            "SELECT COUNT(*) FROM trainings WHERE id = ? AND project_id = ?",
            (training_id, project_id),
        ).fetchone()
        count = row[0] if row else 0
        self.conn.execute(
            "DELETE FROM trainings WHERE id = ? AND project_id = ?",
            (training_id, project_id),
        )
        return count

    def get_all_projects_with_stats(self) -> list[dict]:
        """Devuelve todos los proyectos con contadores de imágenes y anotaciones."""
        rows = self.conn.execute(
            """
            SELECT
                p.id, p.name, p.description, p.base_path,
                p.created_at, p.updated_at,
                COUNT(DISTINCT i.id)                          AS image_count,
                COALESCE(SUM(i.annotation_count), 0)          AS annotation_count,
                MAX(i.updated_at)                             AS last_activity,
                MIN(i.file_path)                              AS sample_image
            FROM projects p
            LEFT JOIN images i ON i.project_id = p.id AND i.status <> 'deleted'
            GROUP BY p.id, p.name, p.description, p.base_path, p.created_at, p.updated_at
            ORDER BY COALESCE(MAX(i.updated_at), p.updated_at) DESC
            """
        ).fetchall()
        keys = [
            "id", "name", "description", "base_path",
            "created_at", "updated_at",
            "image_count", "annotation_count", "last_activity", "sample_image",
        ]
        return [dict(zip(keys, r)) for r in rows]


    @staticmethod
    def _json_value(value):
        if isinstance(value, datetime):
            return value.isoformat()
        return value

    def _rows_as_dicts(self, query: str, params: tuple = ()) -> list[dict]:
        cursor = self.conn.execute(query, params)
        columns = [desc[0] for desc in cursor.description]
        return [
            {col: self._json_value(value) for col, value in zip(columns, row)}
            for row in cursor.fetchall()
        ]

    def export_project_backup(self, project_id: str, output_path: str) -> dict:
        """Exporta un backup JSON con los datos de BD de un proyecto."""
        project_rows = self._rows_as_dicts(
            "SELECT * FROM projects WHERE id = ?",
            (project_id,),
        )
        if not project_rows:
            raise ValueError("El proyecto no existe en la base de datos.")

        data = {
            "format": "visionhub_project_backup",
            "version": 1,
            "exported_at": datetime.now(timezone.utc).isoformat(),
            "project": project_rows[0],
            "classes": self._rows_as_dicts(
                "SELECT * FROM classes WHERE project_id = ? ORDER BY yolo_index",
                (project_id,),
            ),
            "images": self._rows_as_dicts(
                "SELECT * FROM images WHERE project_id = ? ORDER BY file_path",
                (project_id,),
            ),
            "annotations": self._rows_as_dicts(
                """
                SELECT a.*
                FROM annotations a
                WHERE a.image_id IN (SELECT id FROM images WHERE project_id = ?)
                   OR a.class_id IN (SELECT id FROM classes WHERE project_id = ?)
                ORDER BY a.created_at, a.id
                """,
                (project_id, project_id),
            ),
            "trainings": self._rows_as_dicts(
                "SELECT * FROM trainings WHERE project_id = ? ORDER BY started_at",
                (project_id,),
            ),
        }

        out = Path(output_path)
        out.parent.mkdir(parents=True, exist_ok=True)
        out.write_text(json.dumps(data, indent=2, ensure_ascii=False), encoding="utf-8")

        return {
            "classes": len(data["classes"]),
            "images": len(data["images"]),
            "annotations": len(data["annotations"]),
            "trainings": len(data["trainings"]),
            "path": str(out),
        }

    def import_project_backup(self, backup_path: str) -> dict:
        """Importa un backup JSON como un proyecto nuevo y remapea sus IDs."""
        data = json.loads(Path(backup_path).read_text(encoding="utf-8"))
        if data.get("format") != "visionhub_project_backup":
            raise ValueError("El archivo no es un backup valido de VisionHub.")

        project = data.get("project") or {}
        now = datetime.now(timezone.utc)
        new_project_id = str(uuid.uuid4())
        original_name = project.get("name") or "Proyecto importado"
        imported_name = f"{original_name} (backup)"

        class_id_map: dict[str, str] = {}
        image_id_map: dict[str, str] = {}

        self.conn.execute(
            """
            INSERT INTO projects (id, name, description, base_path, created_at, updated_at)
            VALUES (?, ?, ?, ?, ?, ?)
            """,
            (
                new_project_id,
                imported_name,
                project.get("description", ""),
                project.get("base_path", ""),
                now,
                now,
            ),
        )

        for cls in data.get("classes", []):
            new_class_id = str(uuid.uuid4())
            class_id_map[cls["id"]] = new_class_id
            self.conn.execute(
                """
                INSERT INTO classes (id, project_id, name, color, yolo_index)
                VALUES (?, ?, ?, ?, ?)
                """,
                (
                    new_class_id,
                    new_project_id,
                    cls.get("name", ""),
                    cls.get("color", "#89b4fa"),
                    cls.get("yolo_index", 0),
                ),
            )

        for image in data.get("images", []):
            new_image_id = str(uuid.uuid4())
            image_id_map[image["id"]] = new_image_id
            self.conn.execute(
                """
                INSERT INTO images
                    (id, project_id, file_path, split, width, height,
                     status, annotation_count, created_at, updated_at)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    new_image_id,
                    new_project_id,
                    image.get("file_path", ""),
                    image.get("split", "unassigned"),
                    image.get("width"),
                    image.get("height"),
                    image.get("status", "unannotated"),
                    image.get("annotation_count", 0),
                    image.get("created_at") or now,
                    image.get("updated_at") or now,
                ),
            )

        for ann in data.get("annotations", []):
            old_image_id = ann.get("image_id")
            old_class_id = ann.get("class_id")
            if old_image_id not in image_id_map or old_class_id not in class_id_map:
                continue
            self.conn.execute(
                """
                INSERT INTO annotations
                    (id, image_id, class_id, x_center, y_center,
                     width, height, format, created_at, updated_at)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    str(uuid.uuid4()),
                    image_id_map[old_image_id],
                    class_id_map[old_class_id],
                    ann.get("x_center", 0),
                    ann.get("y_center", 0),
                    ann.get("width", 0),
                    ann.get("height", 0),
                    ann.get("format", "yolo_normalized"),
                    ann.get("created_at") or now,
                    ann.get("updated_at") or now,
                ),
            )

        for training in data.get("trainings", []):
            self.conn.execute(
                """
                INSERT INTO trainings
                    (id, project_id, model_name, dataset_yaml_path, epochs,
                     image_size, batch_size, status, metrics_path, weights_path,
                     run_dir, config_path, config_json, started_at, finished_at)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    str(uuid.uuid4()),
                    new_project_id,
                    training.get("model_name", ""),
                    training.get("dataset_yaml_path", ""),
                    training.get("epochs"),
                    training.get("image_size"),
                    training.get("batch_size"),
                    training.get("status", "pending"),
                    training.get("metrics_path", ""),
                    training.get("weights_path", ""),
                    training.get("run_dir", ""),
                    training.get("config_path", ""),
                    training.get("config_json", ""),
                    training.get("started_at"),
                    training.get("finished_at"),
                ),
            )

        annotation_count = self.conn.execute(
            """
            SELECT COUNT(*)
            FROM annotations a
            JOIN images i ON a.image_id = i.id
            WHERE i.project_id = ?
            """,
            (new_project_id,),
        ).fetchone()[0]

        return {
            "project_id": new_project_id,
            "name": imported_name,
            "base_path": project.get("base_path", ""),
            "classes": len(class_id_map),
            "images": len(image_id_map),
            "annotations": annotation_count,
            "trainings": len(data.get("trainings", [])),
        }

    def delete_project_cascade(self, project_id: str) -> dict:
        """Elimina un proyecto y todos sus datos asociados.

        Borra en cascada lógica:
        - annotations (de imágenes del proyecto y clases del proyecto)
        - trainings del proyecto
        - images del proyecto
        - classes del proyecto
        - projects (fila del proyecto)

        Returns:
            Resumen con contadores de filas eliminadas.
        """
        summary = {
            "annotations": 0,
            "trainings": 0,
            "images": 0,
            "classes": 0,
            "projects": 0,
        }

        try:
            summary["annotations"] = self.conn.execute(
                """
                SELECT COUNT(*)
                FROM annotations a
                WHERE a.image_id IN (SELECT id FROM images WHERE project_id = ?)
                   OR a.class_id IN (SELECT id FROM classes WHERE project_id = ?)
                """,
                (project_id, project_id),
            ).fetchone()[0]
            summary["trainings"] = self.conn.execute(
                "SELECT COUNT(*) FROM trainings WHERE project_id = ?",
                (project_id,),
            ).fetchone()[0]
            summary["images"] = self.conn.execute(
                "SELECT COUNT(*) FROM images WHERE project_id = ?",
                (project_id,),
            ).fetchone()[0]
            summary["classes"] = self.conn.execute(
                "SELECT COUNT(*) FROM classes WHERE project_id = ?",
                (project_id,),
            ).fetchone()[0]
            summary["projects"] = self.conn.execute(
                "SELECT COUNT(*) FROM projects WHERE id = ?",
                (project_id,),
            ).fetchone()[0]

            # IDs del proyecto a borrar (reutilizados en toda la cascada)
            image_ids = [
                r[0]
                for r in self.conn.execute(
                    "SELECT id FROM images WHERE project_id = ?",
                    (project_id,),
                ).fetchall()
            ]
            class_ids = [
                r[0]
                for r in self.conn.execute(
                    "SELECT id FROM classes WHERE project_id = ?",
                    (project_id,),
                ).fetchall()
            ]

            # 1) Limpieza directa de tablas conocidas
            self.conn.execute(
                """
                DELETE FROM annotations
                WHERE image_id IN (SELECT id FROM images WHERE project_id = ?)
                   OR class_id IN (SELECT id FROM classes WHERE project_id = ?)
                """,
                (project_id, project_id),
            )

            # DuckDB comprueba algunas FKs contra el estado confirmado. Ejecutar
            # la cascada por fases en autocommit evita falsos bloqueos al borrar
            # padres justo después de borrar hijos.

            # 2) Limpieza defensiva por FKs reales del esquema.
            #    Si existen tablas adicionales que referencian images/classes/projects,
            #    se limpian automáticamente antes de borrar los padres.
            try:
                fk_rows = self.conn.execute(
                    """
                    SELECT
                        tc.table_name AS child_table,
                        kcu.column_name AS child_column,
                        ccu.table_name AS parent_table,
                        ccu.column_name AS parent_column
                    FROM information_schema.table_constraints tc
                    JOIN information_schema.key_column_usage kcu
                      ON tc.constraint_name = kcu.constraint_name
                     AND tc.table_name = kcu.table_name
                    JOIN information_schema.constraint_column_usage ccu
                      ON tc.constraint_name = ccu.constraint_name
                    WHERE tc.constraint_type = 'FOREIGN KEY'
                      AND ccu.table_name IN ('images', 'classes', 'projects')
                    ORDER BY child_table
                    """
                ).fetchall()

                for child_table, child_column, parent_table, parent_column in fk_rows:
                    if child_table in ("annotations", "trainings", "images", "classes", "projects"):
                        continue

                    if parent_table == "images" and parent_column == "id":
                        self.conn.execute(
                            f"""
                            DELETE FROM {child_table}
                            WHERE {child_column} IN (
                                SELECT id FROM images WHERE project_id = ?
                            )
                            """,
                            (project_id,),
                        )
                    elif parent_table == "classes" and parent_column == "id":
                        self.conn.execute(
                            f"""
                            DELETE FROM {child_table}
                            WHERE {child_column} IN (
                                SELECT id FROM classes WHERE project_id = ?
                            )
                            """,
                            (project_id,),
                        )
                    elif parent_table == "projects" and parent_column == "id":
                        self.conn.execute(
                            f"DELETE FROM {child_table} WHERE {child_column} = ?",
                            (project_id,),
                        )
            except Exception:
                # Compatibilidad con versiones de DuckDB sin information_schema.table_constraints.
                image_ref_tables = [
                    r[0]
                    for r in self.conn.execute(
                        """
                        SELECT DISTINCT table_name
                        FROM information_schema.columns
                        WHERE column_name = 'image_id'
                          AND table_name NOT IN ('annotations', 'images', 'classes', 'projects', 'trainings')
                        """
                    ).fetchall()
                ]
                for table_name in image_ref_tables:
                    self.conn.execute(
                        f"DELETE FROM {table_name} WHERE image_id IN (SELECT id FROM images WHERE project_id = ?)",
                        (project_id,),
                    )

                class_ref_tables = [
                    r[0]
                    for r in self.conn.execute(
                        """
                        SELECT DISTINCT table_name
                        FROM information_schema.columns
                        WHERE column_name = 'class_id'
                          AND table_name NOT IN ('annotations', 'images', 'classes', 'projects', 'trainings')
                        """
                    ).fetchall()
                ]
                for table_name in class_ref_tables:
                    self.conn.execute(
                        f"DELETE FROM {table_name} WHERE class_id IN (SELECT id FROM classes WHERE project_id = ?)",
                        (project_id,),
                    )

                project_ref_tables = [
                    r[0]
                    for r in self.conn.execute(
                        """
                        SELECT DISTINCT table_name
                        FROM information_schema.columns
                        WHERE column_name = 'project_id'
                          AND table_name NOT IN ('images', 'classes', 'projects', 'trainings')
                        """
                    ).fetchall()
                ]
                for table_name in project_ref_tables:
                    self.conn.execute(
                        f"DELETE FROM {table_name} WHERE project_id = ?",
                        (project_id,),
                    )

            # 3) Último barrido defensivo:
            #    elimina referencias por columnas image_id/class_id en cualquier tabla
            #    visible del esquema (excepto tablas base ya tratadas).
            #    Esto cubre esquemas legacy o tablas auxiliares no contempladas.
            tables = [
                r[0]
                for r in self.conn.execute(
                    """
                    SELECT table_name
                    FROM information_schema.tables
                    WHERE table_schema = 'main'
                    """
                ).fetchall()
            ]
            excluded = {"images", "classes", "projects", "annotations", "trainings"}
            target_tables = [t for t in tables if t not in excluded]

            for table_name in target_tables:
                cols = {
                    r[0]
                    for r in self.conn.execute(
                        """
                        SELECT column_name
                        FROM information_schema.columns
                        WHERE table_name = ?
                        """,
                        (table_name,),
                    ).fetchall()
                }
                if "image_id" in cols and image_ids:
                    for img_id in image_ids:
                        self.conn.execute(
                            f"DELETE FROM {table_name} WHERE image_id = ?",
                            (img_id,),
                        )
                if "class_id" in cols and class_ids:
                    for cls_id in class_ids:
                        self.conn.execute(
                            f"DELETE FROM {table_name} WHERE class_id = ?",
                            (cls_id,),
                        )
                if "project_id" in cols:
                    self.conn.execute(
                        f"DELETE FROM {table_name} WHERE project_id = ?",
                        (project_id,),
                    )

            self.conn.execute("DELETE FROM trainings WHERE project_id = ?", (project_id,))
            self.conn.execute("DELETE FROM images WHERE project_id = ?", (project_id,))
            self.conn.execute("DELETE FROM classes WHERE project_id = ?", (project_id,))
            self.conn.execute("DELETE FROM projects WHERE id = ?", (project_id,))
        except Exception:
            raise

        return summary

    def load_annotations_for_project(self, project_id: str) -> dict:
        """Reconstruye los ImageAnnotation de un proyecto desde la BD.

        Returns:
            dict[image_path, ImageAnnotation]
        """
        from app.annotation import BoundingBox, ImageAnnotation, LabelClass

        class_rows = self.conn.execute(
            "SELECT id, name, color, yolo_index FROM classes WHERE project_id = ?",
            (project_id,),
        ).fetchall()
        class_map = {r[0]: LabelClass(name=r[1], color=r[2], class_id=r[3]) for r in class_rows}

        image_rows = self.conn.execute(
            """
            SELECT id, file_path, width, height
            FROM images
            WHERE project_id = ? AND status <> 'deleted'
            """,
            (project_id,),
        ).fetchall()

        result: dict = {}
        for img_id, file_path, width, height in image_rows:
            if not width or not height:
                continue
            ann_rows = self.conn.execute(
                """
                SELECT id, class_id, x_center, y_center, width, height
                FROM annotations WHERE image_id = ?
                """,
                (img_id,),
            ).fetchall()
            boxes = []
            for ann_id, class_id, xc, yc, bw, bh in ann_rows:
                cls = class_map.get(class_id)
                if not cls:
                    continue
                boxes.append(BoundingBox(
                    x=(xc - bw / 2) * width,
                    y=(yc - bh / 2) * height,
                    width=bw * width,
                    height=bh * height,
                    label_class=cls,
                    id=ann_id,
                ))
            if boxes:
                result[file_path] = ImageAnnotation(
                    image_path=file_path,
                    image_width=width,
                    image_height=height,
                    boxes=boxes,
                )
        return result
