from __future__ import annotations

import logging
import uuid
from datetime import datetime, timezone
from pathlib import Path

from app.annotation import BoundingBox, ImageAnnotation, LabelClass
from app.database import DatabaseManager

_log = logging.getLogger(__name__)


class AnnotationController:
    """Encapsula toda la lógica de persistencia en DuckDB relacionada con anotaciones,
    clases e imágenes. MainWindow delega aquí cualquier operación contra la BD."""

    def __init__(self, db: DatabaseManager, project_id: str):
        self._db = db
        self._project_id = project_id

    # ------------------------------------------------------------------ #
    # Proyecto seed
    # ------------------------------------------------------------------ #

    def seed_default_project(self) -> None:
        """Crea el proyecto 'animales' y sus clases YOLO si no existen. Idempotente."""
        now = datetime.now(timezone.utc)
        PROJECT_ID = "animales"

        existing = self._db.conn.execute(
            "SELECT id FROM projects WHERE id = ?", (PROJECT_ID,)
        ).fetchone()

        if existing is None:
            self._db.conn.execute(
                """
                INSERT INTO projects (id, name, description, base_path, created_at, updated_at)
                VALUES (?, ?, ?, ?, ?, ?)
                """,
                (PROJECT_ID, "animales", "Proyecto de clasificación de animales", "", now, now),
            )

        _DEFAULT_CLASSES = [
            ("perros",   0, "#f38ba8"),
            ("gatos",    1, "#a6e3a1"),
            ("caballos", 2, "#89b4fa"),
        ]
        for class_name, yolo_idx, color in _DEFAULT_CLASSES:
            exists = self._db.conn.execute(
                "SELECT id FROM classes WHERE project_id = ? AND yolo_index = ?",
                (PROJECT_ID, yolo_idx),
            ).fetchone()
            if exists is None:
                self._db.conn.execute(
                    "INSERT INTO classes (id, project_id, name, color, yolo_index) VALUES (?, ?, ?, ?, ?)",
                    (str(uuid.uuid4()), PROJECT_ID, class_name, color, yolo_idx),
                )

    # ------------------------------------------------------------------ #
    # Imágenes
    # ------------------------------------------------------------------ #

    def upsert_image(self, image_path: str, width: int, height: int) -> str:
        """Garantiza que la imagen esté registrada. Devuelve su UUID."""
        row = self._db.conn.execute(
            "SELECT id, status FROM images WHERE file_path = ? AND project_id = ?",
            (image_path, self._project_id),
        ).fetchone()

        if row is not None:
            if row[1] == "deleted":
                self._db.conn.execute(
                    """
                    UPDATE images
                    SET status = 'unannotated',
                        annotation_count = 0,
                        width = ?,
                        height = ?,
                        updated_at = ?
                    WHERE id = ?
                    """,
                    (width, height, datetime.now(timezone.utc), row[0]),
                )
            return row[0]

        new_id = str(uuid.uuid4())
        now = datetime.now(timezone.utc)
        try:
            self._db.conn.execute(
                """
                INSERT INTO images
                    (id, project_id, file_path, split, width, height,
                     status, annotation_count, created_at, updated_at)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                """,
                (new_id, self._project_id, image_path,
                 "unassigned", width, height, "unannotated", 0, now, now),
            )
            return new_id
        except Exception as e:
            msg = str(e).lower()
            if "duplicate key" in msg and "file_path" in msg:
                existing = self._db.conn.execute(
                    "SELECT id FROM images WHERE file_path = ? AND project_id = ?",
                    (image_path, self._project_id),
                ).fetchone()
                if existing is not None:
                    self._db.conn.execute(
                        "UPDATE images SET width = COALESCE(width, ?), height = COALESCE(height, ?), updated_at = ? WHERE id = ?",
                        (width, height, now, existing[0]),
                    )
                    return existing[0]

                other = self._db.conn.execute(
                    "SELECT id, project_id FROM images WHERE file_path = ?",
                    (image_path,),
                ).fetchone()
                if other is not None:
                    raise RuntimeError(
                        f"Conflicto de imagen: la ruta ya existe asociada a otro proyecto ('{other[1]}')."
                    ) from e
            raise

    def update_image_status(self, image_path: str, delta: int) -> None:
        """Actualiza annotation_count y status en la tabla images."""
        now = datetime.now(timezone.utc)
        try:
            self._db.conn.execute(
                """
                UPDATE images
                SET annotation_count = CASE
                        WHEN annotation_count + ? < 0 THEN 0
                        ELSE annotation_count + ?
                    END,
                    updated_at = ?
                WHERE file_path = ? AND project_id = ? AND status <> 'deleted'
                """,
                (delta, delta, now, image_path, self._project_id),
            )
            self._db.conn.execute(
                """
                UPDATE images
                SET status = CASE
                        WHEN annotation_count > 0 THEN 'annotated'
                        ELSE 'unannotated'
                    END
                WHERE file_path = ? AND project_id = ? AND status <> 'deleted'
                """,
                (image_path, self._project_id),
            )
        except Exception as e:
            _log.error("Error al actualizar estado en BD: %s", e, exc_info=e)
            raise

    def sync_image_annotation_summary(self, image_path: str) -> None:
        """Sincroniza contador/estado de una imagen a partir de sus annotations."""
        now = datetime.now(timezone.utc)
        self._db.conn.execute(
            """
            UPDATE images
            SET annotation_count = (
                    SELECT COUNT(a.id)
                    FROM annotations a
                    WHERE a.image_id = images.id
                ),
                status = CASE
                    WHEN (
                        SELECT COUNT(a.id)
                        FROM annotations a
                        WHERE a.image_id = images.id
                    ) > 0 THEN 'annotated'
                    ELSE 'unannotated'
                END,
                updated_at = ?
            WHERE file_path = ? AND project_id = ? AND status <> 'deleted'
            """,
            (now, image_path, self._project_id),
        )

    def sync_project_annotation_summaries(self) -> None:
        """Recalcula contador/estado de todas las imágenes activas del proyecto."""
        now = datetime.now(timezone.utc)
        self._db.conn.execute(
            """
            UPDATE images
            SET annotation_count = (
                    SELECT COUNT(a.id)
                    FROM annotations a
                    WHERE a.image_id = images.id
                ),
                status = CASE
                    WHEN (
                        SELECT COUNT(a.id)
                        FROM annotations a
                        WHERE a.image_id = images.id
                    ) > 0 THEN 'annotated'
                    ELSE 'unannotated'
                END,
                updated_at = ?
            WHERE project_id = ? AND status <> 'deleted'
            """,
            (now, self._project_id),
        )

    def update_split(self, image_path: str, split: str) -> None:
        """Persiste el split de una imagen."""
        self._db.conn.execute(
            """
            UPDATE images
            SET split = ?, updated_at = ?
            WHERE file_path = ? AND project_id = ? AND status <> 'deleted'
            """,
            (split, datetime.now(timezone.utc), image_path, self._project_id),
        )

    def load_annotation_for_image(self, image_path: str) -> ImageAnnotation | None:
        """Carga las anotaciones de una única imagen desde la BD. Devuelve None si no existe."""
        row = self._db.conn.execute(
            "SELECT id, width, height FROM images WHERE file_path = ? AND project_id = ? AND status <> 'deleted'",
            (image_path, self._project_id),
        ).fetchone()
        if not row or not row[1] or not row[2]:
            return None
        img_id, width, height = row

        class_rows = self._db.conn.execute(
            "SELECT id, name, color, yolo_index FROM classes WHERE project_id = ?",
            (self._project_id,),
        ).fetchall()
        class_map = {r[0]: LabelClass(name=r[1], color=r[2], class_id=r[3]) for r in class_rows}

        ann_rows = self._db.conn.execute(
            "SELECT id, class_id, x_center, y_center, width, height FROM annotations WHERE image_id = ?",
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
        return ImageAnnotation(image_path=image_path, image_width=width, image_height=height, boxes=boxes)

    def load_gallery_splits(self) -> dict[str, str]:
        """Devuelve {file_path: split} para las imágenes del proyecto."""
        rows = self._db.conn.execute(
            """
            SELECT file_path, split
            FROM images
            WHERE project_id = ? AND status <> 'deleted'
            """,
            (self._project_id,),
        ).fetchall()
        return {r[0]: r[1] for r in rows if r[1] and r[1] != "unassigned"}

    def delete_image_from_project(self, image_path: str) -> None:
        """Quita una imagen del proyecto sin borrar el archivo físico.

        Se borran sus anotaciones y se deja una marca persistente para que,
        al reabrir un proyecto basado en carpeta, la imagen no se reimporte
        automáticamente desde el disco.
        """
        now = datetime.now(timezone.utc)
        try:
            self._db.conn.execute("BEGIN TRANSACTION")
            self._db.conn.execute(
                """
                DELETE FROM annotations
                WHERE image_id IN (
                    SELECT id FROM images
                    WHERE file_path = ? AND project_id = ?
                )
                """,
                (image_path, self._project_id),
            )
            self._db.conn.execute(
                """
                UPDATE images
                SET status = 'deleted',
                    annotation_count = 0,
                    split = 'unassigned',
                    updated_at = ?
                WHERE file_path = ? AND project_id = ?
                """,
                (now, image_path, self._project_id),
            )
            self._db.conn.execute("COMMIT")
        except Exception:
            try:
                self._db.conn.execute("ROLLBACK")
            except Exception:
                pass
            raise

    # ------------------------------------------------------------------ #
    # Clases
    # ------------------------------------------------------------------ #

    def load_classes(self) -> list[LabelClass]:
        """Lee las clases del proyecto desde la BD ordenadas por yolo_index."""
        rows = self._db.conn.execute(
            "SELECT id, name, color, yolo_index FROM classes WHERE project_id = ? ORDER BY yolo_index",
            (self._project_id,),
        ).fetchall()
        return [LabelClass(name=name, color=color, class_id=yolo_idx) for _, name, color, yolo_idx in rows]

    def resolve_class_uuid(self, label_class: LabelClass) -> str | None:
        """Devuelve el UUID de BD para una clase por yolo_index, o None si no existe."""
        row = self._db.conn.execute(
            "SELECT id FROM classes WHERE project_id = ? AND yolo_index = ?",
            (self._project_id, label_class.class_id),
        ).fetchone()
        return row[0] if row else None

    def insert_class(self, name: str, color: str, yolo_idx: int) -> bool:
        """Inserta una nueva clase si no existe ya por nombre. Devuelve True si insertó."""
        existing = self._db.conn.execute(
            "SELECT id FROM classes WHERE project_id = ? AND name = ?",
            (self._project_id, name),
        ).fetchone()
        if existing is not None:
            return False
        self._db.conn.execute(
            "INSERT INTO classes (id, project_id, name, color, yolo_index) VALUES (?, ?, ?, ?, ?)",
            (str(uuid.uuid4()), self._project_id, name, color, yolo_idx),
        )
        return True

    def count_annotations_for_class(self, name: str) -> int:
        """Devuelve cuántos bboxes en la BD usan esta clase."""
        row = self._db.conn.execute(
            """
            SELECT COUNT(*) FROM annotations a
            JOIN classes c ON a.class_id = c.id
            WHERE c.project_id = ? AND c.name = ?
            """,
            (self._project_id, name),
        ).fetchone()
        return row[0] if row else 0

    def delete_class(self, name: str) -> None:
        """Elimina una clase y todas sus anotaciones en cascada."""
        # Obtener el UUID de la clase
        row = self._db.conn.execute(
            "SELECT id FROM classes WHERE project_id = ? AND name = ?",
            (self._project_id, name),
        ).fetchone()
        if row is None:
            return
        class_id = row[0]

        # Borrar anotaciones que usan esta clase
        self._db.conn.execute(
            "DELETE FROM annotations WHERE class_id = ?",
            (class_id,),
        )

        # Borrar la clase
        self._db.conn.execute(
            "DELETE FROM classes WHERE id = ?",
            (class_id,),
        )

    # ------------------------------------------------------------------ #
    # Anotaciones
    # ------------------------------------------------------------------ #

    def persist_annotation(self, annotation: ImageAnnotation) -> None:
        """Guarda en DuckDB el estado actual de anotaciones para una imagen."""
        if not annotation or not annotation.image_path:
            return

        image_db_id = self.upsert_image(
            annotation.image_path, annotation.image_width, annotation.image_height
        )

        boxes_for_db: list[BoundingBox] = []
        for box in annotation.boxes:
            cls_row = self._db.conn.execute(
                "SELECT id FROM classes WHERE project_id = ? AND yolo_index = ?",
                (self._project_id, box.label_class.class_id),
            ).fetchone()
            if cls_row is None:
                continue
            boxes_for_db.append(
                BoundingBox(
                    x=box.x, y=box.y,
                    width=box.width, height=box.height,
                    label_class=LabelClass(
                        name=box.label_class.name,
                        color=box.label_class.color,
                        class_id=cls_row[0],
                    ),
                    id=box.id,
                )
            )

        annotation_for_db = ImageAnnotation(
            image_path=annotation.image_path,
            image_width=annotation.image_width,
            image_height=annotation.image_height,
            boxes=boxes_for_db,
        )
        self._db.save_image_annotation(
            image_id=image_db_id,
            annotation=annotation_for_db,
            replace=False,
        )

    def delete_annotation(self, box_id: str) -> None:
        """Elimina un bbox de la BD por su UUID."""
        self._db.conn.execute("DELETE FROM annotations WHERE id = ?", (box_id,))

    def clear_annotations(self, image_path: str, box_ids: list[str]) -> None:
        """Elimina todos los bboxes de una imagen en la BD."""
        for box_id in box_ids:
            self._db.conn.execute("DELETE FROM annotations WHERE id = ?", (box_id,))
        self._db.conn.execute(
            """
            UPDATE images
            SET annotation_count = 0, status = 'unannotated', updated_at = ?
            WHERE file_path = ? AND project_id = ? AND status <> 'deleted'
            """,
            (datetime.now(timezone.utc), image_path, self._project_id),
        )
