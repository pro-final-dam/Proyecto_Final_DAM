from __future__ import annotations
import os
from pathlib import Path
from app.annotation import ImageAnnotation


class YoloExporter:

    @staticmethod
    def export(annotation: ImageAnnotation) -> str:
        """
        Guarda el archivo .txt YOLO junto a la imagen.
        Devuelve la ruta del archivo generado.
        """
        if not annotation.boxes:
            raise ValueError("No hay anotaciones que exportar.")

        image_path = Path(annotation.image_path)
        output_path = image_path.with_suffix(".txt")

        lines = annotation.to_yolo_lines()
        output_path.write_text("\n".join(lines) + "\n", encoding="utf-8")

        return str(output_path)

    @staticmethod
    def export_classes(annotation: ImageAnnotation, output_dir: str | None = None) -> str:
        """
        Guarda classes.txt con los nombres de clase en orden de class_id.
        """
        if not annotation.boxes:
            raise ValueError("No hay anotaciones.")

        classes: dict[int, str] = {}
        for box in annotation.boxes:
            classes[box.label_class.class_id] = box.label_class.name

        lines = [classes[k] for k in sorted(classes)]

        if output_dir is None:
            output_dir = str(Path(annotation.image_path).parent)

        out_path = Path(output_dir) / "classes.txt"
        out_path.write_text("\n".join(lines) + "\n", encoding="utf-8")

        return str(out_path)
