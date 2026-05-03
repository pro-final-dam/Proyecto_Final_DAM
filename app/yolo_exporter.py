from __future__ import annotations
import os
import random
import shutil
from pathlib import Path
from app.annotation import ImageAnnotation, LabelClass


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

    @staticmethod
    def export_dataset(
        annotations: dict[str, ImageAnnotation],
        classes: list[LabelClass],
        output_dir: str,
        splits: dict[str, str] | None = None,
        val_ratio: float = 0.2,
    ) -> tuple[str, dict[str, str]]:
        """Genera la estructura de dataset completa para YOLOv8/YOLO11.

        Crea images/train, images/val, labels/train, labels/val y data.yaml.

        Args:
            splits:    Mapa image_path → 'train'|'val'|'unassigned' desde la BD.
                       Las imágenes 'unassigned' se distribuyen automáticamente.
        Returns:
            (yaml_path, final_splits) — yaml generado y splits definitivos usados.
        """
        dataset_images = [
            p for p, ann in annotations.items()
            if ann is not None and Path(p).exists()
        ]
        if not dataset_images:
            raise ValueError("No hay imagenes para exportar.")

        output = Path(output_dir)
        for split in ("train", "val", "test"):
            (output / "images" / split).mkdir(parents=True, exist_ok=True)
            (output / "labels" / split).mkdir(parents=True, exist_ok=True)

        # Separar imágenes ya asignadas de las que hay que distribuir
        final_splits: dict[str, str] = {}
        unassigned: list[str] = []

        for img_path in dataset_images:
            assigned = (splits or {}).get(img_path, 'unassigned')
            if assigned in ('train', 'val', 'test'):
                final_splits[img_path] = assigned
            else:
                unassigned.append(img_path)

        # Distribuir las no asignadas con ratio val_ratio
        if unassigned:
            random.shuffle(unassigned)
            val_count = max(1, int(len(unassigned) * val_ratio)) if len(unassigned) > 1 else 0
            for i, img_path in enumerate(unassigned):
                final_splits[img_path] = 'val' if i < val_count else 'train'

        for img_path, split in final_splits.items():
            src = Path(img_path)
            shutil.copy2(src, output / "images" / split / src.name)
            lbl_path = output / "labels" / split / src.with_suffix(".txt").name
            lines = annotations[img_path].to_yolo_lines()
            lbl_path.write_text(
                ("\n".join(lines) + "\n") if lines else "",
                encoding="utf-8",
            )

        sorted_classes = sorted(classes, key=lambda c: c.class_id)
        names_block = "\n".join(f"  - {c.name}" for c in sorted_classes)
        yaml_content = (
            f"path: {output.resolve()}\n"
            f"train: images/train\n"
            f"val: images/val\n"
            f"test: images/test\n"
            f"\n"
            f"nc: {len(sorted_classes)}\n"
            f"names:\n{names_block}\n"
        )

        yaml_path = output / "data.yaml"
        yaml_path.write_text(yaml_content, encoding="utf-8")
        return str(yaml_path), final_splits
