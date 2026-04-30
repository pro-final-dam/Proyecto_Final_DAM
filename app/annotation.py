from __future__ import annotations
from dataclasses import dataclass, field
import uuid


@dataclass
class LabelClass:
    name: str
    color: str
    class_id: int


@dataclass
class BoundingBox:
    """Coordenadas en píxeles relativas a la imagen original."""
    x: float
    y: float
    width: float
    height: float
    label_class: LabelClass
    id: str = field(default_factory=lambda: str(uuid.uuid4()))

    def to_yolo(self, img_width: int, img_height: int) -> str:
        """Retorna la línea YOLO: class_id x_center y_center width height (normalizados)."""
        x_center = (self.x + self.width / 2) / img_width
        y_center = (self.y + self.height / 2) / img_height
        w_norm = self.width / img_width
        h_norm = self.height / img_height
        return (
            f"{self.label_class.class_id} "
            f"{x_center:.6f} {y_center:.6f} "
            f"{w_norm:.6f} {h_norm:.6f}"
        )

    @property
    def is_valid(self) -> bool:
        return self.width > 2 and self.height > 2


@dataclass
class ImageAnnotation:
    """Agrupa todos los bounding boxes de una imagen."""
    image_path: str
    image_width: int
    image_height: int
    boxes: list[BoundingBox] = field(default_factory=list)

    def add_box(self, box: BoundingBox) -> None:
        if box.is_valid:
            self.boxes.append(box)

    def remove_box(self, box_id: str) -> None:
        self.boxes = [b for b in self.boxes if b.id != box_id]

    def to_yolo_lines(self) -> list[str]:
        return [b.to_yolo(self.image_width, self.image_height) for b in self.boxes]
