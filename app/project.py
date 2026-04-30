from __future__ import annotations
import json
import uuid
from pathlib import Path
from dataclasses import dataclass, field

from app.annotation import LabelClass, BoundingBox, ImageAnnotation

SUPPORTED_EXTENSIONS = {'.jpg', '.jpeg', '.png', '.bmp', '.tiff', '.webp'}
PROJECT_FILE = 'visionhub_project.json'


@dataclass
class Project:
    folder: str
    image_paths: list[str] = field(default_factory=list)
    classes: list[LabelClass] = field(default_factory=list)
    annotations: dict[str, ImageAnnotation] = field(default_factory=dict)

    @classmethod
    def open_folder(cls, folder: str) -> Project:
        p = cls(folder=folder)
        p.image_paths = sorted([
            str(f) for f in Path(folder).iterdir()
            if f.suffix.lower() in SUPPORTED_EXTENSIONS
        ])
        project_file = Path(folder) / PROJECT_FILE
        if project_file.exists():
            p._load_json(project_file)
        return p

    @classmethod
    def from_single_image(cls, image_path: str) -> Project:
        folder = str(Path(image_path).parent)
        p = cls(folder=folder, image_paths=[image_path])
        return p

    def _load_json(self, path: Path):
        data = json.loads(path.read_text(encoding='utf-8'))
        self.classes = [
            LabelClass(name=c['name'], color=c['color'], class_id=c['class_id'])
            for c in data.get('classes', [])
        ]
        class_map = {c.name: c for c in self.classes}
        for img_path, ann_data in data.get('annotations', {}).items():
            boxes = [
                BoundingBox(
                    x=b['x'], y=b['y'],
                    width=b['width'], height=b['height'],
                    label_class=class_map[b['class_name']],
                    id=b.get('id', str(uuid.uuid4())),
                )
                for b in ann_data.get('boxes', [])
                if b.get('class_name') in class_map
            ]
            self.annotations[img_path] = ImageAnnotation(
                image_path=img_path,
                image_width=ann_data['image_width'],
                image_height=ann_data['image_height'],
                boxes=boxes,
            )

    def save(self):
        data = {
            'classes': [
                {'name': c.name, 'color': c.color, 'class_id': c.class_id}
                for c in self.classes
            ],
            'annotations': {
                img_path: {
                    'image_width': ann.image_width,
                    'image_height': ann.image_height,
                    'boxes': [
                        {
                            'x': b.x, 'y': b.y,
                            'width': b.width, 'height': b.height,
                            'class_name': b.label_class.name,
                            'id': b.id,
                        }
                        for b in ann.boxes
                    ],
                }
                for img_path, ann in self.annotations.items()
                if ann.boxes
            },
        }
        out = Path(self.folder) / PROJECT_FILE
        out.write_text(json.dumps(data, indent=2, ensure_ascii=False), encoding='utf-8')

    def get_or_create_annotation(self, image_path: str, w: int, h: int) -> ImageAnnotation:
        if image_path not in self.annotations:
            ann = self._try_import_yolo(image_path, w, h)
            self.annotations[image_path] = ann or ImageAnnotation(
                image_path=image_path, image_width=w, image_height=h
            )
        return self.annotations[image_path]

    def _try_import_yolo(self, image_path: str, w: int, h: int) -> ImageAnnotation | None:
        txt = Path(image_path).with_suffix('.txt')
        if not txt.exists() or not self.classes:
            return None
        class_map = {c.class_id: c for c in self.classes}
        boxes = []
        for line in txt.read_text(encoding='utf-8').strip().splitlines():
            parts = line.split()
            if len(parts) != 5:
                continue
            try:
                cid, xc, yc, bw, bh = int(parts[0]), float(parts[1]), float(parts[2]), float(parts[3]), float(parts[4])
            except ValueError:
                continue
            cls = class_map.get(cid)
            if not cls:
                continue
            boxes.append(BoundingBox(
                x=(xc - bw / 2) * w,
                y=(yc - bh / 2) * h,
                width=bw * w,
                height=bh * h,
                label_class=cls,
            ))
        return ImageAnnotation(image_path=image_path, image_width=w, image_height=h, boxes=boxes) if boxes else None

    def index_of(self, image_path: str) -> int:
        try:
            return self.image_paths.index(image_path)
        except ValueError:
            return -1

    @property
    def image_count(self) -> int:
        return len(self.image_paths)
