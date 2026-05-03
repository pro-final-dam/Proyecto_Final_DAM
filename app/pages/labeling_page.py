from __future__ import annotations

import ast
import logging
import zipfile
from pathlib import Path

from PyQt6.QtWidgets import (
    QWidget, QVBoxLayout, QHBoxLayout, QFrame, QLabel, QPushButton, QToolButton,
    QListWidget, QListWidgetItem, QFileDialog, QMessageBox, QSizePolicy, QDialog,
    QProgressDialog, QScrollArea, QSlider, QStyle,
)
from PyQt6.QtCore import Qt, QSize, QRectF, QThread, pyqtSignal
from PyQt6.QtGui import QCursor, QPixmap, QColor, QPainter, QBrush, QPen

from app.canvas import AnnotationCanvas, BBoxItem
from app.annotation import BoundingBox, LabelClass, ImageAnnotation
from app.project import Project
from app.gallery import GalleryWidget
from app.yolo_exporter import YoloExporter
from app.styles import CLASS_COLORS
from app.database import DatabaseManager
from app.bbox_overlay import BBoxOverlay
from app.dialogs import AddClassDialog, ClassItemWidget, ClassListItem
from app.annotation_controller import AnnotationController

_log = logging.getLogger(__name__)

_PANEL_BG  = "#181825"
_NAV_W, _NAV_H = 200, 126   # tamaño fijo del minimap


class NavigatorWidget(QFrame):
    """Miniatura de la imagen con rectángulo que indica el área visible del canvas."""

    def __init__(self, parent=None):
        super().__init__(parent)
        self.setFixedSize(_NAV_W, _NAV_H)
        self.setStyleSheet("QFrame { background: #0d0d1a; border: 1px solid #313244; border-radius: 4px; }")
        self._thumb: QPixmap | None = None
        self._viewport_rect: QRectF | None = None   # normalizado 0-1

    def clear(self):
        self._thumb = None
        self._viewport_rect = None
        self.update()

    def set_image(self, path: str):
        raw = QPixmap(path)
        if raw.isNull():
            self.clear()
            return
        self._thumb = raw.scaled(
            _NAV_W - 2, _NAV_H - 2,
            Qt.AspectRatioMode.KeepAspectRatio,
            Qt.TransformationMode.SmoothTransformation,
        )
        self._viewport_rect = None
        self.update()

    def set_viewport(self, norm: QRectF | None):
        self._viewport_rect = norm
        self.update()

    def paintEvent(self, event):
        super().paintEvent(event)
        if not self._thumb:
            return
        painter = QPainter(self)
        painter.setRenderHint(QPainter.RenderHint.Antialiasing)

        # Centrar la miniatura en el widget
        ox = (self.width()  - self._thumb.width())  // 2
        oy = (self.height() - self._thumb.height()) // 2
        painter.drawPixmap(ox, oy, self._thumb)

        # Rectángulo de viewport
        if self._viewport_rect:
            tw, th = self._thumb.width(), self._thumb.height()
            r = self._viewport_rect
            rx = ox + max(0.0, r.x()) * tw
            ry = oy + max(0.0, r.y()) * th
            rw = min(r.width(),  1.0 - max(0.0, r.x())) * tw
            rh = min(r.height(), 1.0 - max(0.0, r.y())) * th

            fill = QColor("#89b4fa")
            fill.setAlpha(40)
            painter.setBrush(QBrush(fill))
            painter.setPen(QPen(QColor("#89b4fa"), 2))
            painter.drawRect(int(rx), int(ry), int(rw), int(rh))

        painter.end()


_PANEL_BG   = "#181825"
_PANEL_SEP  = "#313244"
_TEXT_DIM   = "#6c7086"
_TEXT_MAIN  = "#cdd6f4"
_ACCENT     = "#89b4fa"
_ITEM_BG    = "#1e1e2e"
_ITEM_HV    = "#2a2a3c"
_ITEM_SEL   = "#313244"


class _YoloImportWorker(QThread):
    progress_msg  = pyqtSignal(str)
    finished_ok   = pyqtSignal(list, list, str)   # class_names, image_records, dataset_root
    error_occurred = pyqtSignal(str)

    def __init__(self, zip_path: str, base_dir: str, parent=None):
        super().__init__(parent)
        self._zip_path = zip_path
        self._base_dir = base_dir
        self._cancelled = False

    def cancel(self) -> None:
        self._cancelled = True

    def run(self) -> None:
        try:
            import cv2 as _cv2
            from app.project import SUPPORTED_EXTENSIONS

            source = Path(self._zip_path)
            self.progress_msg.emit("Extrayendo ZIP…")
            dataset_root = self._extract(source, Path(self._base_dir))

            self.progress_msg.emit("Leyendo clases…")
            class_names = self._read_class_names(dataset_root)
            if not class_names:
                self.error_occurred.emit("No se encontraron clases en data.yaml ni classes.txt.")
                return

            all_images = sorted(
                p for p in dataset_root.rglob("*")
                if p.is_file() and p.suffix.lower() in SUPPORTED_EXTENSIONS
            )
            total = len(all_images)
            image_records: list[dict] = []
            for i, img_path in enumerate(all_images, 1):
                if self._cancelled:
                    return
                self.progress_msg.emit(f"Procesando {i}/{total}: {img_path.name}")
                img = _cv2.imread(str(img_path))
                if img is None:
                    continue
                h, w = img.shape[:2]
                split = self._detect_split(img_path)
                label_path = self._find_label(img_path)
                boxes = self._parse_labels(label_path) if (label_path and label_path.exists()) else []
                image_records.append({"path": str(img_path), "width": w, "height": h, "boxes": boxes, "split": split})

            self.finished_ok.emit(class_names, image_records, str(dataset_root))
        except Exception as exc:
            self.error_occurred.emit(str(exc))

    def _extract(self, source: Path, base_dir: Path) -> Path:
        safe_stem = "".join(ch if ch.isalnum() or ch in "-_" else "_" for ch in source.stem)
        target = base_dir / "imported_datasets" / safe_stem
        suffix = 1
        while target.exists():
            suffix += 1
            target = base_dir / "imported_datasets" / f"{safe_stem}_{suffix}"
        target.mkdir(parents=True, exist_ok=False)
        with zipfile.ZipFile(source) as zf:
            for member in zf.infolist():
                try:
                    (target / member.filename).resolve().relative_to(target.resolve())
                except ValueError:
                    raise ValueError(f"Ruta insegura dentro del ZIP: {member.filename}")
            zf.extractall(target)
        return target

    def _read_class_names(self, dataset_root: Path) -> list[str]:
        yaml_path = next(dataset_root.rglob("data.yaml"), None)
        if yaml_path:
            lines = yaml_path.read_text(encoding="utf-8").splitlines()
            for i, line in enumerate(lines):
                stripped = line.strip()
                if not stripped.startswith("names:"):
                    continue
                value = stripped.split(":", 1)[1].strip()
                if value:
                    parsed = ast.literal_eval(value)
                    if isinstance(parsed, dict):
                        return [str(parsed[k]) for k in sorted(parsed, key=lambda x: int(x))]
                    if isinstance(parsed, list):
                        return [str(v) for v in parsed]
                names: list[str] = []
                for child in lines[i + 1:]:
                    child = child.strip()
                    if not child:
                        continue
                    if not child.startswith("-"):
                        break
                    names.append(child[1:].strip().strip("'\""))
                if names:
                    return names
        classes_path = next(dataset_root.rglob("classes.txt"), None)
        if classes_path:
            return [ln.strip() for ln in classes_path.read_text(encoding="utf-8").splitlines() if ln.strip()]
        return []

    def _detect_split(self, img_path: Path) -> str | None:
        split_map = {"train": "train", "valid": "val", "val": "val", "test": "test"}
        for part in img_path.parts:
            if part.lower() in split_map:
                return split_map[part.lower()]
        return None

    def _find_label(self, img_path: Path) -> Path | None:
        parts = list(img_path.parts)
        if "images" in parts:
            parts[parts.index("images")] = "labels"
            return Path(*parts).with_suffix(".txt")
        sibling = img_path.with_suffix(".txt")
        return sibling if sibling.exists() else None

    def _parse_labels(self, label_path: Path) -> list[tuple]:
        boxes = []
        for line in label_path.read_text(encoding="utf-8").splitlines():
            parts = line.strip().split()
            if len(parts) == 5:
                try:
                    boxes.append((int(parts[0]), float(parts[1]), float(parts[2]), float(parts[3]), float(parts[4])))
                except ValueError:
                    pass
        return boxes


class LabelingPage(QWidget):

    status_message   = pyqtSignal(str)
    dataset_exported = pyqtSignal(str)

    def __init__(
        self,
        db: DatabaseManager,
        project_id: str,
        ctrl: AnnotationController,
        parent=None,
    ):
        super().__init__(parent)
        self._db = db
        self._project_id = project_id
        self._ctrl = ctrl

        self._project: Project | None = None
        self._current_image_path: str | None = None
        self._annotation: ImageAnnotation | None = None
        self._classes: list[LabelClass] = []
        self._selected_item: BBoxItem | None = None
        self._next_color_index = 0

        self._build_ui()
        self._overlay = BBoxOverlay(self.canvas.viewport())
        self._connect_signals()
        self._reload_classes()
        self._update_ui_state()

    # ------------------------------------------------------------------ #
    # Construcción de UI
    # ------------------------------------------------------------------ #

    def _build_ui(self):
        outer = QHBoxLayout(self)
        outer.setContentsMargins(0, 0, 0, 0)
        outer.setSpacing(0)

        outer.addWidget(self._build_left_panel())

        # Canvas + toolbar en columna central
        center = QWidget()
        center_layout = QVBoxLayout(center)
        center_layout.setContentsMargins(0, 0, 0, 0)
        center_layout.setSpacing(0)
        center_layout.addWidget(self._build_toolbar_row())

        self.canvas = AnnotationCanvas()
        self.canvas.setSizePolicy(QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Expanding)
        center_layout.addWidget(self.canvas)
        outer.addWidget(center)

        outer.addWidget(self._build_right_panel())

    # ── Panel izquierdo: archivo + nav + galería + clases ────────────────#

    def _build_left_panel(self) -> QFrame:
        panel = QFrame()
        panel.setObjectName("left_panel")
        panel.setFixedWidth(220)
        layout = QVBoxLayout(panel)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.setSpacing(0)

        # — Abrir imagen / carpeta —
        file_row = QFrame()
        file_row.setStyleSheet(
            f"QFrame {{ background: {_PANEL_BG}; border-bottom: 1px solid {_PANEL_SEP}; }}"
        )
        fr_layout = QHBoxLayout(file_row)
        fr_layout.setContentsMargins(6, 6, 6, 6)
        fr_layout.setSpacing(4)

        open_btn_style = "QPushButton { text-align: left; padding-left: 10px; }"

        self.btn_open_image = QPushButton("Imagen")
        self.btn_open_image.setObjectName("btn_primary")
        self.btn_open_image.setIcon(
            self.style().standardIcon(QStyle.StandardPixmap.SP_FileIcon)
        )
        self.btn_open_image.setIconSize(QSize(14, 14))
        self.btn_open_image.setSizePolicy(
            QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Fixed
        )
        self.btn_open_image.setMinimumWidth(0)
        self.btn_open_image.setMinimumHeight(30)
        self.btn_open_image.setStyleSheet(open_btn_style)
        fr_layout.addWidget(self.btn_open_image)

        self.btn_open_folder = QPushButton("Carpeta")
        self.btn_open_folder.setIcon(
            self.style().standardIcon(QStyle.StandardPixmap.SP_DirIcon)
        )
        self.btn_open_folder.setIconSize(QSize(14, 14))
        self.btn_open_folder.setSizePolicy(
            QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Fixed
        )
        self.btn_open_folder.setMinimumWidth(0)
        self.btn_open_folder.setMinimumHeight(30)
        self.btn_open_folder.setStyleSheet(open_btn_style)
        fr_layout.addWidget(self.btn_open_folder)
        layout.addWidget(file_row)

        # — Fila nombre de imagen —
        name_row = QFrame()
        name_row.setStyleSheet(
            f"QFrame {{ background: {_PANEL_BG}; border-bottom: 1px solid {_PANEL_SEP}; }}"
        )
        name_layout = QHBoxLayout(name_row)
        name_layout.setContentsMargins(8, 6, 8, 6)
        name_layout.setSpacing(6)

        img_icon = QLabel("🖼")
        img_icon.setStyleSheet("font-size: 13px;")
        img_icon.setFixedWidth(20)
        name_layout.addWidget(img_icon)

        self.lbl_current_image = QLabel("Sin imagen")
        self.lbl_current_image.setStyleSheet(
            f"color: {_TEXT_MAIN}; font-size: 11px; font-weight: bold;"
        )
        self.lbl_current_image.setMaximumWidth(160)
        name_layout.addWidget(self.lbl_current_image, 1)


        self.btn_delete_image = QToolButton()
        self.btn_delete_image.setText("🗑")
        self.btn_delete_image.setFixedSize(24, 24)
        self.btn_delete_image.setToolTip("Eliminar imagen del proyecto")
        self.btn_delete_image.setStyleSheet(
            "QToolButton { background: transparent; border: none; font-size: 14px; }"
            "QToolButton:hover { background: rgba(243,139,168,0.2); border-radius: 4px; }"
        )
        name_layout.addWidget(self.btn_delete_image)
        layout.addWidget(name_row)

        # — Fila navegación: ◀ | 1/10 | ▶ —
        nav_row = QFrame()
        nav_row.setStyleSheet(
            f"QFrame {{ background: {_PANEL_BG}; border-bottom: 1px solid {_PANEL_SEP}; }}"
        )
        nr_layout = QHBoxLayout(nav_row)
        nr_layout.setContentsMargins(0, 0, 0, 0)
        nr_layout.setSpacing(0)

        _nav_btn = (
            "QPushButton {{ background: transparent; border: none; min-width: 0;"
            " color: {color}; font-size: 36px; font-weight: bold; }}"
            "QPushButton:hover {{ background: #313244; color: {accent}; }}"
            "QPushButton:disabled {{ color: #3b3b52; }}"
        )

        self.btn_prev = QPushButton("‹")
        self.btn_prev.setFixedSize(70, 52)
        self.btn_prev.setToolTip("Anterior (←)")
        self.btn_prev.setStyleSheet(_nav_btn.format(color=_TEXT_MAIN, accent=_ACCENT))
        nr_layout.addWidget(self.btn_prev)

        self.lbl_nav = QLabel("—")
        self.lbl_nav.setStyleSheet(
            f"color: {_TEXT_MAIN}; font-size: 14px; font-weight: bold;"
        )
        self.lbl_nav.setAlignment(Qt.AlignmentFlag.AlignCenter)
        nr_layout.addWidget(self.lbl_nav, 1)

        self.btn_next = QPushButton("›")
        self.btn_next.setFixedSize(70, 52)
        self.btn_next.setToolTip("Siguiente (→)")
        self.btn_next.setStyleSheet(_nav_btn.format(color=_TEXT_MAIN, accent=_ACCENT))
        nr_layout.addWidget(self.btn_next)

        layout.addWidget(nav_row)

        # — Galería —
        layout.addWidget(self._section_header("GALERÍA"))
        layout.addWidget(self._build_gallery_filter_row())
        self.gallery = GalleryWidget()
        self.gallery.setSizePolicy(QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Expanding)
        layout.addWidget(self.gallery, 1)

        layout.addWidget(self._hsep())

        # — Label Classes —
        cls_header = QFrame()
        cls_header.setStyleSheet(
            f"QFrame {{ background: {_PANEL_BG}; border-bottom: 1px solid {_PANEL_SEP}; }}"
        )
        clh = QHBoxLayout(cls_header)
        clh.setContentsMargins(10, 6, 6, 6)
        clh.setSpacing(4)
        lbl_cls = QLabel("LABEL CLASSES")
        lbl_cls.setStyleSheet(
            f"color: {_ACCENT}; font-size: 10px; font-weight: bold; letter-spacing: 1px;"
        )
        clh.addWidget(lbl_cls)
        clh.addStretch()

        _tb_style = (
            "QToolButton {{ background: transparent; border: none;"
            " border-radius: 4px; font-size: 16px; font-weight: bold;"
            " color: {color}; padding: 2px 4px; }}"
            "QToolButton:hover {{ background: rgba(137,180,250,0.15); }}"
            "QToolButton:pressed {{ background: rgba(137,180,250,0.25); }}"
        )
        self.btn_add_class = QToolButton()
        self.btn_add_class.setText("＋")
        self.btn_add_class.setFixedSize(26, 26)
        self.btn_add_class.setToolTip("Nueva etiqueta")
        self.btn_add_class.setStyleSheet(_tb_style.format(color=_ACCENT))
        clh.addWidget(self.btn_add_class)

        self.btn_remove_class = QToolButton()
        self.btn_remove_class.setText("✕")
        self.btn_remove_class.setFixedSize(26, 26)
        self.btn_remove_class.setToolTip("Eliminar clase seleccionada")
        self.btn_remove_class.setStyleSheet(_tb_style.format(color="#f38ba8"))
        clh.addWidget(self.btn_remove_class)
        layout.addWidget(cls_header)

        self.class_list = QListWidget()
        self.class_list.setFixedHeight(130)
        self.class_list.setStyleSheet(f"""
            QListWidget {{
                background: {_ITEM_BG}; border: none; padding: 2px;
            }}
            QListWidget::item {{ border-radius: 4px; margin: 1px 2px; padding: 2px 4px; }}
            QListWidget::item:selected {{ background: {_ITEM_SEL}; color: {_ACCENT}; }}
            QListWidget::item:hover {{ background: {_ITEM_HV}; }}
        """)
        layout.addWidget(self.class_list)

        return panel

    # ── Filtro de galería ────────────────────────────────────────────── #

    def _build_gallery_filter_row(self) -> QFrame:
        row = QFrame()
        row.setStyleSheet(
            f"QFrame {{ background: {_PANEL_BG}; border-bottom: 1px solid {_PANEL_SEP}; }}"
        )
        lay = QHBoxLayout(row)
        lay.setContentsMargins(8, 5, 8, 5)

        from PyQt6.QtWidgets import QComboBox
        self._filter_combo = QComboBox()
        self._filter_combo.setStyleSheet("""
            QComboBox {
                background: #1e1e2e;
                color: #cdd6f4;
                border: 1px solid #313244;
                border-radius: 6px;
                padding: 3px 8px;
                font-size: 11px;
            }
            QComboBox::drop-down { border: none; width: 18px; }
            QComboBox::down-arrow { image: none; width: 0; }
            QComboBox QAbstractItemView {
                background: #1e1e2e;
                color: #cdd6f4;
                border: 1px solid #45475a;
                selection-background-color: #313244;
            }
        """)
        self._filter_combo.addItem('Todos',  'all')
        self._filter_combo.addItem('Train',  'train')
        self._filter_combo.addItem('Val',    'val')
        self._filter_combo.addItem('Test',   'test')
        self._filter_combo.currentIndexChanged.connect(self._on_gallery_filter_combo)
        lay.addWidget(self._filter_combo)
        return row

    def _on_gallery_filter_combo(self, index: int) -> None:
        split = self._filter_combo.itemData(index)
        self.gallery.set_split_filter(None if split == 'all' else split)

    # ── Barra de herramientas: Guardar / Exportar / Importar + Dibujar ── #

    def _build_toolbar_row(self) -> QFrame:
        wrapper = QFrame()
        wrapper.setObjectName("labeling_toolbar")
        wrapper.setStyleSheet(
            f"#labeling_toolbar {{ background: {_PANEL_BG}; border-bottom: 1px solid {_PANEL_SEP}; }}"
        )
        wl = QVBoxLayout(wrapper)
        wl.setContentsMargins(0, 0, 0, 0)
        wl.setSpacing(0)

        # — Fila 1: Guardar | Exportar YOLO | Importar YOLO —
        row1 = QFrame()
        row1.setStyleSheet("QFrame { background: transparent; }")
        r1 = QHBoxLayout(row1)
        r1.setContentsMargins(8, 5, 8, 4)
        r1.setSpacing(6)

        self.btn_save = QPushButton("💾 Guardar")
        self.btn_save.setObjectName("btn_primary")
        self.btn_save.setVisible(False)
        r1.addWidget(self.btn_save)

        self.btn_export = QPushButton("⬆ Exportar YOLO")
        r1.addWidget(self.btn_export)

        self.btn_import_yolo = QPushButton("⬇ Importar YOLO")
        r1.addWidget(self.btn_import_yolo)

        r1.addStretch()
        wl.addWidget(row1)

        # — Separador fino —
        wl.addWidget(self._hsep())

        # — Fila 2: Dibujar —
        row2 = QFrame()
        row2.setStyleSheet("QFrame { background: transparent; }")
        r2 = QHBoxLayout(row2)
        r2.setContentsMargins(8, 4, 8, 5)
        r2.setSpacing(6)

        self.btn_draw = QPushButton("✏  Dibujar bbox")
        self.btn_draw.setCheckable(True)
        self.btn_draw.setToolTip("Modo dibujo (D)")
        r2.addWidget(self.btn_draw)
        r2.addStretch()
        wl.addWidget(row2)

        return wrapper

    # ── Panel derecho: navigator + instancias ─────────────────────────── #

    def _build_right_panel(self) -> QFrame:
        panel = QFrame()
        panel.setObjectName("right_panel")
        panel.setFixedWidth(220)
        layout = QVBoxLayout(panel)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.setSpacing(0)

        # — Navigator —
        layout.addWidget(self._section_header("NAVIGATOR"))

        nav_frame = QFrame()
        nav_frame.setStyleSheet(f"QFrame {{ background: #11111b; border-bottom: 1px solid {_PANEL_SEP}; }}")
        nav_layout = QVBoxLayout(nav_frame)
        nav_layout.setContentsMargins(8, 8, 8, 8)
        nav_layout.setSpacing(6)

        self.navigator = NavigatorWidget()
        nav_layout.addWidget(self.navigator, alignment=Qt.AlignmentFlag.AlignCenter)

        # — Controles de zoom —
        zoom_row = QHBoxLayout()
        zoom_row.setSpacing(4)

        self.btn_zoom_fit = QToolButton()
        self.btn_zoom_fit.setText("Fit")
        self.btn_zoom_fit.setToolTip("Ajustar a ventana (F)")
        self.btn_zoom_fit.setStyleSheet(
            f"QToolButton {{ background: {_ITEM_SEL}; color: {_TEXT_MAIN}; border: none;"
            f" border-radius: 4px; padding: 2px 6px; font-size: 10px; }}"
            f"QToolButton:hover {{ background: #45475a; }}"
        )
        zoom_row.addWidget(self.btn_zoom_fit)

        self.btn_zoom_out = QToolButton()
        self.btn_zoom_out.setText("−")
        self.btn_zoom_out.setFixedSize(22, 22)
        self.btn_zoom_out.setStyleSheet(
            f"QToolButton {{ background: transparent; color: {_TEXT_MAIN}; border: none;"
            f" font-size: 15px; font-weight: bold; }}"
            f"QToolButton:hover {{ color: {_ACCENT}; }}"
        )
        zoom_row.addWidget(self.btn_zoom_out)

        self.zoom_slider = QSlider(Qt.Orientation.Horizontal)
        self.zoom_slider.setRange(10, 1500)
        self.zoom_slider.setValue(100)
        self.zoom_slider.setToolTip("Zoom")
        self.zoom_slider.setStyleSheet(f"""
            QSlider::groove:horizontal {{
                background: {_ITEM_SEL}; height: 4px; border-radius: 2px;
            }}
            QSlider::handle:horizontal {{
                background: {_ACCENT}; width: 12px; height: 12px;
                margin: -4px 0; border-radius: 6px;
            }}
            QSlider::sub-page:horizontal {{
                background: {_ACCENT}; border-radius: 2px;
            }}
        """)
        zoom_row.addWidget(self.zoom_slider, 1)

        self.btn_zoom_in = QToolButton()
        self.btn_zoom_in.setText("＋")
        self.btn_zoom_in.setFixedSize(22, 22)
        self.btn_zoom_in.setStyleSheet(
            f"QToolButton {{ background: transparent; color: {_TEXT_MAIN}; border: none;"
            f" font-size: 15px; font-weight: bold; }}"
            f"QToolButton:hover {{ color: {_ACCENT}; }}"
        )
        zoom_row.addWidget(self.btn_zoom_in)


        nav_layout.addLayout(zoom_row)
        layout.addWidget(nav_frame)

        layout.addWidget(self._hsep())

        # — Label Instances —
        inst_header = QFrame()
        ih_layout = QHBoxLayout(inst_header)
        ih_layout.setContentsMargins(10, 6, 10, 6)
        lbl_inst = QLabel("LABEL INSTANCES")
        lbl_inst.setStyleSheet(
            f"color: {_ACCENT}; font-size: 10px; font-weight: bold; letter-spacing: 1px;"
        )
        ih_layout.addWidget(lbl_inst)
        ih_layout.addStretch()
        self.lbl_box_count = QLabel("0")
        self.lbl_box_count.setStyleSheet(
            f"color: {_TEXT_DIM}; font-size: 10px; background: {_ITEM_SEL}; "
            f"border-radius: 8px; padding: 1px 6px;"
        )
        ih_layout.addWidget(self.lbl_box_count)
        layout.addWidget(inst_header)

        self.instances_list = QListWidget()
        self.instances_list.setStyleSheet(f"""
            QListWidget {{
                background: {_ITEM_BG}; border: none;
                border-top: 1px solid {_PANEL_SEP};
            }}
            QListWidget::item {{
                border-radius: 4px; margin: 1px 4px; padding: 4px 6px;
            }}
            QListWidget::item:selected {{ background: {_ITEM_SEL}; color: {_ACCENT}; }}
            QListWidget::item:hover {{ background: {_ITEM_HV}; }}
        """)
        layout.addWidget(self.instances_list, 1)

        # — Limpiar —
        clear_row = QFrame()
        clear_row.setStyleSheet(f"QFrame {{ background: {_PANEL_BG}; border-top: 1px solid {_PANEL_SEP}; }}")
        cr_layout = QHBoxLayout(clear_row)
        cr_layout.setContentsMargins(8, 6, 8, 6)
        self.btn_clear_all = QPushButton("✕  Limpiar todo")
        self.btn_clear_all.setStyleSheet(
            "QPushButton { background: transparent; color: #f38ba8; border: none; "
            "font-size: 11px; min-width: 0; text-align: left; }"
            "QPushButton:hover { color: #fab387; }"
        )
        cr_layout.addWidget(self.btn_clear_all)
        cr_layout.addStretch()
        layout.addWidget(clear_row)

        return panel

    # ── Helpers de construcción ───────────────────────────────────────── #

    def _section_header(self, text: str) -> QFrame:
        frame = QFrame()
        frame.setStyleSheet(
            f"background: {_PANEL_BG}; border-bottom: 1px solid {_PANEL_SEP};"
        )
        lay = QHBoxLayout(frame)
        lay.setContentsMargins(10, 7, 10, 7)
        lbl = QLabel(text)
        lbl.setStyleSheet(
            f"color: {_ACCENT}; font-size: 10px; font-weight: bold; letter-spacing: 1px;"
        )
        lay.addWidget(lbl)
        return frame

    def _hsep(self) -> QFrame:
        sep = QFrame()
        sep.setFrameShape(QFrame.Shape.HLine)
        sep.setStyleSheet(f"color: {_PANEL_SEP};")
        sep.setFixedHeight(1)
        return sep

    def _vsep(self) -> QFrame:
        sep = QFrame()
        sep.setFrameShape(QFrame.Shape.VLine)
        sep.setStyleSheet(f"color: {_PANEL_SEP};")
        sep.setFixedWidth(1)
        return sep

    def _icon_btn_style(self, color: str) -> str:
        return (
            f"QPushButton {{ background: transparent; color: {color}; border: none; "
            f"font-size: 15px; font-weight: bold; min-width: 0; border-radius: 4px; }}"
            f"QPushButton:hover {{ background: rgba(137,180,250,0.12); }}"
        )

    # ------------------------------------------------------------------ #
    # Señales
    # ------------------------------------------------------------------ #

    def _connect_signals(self):
        self.btn_open_image.clicked.connect(self._on_open_image)
        self.btn_open_folder.clicked.connect(self._on_open_folder)
        self.btn_save.clicked.connect(self._on_save_project)
        self.btn_prev.clicked.connect(self._on_prev)
        self.btn_next.clicked.connect(self._on_next)
        self.btn_draw.clicked.connect(self._on_draw_clicked)
        self.btn_export.clicked.connect(self.on_export_dataset)
        self.btn_import_yolo.clicked.connect(self._on_import_yolo)

        self.btn_delete_image.clicked.connect(self._on_delete_image)
        self.btn_add_class.clicked.connect(self._on_add_class)
        self.btn_remove_class.clicked.connect(self._on_remove_class)
        self.class_list.currentItemChanged.connect(self._on_class_selected)
        self.btn_clear_all.clicked.connect(self._on_clear_all)

        self.gallery.image_selected.connect(self._on_gallery_image_selected)
        self.gallery.split_changed.connect(self._on_split_changed)

        self.canvas.box_created.connect(self._on_box_created)
        self.canvas.box_deleted.connect(self._on_box_deleted)
        self.canvas.box_selected.connect(self._on_box_selected)
        self.canvas.view_changed.connect(self._reposition_overlay)
        self.canvas.view_changed.connect(self._on_view_changed)
        self.canvas.status_message.connect(self.status_message)

        self.zoom_slider.valueChanged.connect(self._on_slider_zoom)
        self.btn_zoom_fit.clicked.connect(self._zoom_fit)
        self.btn_zoom_in.clicked.connect(self._zoom_in)
        self.btn_zoom_out.clicked.connect(self._zoom_out)

        self._overlay.class_change_requested.connect(self._on_overlay_class_change)
        self._overlay.delete_requested.connect(self._on_overlay_delete)

    # ------------------------------------------------------------------ #
    # Clases
    # ------------------------------------------------------------------ #

    def _reload_classes(self) -> None:
        self._classes = self._ctrl.load_classes()
        self.class_list.clear()
        for lc in self._classes:
            self._add_class_to_list(lc)
        self._next_color_index = len(self._classes)
        if self._classes:
            self.status_message.emit(
                f"✅ {len(self._classes)} clase(s) cargada(s) desde la base de datos."
            )

    refresh_classes_from_db = _reload_classes

    def _on_add_class(self):
        suggested = CLASS_COLORS[self._next_color_index % len(CLASS_COLORS)]
        dialog = AddClassDialog(suggested_color=suggested, db_classes=self._classes, parent=self)
        if dialog.exec() != QDialog.DialogCode.Accepted:
            return

        name, color = dialog.get_result()
        if any(c.name == name for c in self._classes):
            self.status_message.emit(f"La clase '{name}' ya está activa en la UI.")
            return

        yolo_idx = len(self._classes)
        if dialog.new_class_confirmed:
            inserted = self._ctrl.insert_class(name, color, yolo_idx)
            if inserted:
                self.status_message.emit(f"✅ Clase '{name}' registrada en la BD y añadida.")
            else:
                self.status_message.emit(f"ℹ️ Clase '{name}' ya existía en BD (reutilizada).")
        else:
            self.status_message.emit(f"Clase '{name}' añadida a la UI (ya existía en BD).")

        self._reload_classes()

        for i in range(self.class_list.count()):
            item = self.class_list.item(i)
            if isinstance(item, ClassListItem) and item.label_class.name == name:
                self.class_list.setCurrentItem(item)
                break

    def _add_class_to_list(self, label_class: LabelClass):
        item = ClassListItem(label_class)
        widget = ClassItemWidget(label_class)
        item.setSizeHint(QSize(0, 30))
        self.class_list.addItem(item)
        self.class_list.setItemWidget(item, widget)

    def _on_remove_class(self):
        item = self.class_list.currentItem()
        if not item or not isinstance(item, ClassListItem):
            return
        name = item.label_class.name

        bbox_count = self._ctrl.count_annotations_for_class(name)
        if bbox_count > 0:
            msg = (
                f"La clase '{name}' tiene {bbox_count} anotación(es) en la base de datos.\n\n"
                "Si la eliminas, se borrarán también todas esas anotaciones.\n"
                "¿Continuar?"
            )
        else:
            msg = f"¿Eliminar la clase '{name}' de la base de datos?"

        reply = QMessageBox.question(
            self, "Eliminar clase", msg,
            QMessageBox.StandardButton.Yes | QMessageBox.StandardButton.No,
        )
        if reply != QMessageBox.StandardButton.Yes:
            return

        try:
            self._ctrl.delete_class(name)
        except Exception as e:
            self._log_db_error("Error al eliminar clase", e)
            return

        self.class_list.takeItem(self.class_list.row(item))
        self._classes = [c for c in self._classes if c.name != name]

        # Limpiar del canvas si la imagen actual tiene bboxes de esta clase
        if self._annotation:
            ids_to_remove = [b.id for b in self._annotation.boxes if b.label_class.name == name]
            for box_id in ids_to_remove:
                self._annotation.remove_box(box_id)
                self.canvas.delete_box_by_id(box_id)
            if ids_to_remove:
                self._refresh_instances_list()

        if bbox_count > 0:
            self.status_message.emit(
                f"Clase '{name}' eliminada junto con {bbox_count} anotación(es)."
            )

    def _on_class_selected(self, current, previous):
        if current and isinstance(current, ClassListItem):
            self.canvas.set_active_class(current.label_class)
            self.status_message.emit(
                f"Clase activa: {current.label_class.name}  |  "
                "Activa modo dibujo y arrastra sobre la imagen"
            )

    # ------------------------------------------------------------------ #
    # Cargar imagen / carpeta / proyecto
    # ------------------------------------------------------------------ #

    def load_project_from_db(self, project_id: str, base_path: str) -> None:
        from app.project import SUPPORTED_EXTENSIONS

        rows = self._db.conn.execute(
            """
            SELECT file_path
            FROM images
            WHERE project_id = ? AND status <> 'deleted'
            ORDER BY file_path
            """,
            (project_id,),
        ).fetchall()
        images_from_db = [r[0] for r in rows if r and r[0]]

        deleted_rows = self._db.conn.execute(
            """
            SELECT file_path
            FROM images
            WHERE project_id = ? AND status = 'deleted'
            """,
            (project_id,),
        ).fetchall()
        deleted_paths = {r[0] for r in deleted_rows if r and r[0]}

        images_from_folder: list[str] = []
        if base_path and Path(base_path).exists():
            images_from_folder = sorted([
                str(f) for f in Path(base_path).iterdir()
                if f.suffix.lower() in SUPPORTED_EXTENSIONS
                and str(f) not in deleted_paths
            ])

        images = sorted(set(images_from_db + images_from_folder))
        if not images:
            self.status_message.emit(
                "El proyecto no tiene imágenes accesibles. "
                "Carga imágenes desde la galería para comenzar."
            )
            return

        project = Project(folder=base_path, image_paths=images)
        project.annotations = {}

        # Carga solo contadores para los badges — mucho más rápido que cargar todas las anotaciones
        annotation_counts: dict[str, int] = {
            r[0]: r[1]
            for r in self._db.conn.execute(
                "SELECT file_path, annotation_count FROM images WHERE project_id = ? AND status <> 'deleted'",
                (project_id,),
            ).fetchall()
            if r[1]
        }

        try:
            from datetime import datetime, timezone
            self._db.conn.execute(
                "UPDATE projects SET base_path = ?, updated_at = ? WHERE id = ?",
                (base_path, datetime.now(timezone.utc), project_id),
            )
        except Exception:
            pass

        self._load_project(project, annotation_counts=annotation_counts)
        if images:
            self._navigate_to(images[0], save_current=False)

    def _on_open_image(self):
        path, _ = QFileDialog.getOpenFileName(
            self, "Cargar imagen", "",
            "Imágenes (*.jpg *.jpeg *.png *.bmp *.tiff *.webp)"
        )
        if not path:
            return
        if self._project is None:
            self._load_project(Project.from_single_image(path))
            self._navigate_to(path, save_current=False)
        else:
            added = self._add_images_to_project([path])
            if added:
                self._navigate_to(path)
            else:
                self.status_message.emit("La imagen ya estaba en el proyecto.")

    def _on_open_folder(self):
        folder = QFileDialog.getExistingDirectory(self, "Cargar imágenes de directorio")
        if not folder:
            return
        from app.project import SUPPORTED_EXTENSIONS
        new_paths = sorted([
            str(f) for f in Path(folder).iterdir()
            if f.suffix.lower() in SUPPORTED_EXTENSIONS
        ])
        if not new_paths:
            QMessageBox.information(self, "Sin imágenes", "No se encontraron imágenes en esa carpeta.")
            return
        if self._project is None:
            project = Project.open_folder(folder)
            self._load_project(project)
            self._navigate_to(project.image_paths[0], save_current=False)
        else:
            added = self._add_images_to_project(new_paths)
            self.status_message.emit(
                f"{added} imagen(es) añadida(s)." if added
                else "No hay imágenes nuevas en ese directorio."
            )

    def _add_images_to_project(self, paths: list[str]) -> int:
        existing = set(self._project.image_paths)
        new = [p for p in paths if p not in existing]
        if not new:
            return 0
        self._project.image_paths = sorted(self._project.image_paths + new)
        self.gallery.load_images(self._project.image_paths)
        for img_path, ann in self._project.annotations.items():
            if ann.boxes:
                self.gallery.update_badge(img_path, len(ann.boxes))
        for file_path, split in self._ctrl.load_gallery_splits().items():
            self.gallery.update_split(file_path, split)
        self._update_ui_state()
        return len(new)

    def _on_split_changed(self, image_path: str, split: str) -> None:
        try:
            self._ctrl.update_split(image_path, split)
            self.gallery.update_split(image_path, split)
        except Exception as e:
            self._log_db_error("Error al asignar split", e)

    def _load_project(self, project: Project, annotation_counts: dict[str, int] | None = None):
        self._project = project
        self.gallery.load_images(project.image_paths)

        counts = annotation_counts if annotation_counts is not None else {
            p: len(a.boxes) for p, a in project.annotations.items() if a.boxes
        }
        for img_path, count in counts.items():
            if count > 0:
                self.gallery.update_badge(img_path, count)

        for file_path, split in self._ctrl.load_gallery_splits().items():
            self.gallery.update_split(file_path, split)
        self._update_ui_state()
        self._reload_classes()

    # ------------------------------------------------------------------ #
    # Navegación
    # ------------------------------------------------------------------ #

    def _navigate_to(self, image_path: str, save_current: bool = True):
        if save_current:
            self._commit_current_annotation()
        if not self.canvas.load_image(image_path):
            QMessageBox.warning(self, "Error", f"No se pudo cargar:\n{image_path}")
            return

        self._current_image_path = image_path
        w, h = self.canvas.get_image_size()

        # Carga lazy: solo consulta la BD la primera vez que se visita la imagen
        if image_path not in self._project.annotations:
            db_ann = self._ctrl.load_annotation_for_image(image_path)
            if db_ann:
                self._project.annotations[image_path] = db_ann

        self._annotation = self._project.get_or_create_annotation(image_path, w, h)
        self._ctrl.upsert_image(image_path, w, h)

        self.canvas.clear_all_boxes()
        for box in self._annotation.boxes:
            self.canvas.add_bbox_item(box)

        self.gallery.set_current_by_path(image_path)
        self._update_navigator(image_path)
        self._refresh_instances_list()
        self._update_nav_label()
        self._update_ui_state()

        name = Path(image_path).name
        elided = self.lbl_current_image.fontMetrics().elidedText(
            name, Qt.TextElideMode.ElideMiddle, self.lbl_current_image.maximumWidth()
        )
        self.lbl_current_image.setText(elided)
        self.lbl_current_image.setToolTip(name)
        self.status_message.emit(f"{name}  —  {w} × {h} px")

    def _commit_current_annotation(self):
        if self._annotation and self._project:
            self._project.annotations[self._annotation.image_path] = self._annotation
            self.gallery.update_badge(self._annotation.image_path, len(self._annotation.boxes))

    def _on_gallery_image_selected(self, image_path: str):
        if image_path != self._current_image_path:
            self._navigate_to(image_path)

    def _on_prev(self):
        if not self._project or not self._current_image_path:
            return
        idx = self._project.index_of(self._current_image_path)
        if idx > 0:
            self._navigate_to(self._project.image_paths[idx - 1])

    def _on_next(self):
        if not self._project or not self._current_image_path:
            return
        idx = self._project.index_of(self._current_image_path)
        if idx < self._project.image_count - 1:
            self._navigate_to(self._project.image_paths[idx + 1])

    def _update_nav_label(self):
        if not self._project or not self._current_image_path:
            self.lbl_nav.setText("—")
            return
        idx = self._project.index_of(self._current_image_path)
        self.lbl_nav.setText(f"{idx + 1} / {self._project.image_count}")

    # ------------------------------------------------------------------ #
    # Navigator minimap y zoom
    # ------------------------------------------------------------------ #

    def _update_navigator(self, image_path: str):
        self.navigator.set_image(image_path)

    def _on_view_changed(self):
        """Actualiza el rectángulo de viewport en el navigator y el slider de zoom."""
        norm = self.canvas.get_viewport_normalized_rect()
        self.navigator.set_viewport(norm)

        zoom = self.canvas.get_zoom_level()
        pct = int(zoom * 100)
        self.zoom_slider.blockSignals(True)
        self.zoom_slider.setValue(max(10, min(1500, pct)))
        self.zoom_slider.blockSignals(False)

    def _on_slider_zoom(self, value: int):
        fit = self.canvas.get_fit_zoom()
        target = max(value / 100.0, fit)
        self.canvas.set_zoom_level(target)

    # ------------------------------------------------------------------ #
    # Label Instances (lista de bboxes actuales)
    # ------------------------------------------------------------------ #

    def _refresh_instances_list(self):
        self.instances_list.clear()
        boxes = self._annotation.boxes if self._annotation else []
        for i, box in enumerate(boxes, 1):
            item = QListWidgetItem()
            item.setData(Qt.ItemDataRole.UserRole, box.id)

            # Icono de color
            pix = QPixmap(12, 12)
            pix.fill(QColor(box.label_class.color))
            painter = QPainter(pix)
            painter.setRenderHint(QPainter.RenderHint.Antialiasing)
            painter.setBrush(QBrush(QColor(box.label_class.color)))
            painter.setPen(Qt.PenStyle.NoPen)
            painter.drawRoundedRect(0, 0, 12, 12, 3, 3)
            painter.end()

            from PyQt6.QtGui import QIcon
            item.setIcon(QIcon(pix))
            item.setText(f"  {box.label_class.name}  #{i}")
            self.instances_list.addItem(item)

        count = len(boxes)
        self.lbl_box_count.setText(str(count))

    # ------------------------------------------------------------------ #
    # Guardar
    # ------------------------------------------------------------------ #

    def _on_save_project(self):
        if not self._project:
            return
        self._commit_current_annotation()
        self._project.classes = self._classes
        self._project.save()
        self.status_message.emit(
            f"Proyecto guardado en {self._project.folder}/visionhub_project.json"
        )

    def _on_delete_image(self):
        if not self._current_image_path or not self._project:
            return
        name = Path(self._current_image_path).name
        reply = QMessageBox.question(
            self, "Eliminar imagen",
            f"¿Eliminar '{name}' del proyecto?\n\nSus anotaciones también se borrarán.",
            QMessageBox.StandardButton.Yes | QMessageBox.StandardButton.No,
        )
        if reply != QMessageBox.StandardButton.Yes:
            return

        path = self._current_image_path
        idx = self._project.index_of(path)

        try:
            self._ctrl.delete_image_from_project(path)
        except Exception as e:
            self._log_db_error("Error al eliminar imagen de BD", e)
            return

        # Eliminar del proyecto en memoria
        self._project.image_paths.remove(path)
        self._project.annotations.pop(path, None)

        # Limpiar canvas y overlay
        self.canvas.clear_all_boxes()
        self._overlay.hide()
        self._selected_item = None
        self._annotation = None
        self._current_image_path = None

        # Recargar galería
        self.gallery.load_images(self._project.image_paths)
        for img_path, ann in self._project.annotations.items():
            if ann.boxes:
                self.gallery.update_badge(img_path, len(ann.boxes))
        for file_path, split in self._ctrl.load_gallery_splits().items():
            self.gallery.update_split(file_path, split)
        self._refresh_instances_list()

        # Navegar a la imagen más cercana
        if self._project.image_paths:
            next_idx = min(idx, self._project.image_count - 1)
            self._navigate_to(self._project.image_paths[next_idx], save_current=False)
        else:
            self.canvas.clear_image()
            self.navigator.clear()
            self.btn_draw.setChecked(False)
            self.btn_draw.setText("✏  Dibujar bbox")
            self.canvas.set_draw_mode(False)
            self.lbl_current_image.setText("Sin imagen")
            self.lbl_nav.setText("—")
            self._update_ui_state()

        self.status_message.emit(f"Imagen '{name}' eliminada del proyecto.")

    # ------------------------------------------------------------------ #
    # Bboxes
    # ------------------------------------------------------------------ #

    def _on_box_created(self, bbox: BoundingBox):
        if self._annotation:
            self._annotation.add_box(bbox)
            self._refresh_instances_list()
            try:
                self._ctrl.persist_annotation(self._annotation)
                self._ctrl.update_image_status(self._current_image_path, +1)
            except Exception as e:
                self._log_db_error("Error al guardar en BD", e)

    def _on_box_deleted(self, box_id: str):
        if self._annotation:
            self._annotation.remove_box(box_id)
            self._refresh_instances_list()
            try:
                self._ctrl.delete_annotation(box_id)
                self._ctrl.update_image_status(self._current_image_path, -1)
            except Exception as e:
                self._log_db_error("Error al borrar en BD", e)

    def _on_clear_all(self):
        if not self._annotation:
            return
        reply = QMessageBox.question(
            self, "Confirmar", "¿Eliminar todos los bounding boxes?",
            QMessageBox.StandardButton.Yes | QMessageBox.StandardButton.No,
        )
        if reply == QMessageBox.StandardButton.Yes:
            try:
                self._ctrl.clear_annotations(
                    self._current_image_path,
                    [box.id for box in self._annotation.boxes],
                )
            except Exception as e:
                self._log_db_error("Error al borrar en BD", e)
            self._overlay.hide()
            self._selected_item = None
            self.canvas.clear_all_boxes()
            self._annotation.boxes.clear()
            self._refresh_instances_list()

    # ------------------------------------------------------------------ #
    # Overlay
    # ------------------------------------------------------------------ #

    def _on_box_selected(self, item):
        self._selected_item = item
        if item is None:
            self._overlay.hide()
            return
        self._overlay.update_content(
            item.bbox.id, item.bbox.label_class.name,
            item.bbox.label_class.color, self._classes,
        )
        self._reposition_overlay()

    def _reposition_overlay(self):
        if self._selected_item is None:
            return
        top_center = self.canvas.get_item_viewport_top_center(self._selected_item)
        size = self._overlay.sizeHint()
        x = top_center.x() - size.width() // 2
        y = top_center.y() - size.height() - 6
        vp = self.canvas.viewport().rect()
        x = max(4, min(x, vp.width() - size.width() - 4))
        y = max(4, y)
        self._overlay.move(x, y)
        self._overlay.raise_()
        if not self._overlay.isVisible():
            self._overlay.show()

    def _on_overlay_class_change(self, box_id: str, new_class: LabelClass):
        self.canvas.change_box_class(box_id, new_class)
        if self._selected_item and self._selected_item.bbox.id == box_id:
            self._overlay.update_content(box_id, new_class.name, new_class.color, self._classes)
            self._reposition_overlay()
        self._refresh_instances_list()
        try:
            self._ctrl.persist_annotation(self._annotation)
        except Exception as e:
            self._log_db_error("Error al persistir cambio de clase", e)

    def _on_overlay_delete(self, box_id: str):
        self._selected_item = None
        self.canvas.delete_box_by_id(box_id)

    # ------------------------------------------------------------------ #
    # Dibujo
    # ------------------------------------------------------------------ #

    def _on_draw_clicked(self):
        want_draw = self.btn_draw.isChecked()
        if want_draw:
            current_item = self.class_list.currentItem()
            if not current_item or not isinstance(current_item, ClassListItem):
                QMessageBox.warning(
                    self, "Sin clase seleccionada",
                    "Selecciona una clase en el panel izquierdo antes de dibujar.",
                )
                self.btn_draw.setChecked(False)
                return
            self.canvas.set_draw_mode(True)
            self.btn_draw.setText("✏ Dibujando…")
            self._overlay.hide()
            self.status_message.emit(
                f"✏️ Modo dibujo — clase: {current_item.label_class.name}  "
                "| Arrastra para crear un bbox"
            )
        else:
            self.canvas.set_draw_mode(False)
            self.btn_draw.setText("✏ Dibujar")

    # ------------------------------------------------------------------ #
    # Exportar
    # ------------------------------------------------------------------ #

    def _on_export_yolo(self):
        if not self._current_image_path:
            QMessageBox.information(self, "Sin imagen", "No hay imagen cargada.")
            return
        try:
            row = self._db.conn.execute(
                """
                SELECT id, width, height
                FROM images
                WHERE file_path = ? AND project_id = ? AND status <> 'deleted'
                """,
                (self._current_image_path, self._project_id)
            ).fetchone()
            if not row:
                raise ValueError("La imagen no está registrada en la base de datos.")

            img_id, w, h = row
            db_boxes = self._db.get_annotations_for_image(img_id)
            if not db_boxes:
                QMessageBox.information(self, "Sin datos", "No hay anotaciones en BD que exportar.")
                return

            full_annotation = ImageAnnotation(
                image_path=self._current_image_path, image_width=w, image_height=h,
            )
            for b in db_boxes:
                cls_row = self._db.conn.execute(
                    "SELECT name, color, yolo_index FROM classes WHERE id = ?", (b["class_id"],)
                ).fetchone()
                if not cls_row:
                    continue
                c_name, c_color, c_yolo = cls_row
                bw = b["width"] * w
                bh = b["height"] * h
                full_annotation.add_box(BoundingBox(
                    x=(b["x_center"] * w) - (bw / 2),
                    y=(b["y_center"] * h) - (bh / 2),
                    width=bw, height=bh,
                    label_class=LabelClass(name=c_name, color=c_color, class_id=c_yolo),
                    id=b["id"],
                ))

            out_file = YoloExporter.export(full_annotation)
            classes_rows = self._db.conn.execute(
                "SELECT name FROM classes WHERE project_id = ? ORDER BY yolo_index",
                (self._project_id,)
            ).fetchall()
            out_classes = Path(self._current_image_path).parent / "classes.txt"
            out_classes.write_text("\n".join(r[0] for r in classes_rows) + "\n", encoding="utf-8")

            QMessageBox.information(self, "Exportación completada", f"Generado:\n{out_file}")
            self.status_message.emit(f"Exportado: {out_file}")
        except Exception as e:
            QMessageBox.critical(self, "Error", str(e))

    def _on_import_yolo(self):
        """Importa un .txt YOLO individual o un dataset YOLO completo en .zip."""
        import_path, _ = QFileDialog.getOpenFileName(
            self,
            "Importar YOLO",
            "",
            "Dataset YOLO (*.zip);;YOLO labels (*.txt);;Todos (*.*)",
        )
        if not import_path:
            return

        if Path(import_path).suffix.lower() == ".zip":
            self._import_yolo_dataset_zip(import_path)
            return

        if not self._current_image_path or not self._annotation:
            QMessageBox.information(
                self,
                "Sin imagen",
                "Para importar un .txt individual primero carga la imagen correspondiente.",
            )
            return
        txt_path = import_path
        try:
            lines = Path(txt_path).read_text(encoding="utf-8").strip().splitlines()
            imported = 0
            for line in lines:
                parts = line.strip().split()
                if len(parts) != 5:
                    continue
                class_id, xc, yc, w_n, h_n = int(parts[0]), float(parts[1]), float(parts[2]), float(parts[3]), float(parts[4])
                iw, ih = self._annotation.image_width, self._annotation.image_height
                w_px = w_n * iw
                h_px = h_n * ih
                x_px = xc * iw - w_px / 2
                y_px = yc * ih - h_px / 2
                cls = next((c for c in self._classes if c.class_id == class_id), None)
                if cls is None:
                    continue
                bbox = BoundingBox(x=x_px, y=y_px, width=w_px, height=h_px, label_class=cls)
                if bbox.is_valid:
                    self._annotation.add_box(bbox)
                    self.canvas.add_bbox_item(bbox)
                    imported += 1
            self._refresh_instances_list()
            self._ctrl.persist_annotation(self._annotation)
            self.status_message.emit(f"✅ {imported} anotación(es) importada(s) desde YOLO.")
        except Exception as e:
            QMessageBox.critical(self, "Error al importar", str(e))

    def _import_yolo_dataset_zip(self, zip_path: str) -> None:
        source = Path(zip_path)
        try:
            base_dir = self._dataset_import_base_dir(source)
            if base_dir is None:
                return

            self._import_progress = QProgressDialog("Preparando importación…", "Cancelar", 0, 0, self)
            self._import_progress.setWindowTitle("Importando dataset YOLO")
            self._import_progress.setMinimumWidth(420)
            self._import_progress.setWindowModality(Qt.WindowModality.WindowModal)
            self._import_progress.show()

            self._import_worker = _YoloImportWorker(zip_path=zip_path, base_dir=str(base_dir))
            self._import_progress.canceled.connect(self._import_worker.cancel)
            self._import_worker.progress_msg.connect(self._import_progress.setLabelText)
            self._import_worker.finished_ok.connect(self._on_import_finished)
            self._import_worker.error_occurred.connect(self._on_import_error)
            self._import_worker.finished.connect(self._import_progress.close)
            self._import_worker.start()
        except Exception as e:
            QMessageBox.critical(self, "Error al importar dataset", str(e))

    def _dataset_import_base_dir(self, source: Path) -> Path | None:
        if self._project and self._project.folder:
            base = Path(self._project.folder)
            if str(base):
                base.mkdir(parents=True, exist_ok=True)
                return base

        folder = QFileDialog.getExistingDirectory(
            self,
            "Carpeta donde guardar las imágenes importadas",
            str(source.parent),
        )
        if not folder:
            return None

        project = Project(folder=folder, image_paths=[])
        self._load_project(project)
        try:
            self._db.conn.execute(
                "UPDATE projects SET base_path = ? WHERE id = ?",
                (folder, self._project_id),
            )
        except Exception:
            pass
        return Path(folder)

    def _ensure_imported_classes(self, class_names: list[str]) -> dict[int, LabelClass]:
        existing_by_name = {c.name: c for c in self._classes}
        class_map: dict[int, LabelClass] = {}

        for imported_idx, name in enumerate(class_names):
            if name in existing_by_name:
                class_map[imported_idx] = existing_by_name[name]
                continue

            yolo_idx = len(self._classes)
            color = CLASS_COLORS[yolo_idx % len(CLASS_COLORS)]
            self._ctrl.insert_class(name, color, yolo_idx)
            new_class = LabelClass(name=name, color=color, class_id=yolo_idx)
            self._classes.append(new_class)
            existing_by_name[name] = new_class
            class_map[imported_idx] = new_class

        self._reload_classes()
        refreshed_by_name = {c.name: c for c in self._classes}
        return {
            imported_idx: refreshed_by_name[label_class.name]
            for imported_idx, label_class in class_map.items()
            if label_class.name in refreshed_by_name
        }

    def _on_import_finished(self, class_names: list, image_records: list, dataset_root: str) -> None:
        if not image_records:
            QMessageBox.warning(self, "Sin imágenes", "No se encontraron imágenes compatibles dentro del ZIP.")
            return

        class_map = self._ensure_imported_classes(class_names)
        existing = set(self._project.image_paths) if self._project else set()
        imported_images = 0
        imported_boxes = 0

        for record in image_records:
            image_str = record["path"]
            w, h = record["width"], record["height"]

            if image_str not in existing:
                self._project.image_paths.append(image_str)
                existing.add(image_str)
                imported_images += 1

            ann = ImageAnnotation(image_path=image_str, image_width=w, image_height=h)
            for class_idx, xc, yc, w_n, h_n in record["boxes"]:
                cls = class_map.get(class_idx)
                if cls is None:
                    continue
                w_px, h_px = w_n * w, h_n * h
                bbox = BoundingBox(
                    x=xc * w - w_px / 2, y=yc * h - h_px / 2,
                    width=w_px, height=h_px, label_class=cls,
                )
                if bbox.is_valid:
                    ann.add_box(bbox)
                    imported_boxes += 1

            self._project.annotations[image_str] = ann
            self._ctrl.upsert_image(image_str, w, h)
            self._ctrl.persist_annotation(ann)
            self._ctrl.sync_image_annotation_summary(image_str)
            if record["split"]:
                self._ctrl.update_split(image_str, record["split"])

        self._project.image_paths = sorted(self._project.image_paths)
        self.gallery.load_images(self._project.image_paths)
        for img_path, ann in self._project.annotations.items():
            if ann.boxes:
                self.gallery.update_badge(img_path, len(ann.boxes))
        for file_path, split in self._ctrl.load_gallery_splits().items():
            self.gallery.update_split(file_path, split)
        self._update_ui_state()

        if not self._current_image_path and self._project.image_paths:
            self._navigate_to(self._project.image_paths[0], save_current=False)

        QMessageBox.information(
            self, "Dataset importado",
            f"Importadas {imported_images} imagen(es) y {imported_boxes} etiqueta(s).\n\n"
            f"Origen extraído en:\n{dataset_root}",
        )
        self.status_message.emit(f"Dataset YOLO importado: {imported_images} imagen(es), {imported_boxes} bbox.")

    def _on_import_error(self, msg: str) -> None:
        QMessageBox.critical(self, "Error al importar dataset", msg)

    def on_export_dataset(self):
        output_dir = QFileDialog.getExistingDirectory(self, "Carpeta de destino del dataset")
        if not output_dir:
            return
        self.export_dataset_to_dir(output_dir, show_messages=True)

    def export_dataset_to_dir(self, output_dir: str, *, show_messages: bool = False) -> str:
        if not self._project:
            if show_messages:
                QMessageBox.warning(self, "Sin proyecto", "Abre un proyecto con imagenes primero.")
            return ""
        if not self._project.image_paths:
            if show_messages:
                QMessageBox.warning(self, "Sin imagenes", "No hay imagenes para exportar.")
            return ""
        try:
            self._commit_current_annotation()
            # Con carga lazy, project.annotations solo tiene las imágenes visitadas.
            # Para exportar cargamos todo desde la BD y sobreescribimos con lo que haya en memoria.
            all_annotations = self._db.load_annotations_for_project(self._project_id)
            for path, ann in self._project.annotations.items():
                all_annotations[path] = ann
            self._ensure_export_annotations_into(all_annotations)

            splits_from_db = {r[0]: r[1] for r in self._db.conn.execute(
                """
                SELECT file_path, split
                FROM images
                WHERE project_id = ? AND status <> 'deleted'
                """,
                (self._project_id,),
            ).fetchall()}

            yaml_path, final_splits = YoloExporter.export_dataset(
                annotations=all_annotations,
                classes=self._classes,
                output_dir=output_dir,
                splits=splits_from_db,
            )

            for img_path, split in final_splits.items():
                self._ctrl.update_split(img_path, split)
                self.gallery.update_split(img_path, split)

            train_count = sum(1 for s in final_splits.values() if s == "train")
            val_count = sum(1 for s in final_splits.values() if s == "val")
            test_count = sum(1 for s in final_splits.values() if s == "test")
            self.dataset_exported.emit(yaml_path)
            if show_messages:
                QMessageBox.information(
                    self, "Dataset generado",
                    f"Dataset listo en:\n{output_dir}\n\n"
                    f"Train: {train_count} imagenes\n"
                    f"Val: {val_count} imagenes\n"
                    f"Test: {test_count} imagenes",
                )
            return yaml_path
        except Exception as e:
            if show_messages:
                QMessageBox.critical(self, "Error al exportar dataset", str(e))
            return ""

    def _ensure_export_annotations_into(self, annotations: dict) -> None:
        """Garantiza que cada imagen del proyecto tenga entrada en el dict, aunque esté vacía."""
        if not self._project:
            return
        for image_path in self._project.image_paths:
            if image_path in annotations:
                continue
            pix = QPixmap(image_path)
            if pix.isNull():
                continue
            annotations[image_path] = ImageAnnotation(
                image_path=image_path,
                image_width=pix.width(),
                image_height=pix.height(),
            )
            self._ctrl.upsert_image(image_path, pix.width(), pix.height())

    # ------------------------------------------------------------------ #
    # Zoom
    # ------------------------------------------------------------------ #

    def _zoom_fit(self):
        if self.canvas.has_image():
            self.canvas.fit_view()

    def _zoom_in(self):
        if self.canvas.get_zoom_level() < 15.0:
            self._scale_centered(1.25)

    def _zoom_out(self):
        fit = self.canvas.get_fit_zoom()
        current = self.canvas.get_zoom_level()
        new_zoom = current / 1.25
        if new_zoom < fit:
            new_zoom = fit
        self.canvas.set_zoom_level(new_zoom)

    def _scale_centered(self, factor: float):
        from PyQt6.QtWidgets import QGraphicsView
        self.canvas.setTransformationAnchor(QGraphicsView.ViewportAnchor.AnchorViewCenter)
        self.canvas.scale(factor, factor)
        self.canvas.setTransformationAnchor(QGraphicsView.ViewportAnchor.AnchorUnderMouse)

    # ------------------------------------------------------------------ #
    # Helpers
    # ------------------------------------------------------------------ #

    def _update_ui_state(self):
        has_project = self._project is not None
        has_image   = self._current_image_path is not None
        multi       = has_project and self._project.image_count > 1
        idx = self._project.index_of(self._current_image_path) if (has_project and has_image) else -1

        self.btn_save.setEnabled(has_project)
        self.btn_delete_image.setEnabled(has_image)
        self.btn_draw.setEnabled(has_image)
        self.btn_export.setEnabled(has_project)
        self.btn_import_yolo.setEnabled(True)
        self.btn_clear_all.setEnabled(has_image)
        self.btn_zoom_fit.setEnabled(has_image)
        self.btn_zoom_in.setEnabled(has_image)
        self.btn_zoom_out.setEnabled(has_image)
        self.zoom_slider.setEnabled(has_image)
        self.btn_prev.setEnabled(multi and idx > 0)
        self.btn_next.setEnabled(multi and idx < self._project.image_count - 1)

    def _log_db_error(self, msg: str, exc: Exception) -> None:
        full = f"⚠ {msg}: {exc}"
        self.status_message.emit(full)
        _log.error(full, exc_info=exc)

    # ------------------------------------------------------------------ #
    # API pública para MainWindow
    # ------------------------------------------------------------------ #

    def handle_key(self, event) -> bool:
        key = event.key()
        if key == Qt.Key.Key_D:
            self.btn_draw.setChecked(not self.btn_draw.isChecked())
            self._on_draw_clicked()
            return True
        if key == Qt.Key.Key_F:
            self._zoom_fit()
            return True
        if key == Qt.Key.Key_Left:
            self._on_prev()
            return True
        if key == Qt.Key.Key_Right:
            self._on_next()
            return True
        return False

    def on_close(self):
        if self._project:
            self._commit_current_annotation()
            self._project.classes = self._classes
            self._project.save()
