from __future__ import annotations

import logging
from pathlib import Path

from PyQt6.QtWidgets import (
    QWidget, QVBoxLayout, QHBoxLayout, QFrame, QLabel, QPushButton, QToolButton,
    QListWidget, QListWidgetItem, QFileDialog, QMessageBox, QSizePolicy, QDialog,
    QScrollArea, QSlider, QStyle,
)
from PyQt6.QtCore import Qt, QSize, QRectF, pyqtSignal
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

        self.lbl_gallery_count = QLabel("")
        self.lbl_gallery_count.setStyleSheet(
            f"color: {_TEXT_DIM}; font-size: 10px; background: {_ITEM_SEL};"
            f" border-radius: 8px; padding: 1px 5px;"
        )
        name_layout.addWidget(self.lbl_gallery_count)

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
        self.btn_export.clicked.connect(self._on_export_yolo)
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
            "SELECT file_path FROM images WHERE project_id = ? ORDER BY file_path",
            (project_id,),
        ).fetchall()
        images_from_db = [r[0] for r in rows if r and r[0] and Path(r[0]).exists()]

        images_from_folder: list[str] = []
        if base_path and Path(base_path).exists():
            images_from_folder = sorted([
                str(f) for f in Path(base_path).iterdir()
                if f.suffix.lower() in SUPPORTED_EXTENSIONS
            ])

        images = sorted(set(images_from_db + images_from_folder))
        if not images:
            self.status_message.emit(
                "El proyecto no tiene imágenes accesibles. "
                "Carga imágenes desde la galería para comenzar."
            )
            return

        project = Project(folder=base_path, image_paths=images)
        project.annotations = self._db.load_annotations_for_project(project_id)

        try:
            from datetime import datetime, timezone
            self._db.conn.execute(
                "UPDATE projects SET base_path = ?, updated_at = ? WHERE id = ?",
                (base_path, datetime.now(timezone.utc), project_id),
            )
        except Exception:
            pass

        self._load_project(project)
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
        self.lbl_gallery_count.setText(f"{self._project.image_count} img")
        self._update_ui_state()
        return len(new)

    def _on_split_changed(self, image_path: str, split: str) -> None:
        try:
            self._ctrl.update_split(image_path, split)
            self.gallery.update_split(image_path, split)
        except Exception as e:
            self._log_db_error("Error al asignar split", e)

    def _load_project(self, project: Project):
        self._project = project
        self.gallery.load_images(project.image_paths)
        for img_path, ann in project.annotations.items():
            if ann.boxes:
                self.gallery.update_badge(img_path, len(ann.boxes))
        for file_path, split in self._ctrl.load_gallery_splits().items():
            self.gallery.update_split(file_path, split)
        self.lbl_gallery_count.setText(f"{project.image_count} img")
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

        # Borrar anotaciones y fila de imagen en una transaccion
        try:
            self._db.conn.execute("BEGIN TRANSACTION")
            self._db.conn.execute(
                "DELETE FROM annotations WHERE image_id IN "
                "(SELECT id FROM images WHERE file_path = ? AND project_id = ?)",
                (path, self._project_id),
            )
            self._db.conn.execute(
                "DELETE FROM images WHERE file_path = ? AND project_id = ?",
                (path, self._project_id),
            )
            self._db.conn.execute("COMMIT")
        except Exception as e:
            try:
                self._db.conn.execute("ROLLBACK")
            except Exception:
                pass
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
        self.lbl_gallery_count.setText(
            f"{self._project.image_count} img" if self._project.image_count else ""
        )
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
                "SELECT id, width, height FROM images WHERE file_path = ?",
                (self._current_image_path,)
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
        """Importa anotaciones desde un .txt YOLO para la imagen actual."""
        if not self._current_image_path or not self._annotation:
            return
        txt_path, _ = QFileDialog.getOpenFileName(
            self, "Importar anotaciones YOLO", "", "YOLO labels (*.txt)"
        )
        if not txt_path:
            return
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

    def on_export_dataset(self):
        if not self._project:
            QMessageBox.warning(self, "Sin proyecto", "Abre un proyecto con imágenes primero.")
            return
        annotated_count = sum(1 for ann in self._project.annotations.values() if ann.boxes)
        if annotated_count == 0:
            QMessageBox.warning(self, "Sin anotaciones", "No hay imágenes anotadas para exportar.")
            return
        output_dir = QFileDialog.getExistingDirectory(self, "Carpeta de destino del dataset")
        if not output_dir:
            return
        try:
            splits_from_db = {r[0]: r[1] for r in self._db.conn.execute(
                "SELECT file_path, split FROM images WHERE project_id = ?",
                (self._project_id,),
            ).fetchall()}

            yaml_path, final_splits = YoloExporter.export_dataset(
                annotations=self._project.annotations,
                classes=self._classes,
                output_dir=output_dir,
                splits=splits_from_db,
            )

            for img_path, split in final_splits.items():
                self._ctrl.update_split(img_path, split)
                self.gallery.update_split(img_path, split)

            train_count = sum(1 for s in final_splits.values() if s == "train")
            val_count   = sum(1 for s in final_splits.values() if s == "val")
            self.dataset_exported.emit(yaml_path)
            QMessageBox.information(
                self, "Dataset generado",
                f"Dataset listo en:\n{output_dir}\n\n"
                f"Train: {train_count} imágenes\nVal: {val_count} imágenes",
            )
        except Exception as e:
            QMessageBox.critical(self, "Error al exportar dataset", str(e))

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
        self.btn_export.setEnabled(has_image)
        self.btn_import_yolo.setEnabled(has_image)
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
