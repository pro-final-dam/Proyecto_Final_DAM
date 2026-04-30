from __future__ import annotations
from pathlib import Path

from PyQt6.QtWidgets import (
    QMainWindow, QWidget, QHBoxLayout, QVBoxLayout,
    QFrame, QLabel, QPushButton, QListWidget, QListWidgetItem,
    QFileDialog, QMessageBox, QSizePolicy,
    QStatusBar, QToolBar, QMenu, QColorDialog, QDialog, QDialogButtonBox,
    QLineEdit, QStackedWidget, QButtonGroup,
)
from PyQt6.QtCore import Qt, QSize, QPoint, pyqtSignal
from PyQt6.QtGui import QColor, QCursor

from app.canvas import AnnotationCanvas, BBoxItem
from app.annotation import BoundingBox, LabelClass, ImageAnnotation
from app.project import Project
from app.gallery import GalleryWidget
from app.yolo_exporter import YoloExporter
from app.styles import CLASS_COLORS


# ─────────────────────────────────────────────────────────────────────────────
# Overlay flotante sobre el bbox seleccionado
# ─────────────────────────────────────────────────────────────────────────────

class BBoxOverlay(QWidget):
    class_change_requested = pyqtSignal(str, object)
    delete_requested = pyqtSignal(str)

    def __init__(self, parent=None):
        super().__init__(parent)
        self.setAttribute(Qt.WidgetAttribute.WA_StyledBackground, True)
        self.setObjectName("bbox_overlay")
        self.setStyleSheet("""
            #bbox_overlay {
                background: #1e1e2e;
                border: 1px solid #45475a;
                border-radius: 8px;
            }
        """)
        self._box_id: str | None = None
        self._classes: list[LabelClass] = []

        layout = QHBoxLayout(self)
        layout.setContentsMargins(8, 5, 8, 5)
        layout.setSpacing(6)

        self._dot = QLabel()
        self._dot.setFixedSize(10, 10)
        layout.addWidget(self._dot, alignment=Qt.AlignmentFlag.AlignVCenter)

        self._class_btn = QPushButton()
        self._class_btn.setFlat(True)
        self._class_btn.setCursor(QCursor(Qt.CursorShape.PointingHandCursor))
        layout.addWidget(self._class_btn)

        sep = QFrame()
        sep.setFrameShape(QFrame.Shape.VLine)
        sep.setStyleSheet("color: #45475a;")
        sep.setFixedWidth(1)
        layout.addWidget(sep)

        self._trash_btn = QPushButton("✕")
        self._trash_btn.setFixedSize(24, 24)
        self._trash_btn.setCursor(QCursor(Qt.CursorShape.PointingHandCursor))
        self._trash_btn.setStyleSheet(
            "QPushButton { background: transparent; color: #f38ba8; border: none; "
            "border-radius: 4px; font-size: 13px; font-weight: bold; }"
            "QPushButton:hover { background: rgba(243,139,168,0.2); }"
        )
        layout.addWidget(self._trash_btn)

        self._class_btn.clicked.connect(self._show_class_menu)
        self._trash_btn.clicked.connect(self._on_delete)
        self.hide()

    def update_content(self, box_id: str, class_name: str, color: str, classes: list[LabelClass]):
        self._box_id = box_id
        self._classes = classes
        self._dot.setStyleSheet(
            f"background: {color}; border-radius: 5px;"
            f"min-width: 10px; max-width: 10px; min-height: 10px; max-height: 10px;"
        )
        self._class_btn.setText(f"{class_name}  ▾")
        self._class_btn.setStyleSheet(
            f"QPushButton {{ color: {color}; font-weight: bold; font-size: 12px; "
            f"border: none; padding: 0 2px; background: transparent; }}"
            f"QPushButton:hover {{ color: white; }}"
        )
        self.adjustSize()

    def _show_class_menu(self):
        menu = QMenu(self)
        menu.setStyleSheet("""
            QMenu { background: #1e1e2e; border: 1px solid #45475a;
                    border-radius: 6px; padding: 4px; color: #cdd6f4; }
            QMenu::item { padding: 6px 16px; border-radius: 4px; }
            QMenu::item:selected { background: #313244; color: #89b4fa; }
        """)
        for cls in self._classes:
            action = menu.addAction(cls.name)
            action.setData(cls)
        chosen = menu.exec(self._class_btn.mapToGlobal(self._class_btn.rect().bottomLeft()))
        if chosen and chosen.data() is not None:
            self.class_change_requested.emit(self._box_id, chosen.data())

    def _on_delete(self):
        if self._box_id:
            self.delete_requested.emit(self._box_id)
            self.hide()


# ─────────────────────────────────────────────────────────────────────────────
# Diálogo para crear nueva etiqueta
# ─────────────────────────────────────────────────────────────────────────────

class AddClassDialog(QDialog):
    def __init__(self, suggested_color: str, parent=None):
        super().__init__(parent)
        self.setWindowTitle("Nueva etiqueta")
        self.setFixedWidth(320)
        self.setModal(True)
        self._color = suggested_color

        layout = QVBoxLayout(self)
        layout.setSpacing(14)
        layout.setContentsMargins(20, 20, 20, 20)

        lbl_name = QLabel("Nombre de la etiqueta")
        lbl_name.setStyleSheet("color: #89b4fa; font-size: 11px; font-weight: bold; letter-spacing: 1px;")
        layout.addWidget(lbl_name)

        self._name_input = QLineEdit()
        self._name_input.setPlaceholderText("ej: persona, coche, perro…")
        layout.addWidget(self._name_input)

        lbl_color = QLabel("Color")
        lbl_color.setStyleSheet("color: #89b4fa; font-size: 11px; font-weight: bold; letter-spacing: 1px;")
        layout.addWidget(lbl_color)

        color_row = QHBoxLayout()
        self._color_preview = QPushButton()
        self._color_preview.setFixedSize(36, 36)
        self._color_preview.setCursor(QCursor(Qt.CursorShape.PointingHandCursor))
        self._color_preview.setToolTip("Haz clic para cambiar el color")
        self._apply_color_preview()
        color_row.addWidget(self._color_preview)

        self._color_label = QLabel(suggested_color)
        self._color_label.setStyleSheet("color: #6c7086; font-size: 12px;")
        color_row.addWidget(self._color_label)
        color_row.addStretch()
        layout.addLayout(color_row)

        buttons = QDialogButtonBox(
            QDialogButtonBox.StandardButton.Ok | QDialogButtonBox.StandardButton.Cancel
        )
        buttons.button(QDialogButtonBox.StandardButton.Ok).setText("Añadir")
        buttons.button(QDialogButtonBox.StandardButton.Cancel).setText("Cancelar")
        buttons.accepted.connect(self._on_accept)
        buttons.rejected.connect(self.reject)
        layout.addWidget(buttons)

        self._color_preview.clicked.connect(self._pick_color)
        self._name_input.setFocus()

    def _apply_color_preview(self):
        self._color_preview.setStyleSheet(
            f"QPushButton {{ background: {self._color}; border: 2px solid #45475a; "
            f"border-radius: 8px; min-width: 0; max-width: 36px; "
            f"min-height: 0; max-height: 36px; padding: 0; }}"
            f"QPushButton:hover {{ border-color: #89b4fa; }}"
        )

    def _pick_color(self):
        color = QColorDialog.getColor(QColor(self._color), self, "Elige color")
        if color.isValid():
            self._color = color.name()
            self._apply_color_preview()
            self._color_label.setText(self._color)

    def _on_accept(self):
        if not self._name_input.text().strip():
            self._name_input.setPlaceholderText("⚠ Escribe un nombre")
            self._name_input.setFocus()
            return
        self.accept()

    def get_result(self) -> tuple[str, str]:
        return self._name_input.text().strip(), self._color


# ─────────────────────────────────────────────────────────────────────────────
# Item de clase en la lista
# ─────────────────────────────────────────────────────────────────────────────

class ClassItemWidget(QWidget):
    def __init__(self, label_class: LabelClass, parent=None):
        super().__init__(parent)
        self.setStyleSheet("background: transparent;")
        layout = QHBoxLayout(self)
        layout.setContentsMargins(8, 6, 8, 6)
        layout.setSpacing(10)

        dot = QLabel()
        dot.setFixedSize(16, 16)
        dot.setStyleSheet(f"background: {label_class.color}; border-radius: 8px;")
        layout.addWidget(dot)

        name = QLabel(label_class.name)
        name.setStyleSheet("color: #cdd6f4; font-size: 14px;")
        name.setSizePolicy(QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Preferred)
        layout.addWidget(name)


class ClassListItem(QListWidgetItem):
    def __init__(self, label_class: LabelClass):
        super().__init__()
        self.label_class = label_class


# ─────────────────────────────────────────────────────────────────────────────
# Ventana principal
# ─────────────────────────────────────────────────────────────────────────────

class MainWindow(QMainWindow):

    def __init__(self):
        super().__init__()
        self.setWindowTitle("VisionHub Desktop — MVP Etiquetado")
        self.setMinimumSize(1280, 720)
        self.resize(1500, 900)

        self._project: Project | None = None
        self._current_image_path: str | None = None
        self._annotation: ImageAnnotation | None = None
        self._classes: list[LabelClass] = []
        self._selected_item: BBoxItem | None = None
        self._next_color_index = 0

        self._build_ui()
        self._overlay = BBoxOverlay(self.canvas.viewport())
        self._connect_signals()
        self._update_ui_state()

    # ------------------------------------------------------------------ #
    # UI
    # ------------------------------------------------------------------ #

    def _build_ui(self):
        toolbar = QToolBar()
        toolbar.setMovable(False)
        toolbar.setIconSize(QSize(18, 18))
        toolbar.setStyleSheet("QToolBar { spacing: 4px; padding: 4px 8px; }")
        self.addToolBar(toolbar)

        # — Abrir —
        self.btn_open_image = QPushButton("Cargar imagen")
        self.btn_open_image.setObjectName("btn_primary")
        toolbar.addWidget(self.btn_open_image)

        self.btn_open_folder = QPushButton("Cargar imágenes de directorio")
        toolbar.addWidget(self.btn_open_folder)

        self.btn_save = QPushButton("Guardar")
        toolbar.addWidget(self.btn_save)

        toolbar.addSeparator()

        # — Navegación —
        self.btn_prev = QPushButton("◀")
        self.btn_prev.setFixedWidth(32)
        self.btn_prev.setStyleSheet("QPushButton { min-width: 0; }")
        self.btn_prev.setToolTip("Imagen anterior (←)")
        toolbar.addWidget(self.btn_prev)

        self.lbl_nav = QLabel("—")
        self.lbl_nav.setStyleSheet("color: #6c7086; font-size: 12px; padding: 0 6px;")
        self.lbl_nav.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.lbl_nav.setMinimumWidth(60)
        toolbar.addWidget(self.lbl_nav)

        self.btn_next = QPushButton("▶")
        self.btn_next.setFixedWidth(32)
        self.btn_next.setStyleSheet("QPushButton { min-width: 0; }")
        self.btn_next.setToolTip("Imagen siguiente (→)")
        toolbar.addWidget(self.btn_next)

        toolbar.addSeparator()

        # — Herramientas —
        self.btn_draw = QPushButton("Dibujar bbox")
        self.btn_draw.setCheckable(True)
        self.btn_draw.setToolTip("Activar modo dibujo (D)")
        toolbar.addWidget(self.btn_draw)

        toolbar.addSeparator()

        # — Zoom —
        self.btn_zoom_fit = QPushButton("⊡")
        self.btn_zoom_fit.setFixedWidth(28)
        self.btn_zoom_fit.setStyleSheet("QPushButton { min-width: 0; font-size: 14px; }")
        self.btn_zoom_fit.setToolTip("Ajustar (F)")
        toolbar.addWidget(self.btn_zoom_fit)

        self.btn_zoom_in = QPushButton("+")
        self.btn_zoom_in.setFixedWidth(28)
        self.btn_zoom_in.setStyleSheet("QPushButton { min-width: 0; font-size: 14px; }")
        self.btn_zoom_in.setToolTip("Zoom +")
        toolbar.addWidget(self.btn_zoom_in)

        self.btn_zoom_out = QPushButton("−")
        self.btn_zoom_out.setFixedWidth(28)
        self.btn_zoom_out.setStyleSheet("QPushButton { min-width: 0; font-size: 14px; }")
        self.btn_zoom_out.setToolTip("Zoom −")
        toolbar.addWidget(self.btn_zoom_out)

        toolbar.addSeparator()

        # — Exportar —
        self.btn_export = QPushButton("Exportar YOLO")
        toolbar.addWidget(self.btn_export)

        # — Layout central: nav rail + stacked pages —
        central = QWidget()
        self.setCentralWidget(central)
        outer = QHBoxLayout(central)
        outer.setContentsMargins(0, 0, 0, 0)
        outer.setSpacing(0)

        outer.addWidget(self._build_nav_rail())

        self._stack = QStackedWidget()
        outer.addWidget(self._stack)

        # Página 0 — Labeling
        labeling_page = QWidget()
        page_layout = QHBoxLayout(labeling_page)
        page_layout.setContentsMargins(0, 0, 0, 0)
        page_layout.setSpacing(0)
        page_layout.addWidget(self._build_gallery_panel())
        self.canvas = AnnotationCanvas()
        self.canvas.setSizePolicy(QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Expanding)
        page_layout.addWidget(self.canvas)
        page_layout.addWidget(self._build_right_panel())
        self._stack.addWidget(labeling_page)

        # Página 1 — Train (placeholder)
        self._stack.addWidget(self._build_train_page())

        self.status_bar = QStatusBar()
        self.setStatusBar(self.status_bar)
        self.status_bar.showMessage("Abre una imagen o carpeta para comenzar.")

    def _build_nav_rail(self) -> QFrame:
        rail = QFrame()
        rail.setObjectName("nav_rail")
        rail.setFixedWidth(68)
        rail.setStyleSheet("""
            #nav_rail {
                background: #11111b;
                border-right: 1px solid #1e1e2e;
            }
        """)
        layout = QVBoxLayout(rail)
        layout.setContentsMargins(0, 12, 0, 12)
        layout.setSpacing(2)

        self._btn_nav_labeling = self._make_nav_button("🏷", "Labeling")
        self._btn_nav_train    = self._make_nav_button("🧠", "Train")
        self._btn_nav_labeling.setChecked(True)

        layout.addWidget(self._btn_nav_labeling)
        layout.addWidget(self._btn_nav_train)
        layout.addStretch()

        self._nav_group = QButtonGroup(self)
        self._nav_group.addButton(self._btn_nav_labeling, 0)
        self._nav_group.addButton(self._btn_nav_train, 1)

        return rail

    def _make_nav_button(self, icon: str, label: str) -> QPushButton:
        btn = QPushButton(f"{icon}\n{label}")
        btn.setCheckable(True)
        btn.setFixedSize(68, 64)
        btn.setStyleSheet("""
            QPushButton {
                background: transparent;
                border: none;
                border-left: 3px solid transparent;
                color: #585b70;
                font-size: 10px;
                padding: 0;
            }
            QPushButton:hover {
                color: #cdd6f4;
                background: #1e1e2e;
            }
            QPushButton:checked {
                color: #89b4fa;
                background: #1e1e2e;
                border-left: 3px solid #89b4fa;
            }
        """)
        return btn

    def _build_train_page(self) -> QWidget:
        page = QWidget()
        layout = QVBoxLayout(page)
        layout.setAlignment(Qt.AlignmentFlag.AlignCenter)
        layout.setSpacing(16)

        icon = QLabel("🧠")
        icon.setStyleSheet("font-size: 56px;")
        icon.setAlignment(Qt.AlignmentFlag.AlignCenter)
        layout.addWidget(icon)

        title = QLabel("Entrenamiento YOLO")
        title.setStyleSheet("color: #cdd6f4; font-size: 22px; font-weight: bold;")
        title.setAlignment(Qt.AlignmentFlag.AlignCenter)
        layout.addWidget(title)

        desc = QLabel(
            "Esta sección estará disponible en la Fase 4.\n"
            "Podrás lanzar un entrenamiento YOLOv8 con las imágenes etiquetadas,\n"
            "monitorizar métricas en tiempo real y evaluar el modelo."
        )
        desc.setStyleSheet("color: #6c7086; font-size: 13px;")
        desc.setAlignment(Qt.AlignmentFlag.AlignCenter)
        layout.addWidget(desc)

        return page

    def _build_gallery_panel(self) -> QFrame:
        panel = QFrame()
        panel.setObjectName("left_panel")
        panel.setFixedWidth(200)

        layout = QVBoxLayout(panel)
        layout.setContentsMargins(0, 10, 0, 0)
        layout.setSpacing(4)

        header = QHBoxLayout()
        header.setContentsMargins(10, 0, 10, 0)
        self.lbl_gallery_title = QLabel("GALERÍA")
        self.lbl_gallery_title.setObjectName("section_title")
        header.addWidget(self.lbl_gallery_title)
        header.addStretch()
        self.lbl_gallery_count = QLabel("")
        self.lbl_gallery_count.setStyleSheet("color: #6c7086; font-size: 11px;")
        header.addWidget(self.lbl_gallery_count)
        layout.addLayout(header)

        self.gallery = GalleryWidget()
        layout.addWidget(self.gallery)

        return panel

    def _build_right_panel(self) -> QFrame:
        panel = QFrame()
        panel.setObjectName("right_panel")
        panel.setFixedWidth(230)
        layout = QVBoxLayout(panel)
        layout.setContentsMargins(10, 12, 10, 12)
        layout.setSpacing(6)

        # — Clases —
        header_row = QHBoxLayout()
        lbl_classes = QLabel("CLASES")
        lbl_classes.setObjectName("section_title")
        header_row.addWidget(lbl_classes)
        header_row.addStretch()
        self.btn_remove_class = QPushButton("✕ Eliminar")
        self.btn_remove_class.setStyleSheet(
            "QPushButton { background: transparent; color: #f38ba8; border: none; "
            "font-size: 11px; min-width: 0; padding: 0 4px; }"
            "QPushButton:hover { color: #fab387; }"
        )
        header_row.addWidget(self.btn_remove_class)
        layout.addLayout(header_row)

        self.class_list = QListWidget()
        layout.addWidget(self.class_list)

        self.btn_add_class = QPushButton("+ Nueva etiqueta")
        self.btn_add_class.setObjectName("btn_primary")
        layout.addWidget(self.btn_add_class)

        layout.addSpacing(8)

        # — Anotaciones —
        ann_row = QHBoxLayout()
        lbl_ann = QLabel("ANOTACIONES")
        lbl_ann.setObjectName("section_title")
        ann_row.addWidget(lbl_ann)
        ann_row.addStretch()
        self.btn_clear_all = QPushButton("✕ Limpiar")
        self.btn_clear_all.setStyleSheet(
            "QPushButton { background: transparent; color: #6c7086; border: none; "
            "font-size: 11px; min-width: 0; padding: 0 4px; }"
            "QPushButton:hover { color: #f38ba8; }"
        )
        ann_row.addWidget(self.btn_clear_all)
        layout.addLayout(ann_row)

        self.lbl_box_count = QLabel("Bounding boxes: 0")
        self.lbl_box_count.setStyleSheet("color: #6c7086; font-size: 11px;")
        layout.addWidget(self.lbl_box_count)

        layout.addStretch()
        return panel

    # ------------------------------------------------------------------ #
    # Señales
    # ------------------------------------------------------------------ #

    def _connect_signals(self):
        self._nav_group.idClicked.connect(self._on_nav_changed)

        self.btn_open_image.clicked.connect(self._on_open_image)
        self.btn_open_folder.clicked.connect(self._on_open_folder)
        self.btn_save.clicked.connect(self._on_save_project)
        self.btn_prev.clicked.connect(self._on_prev)
        self.btn_next.clicked.connect(self._on_next)
        self.btn_draw.toggled.connect(self._on_toggle_draw_mode)
        self.btn_export.clicked.connect(self._on_export_yolo)

        self.btn_zoom_fit.clicked.connect(self._zoom_fit)
        self.btn_zoom_in.clicked.connect(self._zoom_in)
        self.btn_zoom_out.clicked.connect(self._zoom_out)

        self.btn_add_class.clicked.connect(self._on_add_class)
        self.btn_remove_class.clicked.connect(self._on_remove_class)
        self.class_list.currentItemChanged.connect(self._on_class_selected)
        self.btn_clear_all.clicked.connect(self._on_clear_all)

        self.gallery.image_selected.connect(self._on_gallery_image_selected)

        self.canvas.box_created.connect(self._on_box_created)
        self.canvas.box_deleted.connect(self._on_box_deleted)
        self.canvas.box_selected.connect(self._on_box_selected)
        self.canvas.view_changed.connect(self._reposition_overlay)
        self.canvas.status_message.connect(self.status_bar.showMessage)

        self._overlay.class_change_requested.connect(self._on_overlay_class_change)
        self._overlay.delete_requested.connect(self._on_overlay_delete)

    # ------------------------------------------------------------------ #
    # Abrir imagen / carpeta
    # ------------------------------------------------------------------ #

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
                self.status_bar.showMessage("La imagen ya estaba en el proyecto.")

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
            QMessageBox.information(self, "Sin imágenes",
                                    "No se encontraron imágenes en esa carpeta.")
            return
        if self._project is None:
            project = Project.open_folder(folder)
            self._load_project(project)
            self._navigate_to(project.image_paths[0], save_current=False)
        else:
            added = self._add_images_to_project(new_paths)
            self.status_bar.showMessage(
                f"{added} imagen(es) añadida(s)." if added
                else "No hay imágenes nuevas en ese directorio."
            )

    def _add_images_to_project(self, paths: list[str]) -> int:
        """Añade rutas nuevas al proyecto, actualiza la galería y devuelve el número añadido."""
        existing = set(self._project.image_paths)
        new = [p for p in paths if p not in existing]
        if not new:
            return 0
        self._project.image_paths = sorted(self._project.image_paths + new)
        self.gallery.load_images(self._project.image_paths)
        for img_path, ann in self._project.annotations.items():
            if ann.boxes:
                self.gallery.update_badge(img_path, len(ann.boxes))
        self.lbl_gallery_count.setText(f"{self._project.image_count} img")
        self._update_ui_state()
        return len(new)

    def _load_project(self, project: Project):
        self._project = project
        self._classes = list(project.classes)
        self._next_color_index = len(self._classes)

        # Reconstruir lista de clases en la UI
        self.class_list.clear()
        for cls in self._classes:
            self._add_class_to_list(cls)

        # Cargar galería
        self.gallery.load_images(project.image_paths)
        for img_path, ann in project.annotations.items():
            if ann.boxes:
                self.gallery.update_badge(img_path, len(ann.boxes))

        self.lbl_gallery_count.setText(f"{project.image_count} img")
        self._update_ui_state()

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

        self.canvas.clear_all_boxes()
        for box in self._annotation.boxes:
            self.canvas.add_bbox_item(box)

        self.gallery.set_current_by_path(image_path)
        self._refresh_box_count()
        self._update_nav_label()
        self._update_ui_state()

        name = Path(image_path).name
        self.status_bar.showMessage(f"{name}  —  {w} × {h} px")

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
    # Guardar
    # ------------------------------------------------------------------ #

    def _on_save_project(self):
        if not self._project:
            return
        self._commit_current_annotation()
        self._project.classes = self._classes
        self._project.save()
        self.status_bar.showMessage(
            f"Proyecto guardado en {self._project.folder}/visionhub_project.json"
        )

    # ------------------------------------------------------------------ #
    # Clases
    # ------------------------------------------------------------------ #

    def _on_add_class(self):
        suggested = CLASS_COLORS[self._next_color_index % len(CLASS_COLORS)]
        dialog = AddClassDialog(suggested_color=suggested, parent=self)
        if dialog.exec() != QDialog.DialogCode.Accepted:
            return
        name, color = dialog.get_result()
        if any(c.name == name for c in self._classes):
            self.status_bar.showMessage(f"La clase '{name}' ya existe.")
            return
        label_class = LabelClass(name=name, color=color, class_id=len(self._classes))
        self._classes.append(label_class)
        self._add_class_to_list(label_class)
        self.class_list.setCurrentRow(self.class_list.count() - 1)
        self._next_color_index += 1

    def _add_class_to_list(self, label_class: LabelClass):
        item = ClassListItem(label_class)
        widget = ClassItemWidget(label_class)
        item.setSizeHint(QSize(0, 34))
        self.class_list.addItem(item)
        self.class_list.setItemWidget(item, widget)

    def _on_remove_class(self):
        item = self.class_list.currentItem()
        if not item or not isinstance(item, ClassListItem):
            return
        self.class_list.takeItem(self.class_list.row(item))
        self._classes = [c for c in self._classes if c.name != item.label_class.name]
        if hasattr(self, '_active_class') and self._active_class and \
                self._active_class.name == item.label_class.name:
            self.canvas.set_active_class(None)

    def _on_class_selected(self, current, previous):
        if current and isinstance(current, ClassListItem):
            self.canvas.set_active_class(current.label_class)
            self.status_bar.showMessage(
                f"Clase activa: {current.label_class.name}  |  "
                "Activa modo dibujo y arrastra sobre la imagen"
            )

    # ------------------------------------------------------------------ #
    # Bboxes
    # ------------------------------------------------------------------ #

    def _on_box_created(self, bbox: BoundingBox):
        if self._annotation:
            self._annotation.add_box(bbox)
            self._refresh_box_count()

    def _on_box_deleted(self, box_id: str):
        if self._annotation:
            self._annotation.remove_box(box_id)
            self._refresh_box_count()

    def _on_clear_all(self):
        if not self._annotation:
            return
        reply = QMessageBox.question(
            self, "Confirmar", "¿Eliminar todos los bounding boxes?",
            QMessageBox.StandardButton.Yes | QMessageBox.StandardButton.No,
        )
        if reply == QMessageBox.StandardButton.Yes:
            self._overlay.hide()
            self._selected_item = None
            self.canvas.clear_all_boxes()
            self._annotation.boxes.clear()
            self._refresh_box_count()

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

    def _on_overlay_delete(self, box_id: str):
        self._selected_item = None
        self.canvas.delete_box_by_id(box_id)

    # ------------------------------------------------------------------ #
    # Dibujo y exportar
    # ------------------------------------------------------------------ #

    def _on_toggle_draw_mode(self, checked: bool):
        self.canvas.set_draw_mode(checked)
        self.btn_draw.setText("Dibujar  ■" if checked else "Dibujar bbox")
        if checked:
            self._overlay.hide()

    def _on_export_yolo(self):
        if not self._annotation or not self._annotation.boxes:
            QMessageBox.information(self, "Sin datos",
                                    "No hay anotaciones que exportar.")
            return
        try:
            out_file = YoloExporter.export(self._annotation)
            YoloExporter.export_classes(self._annotation)
            QMessageBox.information(self, "Exportación completada",
                                    f"Guardado en:\n{out_file}")
            self.status_bar.showMessage(f"Exportado: {out_file}")
        except Exception as e:
            QMessageBox.critical(self, "Error", str(e))

    # ------------------------------------------------------------------ #
    # Zoom
    # ------------------------------------------------------------------ #

    def _zoom_fit(self):
        if self.canvas.has_image():
            self.canvas.fitInView(
                self.canvas.scene().sceneRect(), Qt.AspectRatioMode.KeepAspectRatio
            )

    def _zoom_in(self):
        self.canvas.scale(1.25, 1.25)

    def _zoom_out(self):
        self.canvas.scale(1 / 1.25, 1 / 1.25)

    # ------------------------------------------------------------------ #
    # Estado y helpers
    # ------------------------------------------------------------------ #

    def _on_nav_changed(self, index: int):
        self._stack.setCurrentIndex(index)
        self._overlay.hide()
        self._update_ui_state()

    def _refresh_box_count(self):
        count = len(self._annotation.boxes) if self._annotation else 0
        self.lbl_box_count.setText(f"Bounding boxes: {count}")

    def _update_ui_state(self):
        on_labeling = self._stack.currentIndex() == 0
        has_project = self._project is not None
        has_image = self._current_image_path is not None
        multi = has_project and self._project.image_count > 1
        idx = self._project.index_of(self._current_image_path) if (has_project and has_image) else -1

        for btn in (self.btn_open_image, self.btn_open_folder, self.btn_save,
                    self.btn_prev, self.btn_next, self.btn_draw,
                    self.btn_zoom_fit, self.btn_zoom_in, self.btn_zoom_out,
                    self.btn_export):
            btn.setEnabled(on_labeling)

        if on_labeling:
            self.btn_save.setEnabled(has_project)
            self.btn_draw.setEnabled(has_image)
            self.btn_export.setEnabled(has_image)
            self.btn_clear_all.setEnabled(has_image)
            self.btn_zoom_fit.setEnabled(has_image)
            self.btn_zoom_in.setEnabled(has_image)
            self.btn_zoom_out.setEnabled(has_image)
            self.btn_prev.setEnabled(multi and idx > 0)
            self.btn_next.setEnabled(multi and idx < self._project.image_count - 1)

    # ------------------------------------------------------------------ #
    # Atajos de teclado
    # ------------------------------------------------------------------ #

    def keyPressEvent(self, event):
        key = event.key()
        if key == Qt.Key.Key_D:
            self.btn_draw.setChecked(not self.btn_draw.isChecked())
        elif key == Qt.Key.Key_F:
            self._zoom_fit()
        elif key == Qt.Key.Key_Left:
            self._on_prev()
        elif key == Qt.Key.Key_Right:
            self._on_next()
        else:
            super().keyPressEvent(event)

    def closeEvent(self, event):
        if self._project:
            self._commit_current_annotation()
            self._project.classes = self._classes
            self._project.save()
        event.accept()
