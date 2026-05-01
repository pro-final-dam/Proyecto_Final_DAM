from __future__ import annotations
from pathlib import Path

from PyQt6.QtWidgets import (
    QMainWindow, QWidget, QHBoxLayout, QVBoxLayout,
    QFrame, QLabel, QPushButton, QListWidget, QListWidgetItem,
    QFileDialog, QMessageBox, QSizePolicy,
    QStatusBar, QToolBar, QMenu, QColorDialog, QDialog, QDialogButtonBox,
    QLineEdit, QStackedWidget, QButtonGroup, QComboBox,
)
from PyQt6.QtCore import Qt, QSize, QPoint, pyqtSignal
from PyQt6.QtGui import QColor, QCursor

from app.canvas import AnnotationCanvas, BBoxItem
from app.annotation import BoundingBox, LabelClass, ImageAnnotation
from app.project import Project
from app.gallery import GalleryWidget
from app.yolo_exporter import YoloExporter
from app.styles import CLASS_COLORS
# [DB] Importamos el gestor de base de datos DuckDB
from app.database import DatabaseManager

from app.analytics import AnalyticsWidget
from app.yolo_engine import TrainWidget, InferenceWidget


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
# Diálogo para crear / seleccionar etiqueta (con QComboBox de la BD)
# ─────────────────────────────────────────────────────────────────────────────

class AddClassDialog(QDialog):
    """Diálogo que sugiere las clases existentes en la BD mediante un QComboBox.

    Args:
        suggested_color: Color hex sugerido para clases nuevas.
        db_classes:      Lista de LabelClass ya registradas en DuckDB.
                         Se usa para rellenar el combo y para detectar
                         si el nombre introducido es nuevo.
        parent:          Widget padre Qt.
    """

    # Resultado de la validación: True = clase nueva confirmada por usuario
    new_class_confirmed: bool = False

    def __init__(
        self,
        suggested_color: str,
        db_classes: list,          # list[LabelClass]
        parent=None,
    ):
        super().__init__(parent)
        self.setWindowTitle("Seleccionar / nueva etiqueta")
        self.setFixedWidth(360)
        self.setModal(True)
        self._color = suggested_color
        # [DB] Clases conocidas en la BD; usamos el nombre como clave de búsqueda
        self._db_class_names: list[str] = [c.name for c in db_classes]
        self._db_classes_by_name: dict[str, object] = {c.name: c for c in db_classes}

        layout = QVBoxLayout(self)
        layout.setSpacing(14)
        layout.setContentsMargins(20, 20, 20, 20)

        # — Etiqueta + combo —
        lbl_name = QLabel("Etiqueta")
        lbl_name.setStyleSheet(
            "color: #89b4fa; font-size: 11px; font-weight: bold; letter-spacing: 1px;"
        )
        layout.addWidget(lbl_name)

        # [DB] QComboBox editable: permite seleccionar o escribir libremente
        self._combo = QComboBox()
        self._combo.setEditable(True)
        self._combo.setInsertPolicy(QComboBox.InsertPolicy.NoInsert)
        self._combo.addItems(self._db_class_names)
        self._combo.setCurrentIndex(-1)
        self._combo.lineEdit().setPlaceholderText("Selecciona o escribe un nombre…")
        self._combo.setStyleSheet(
            "QComboBox { background: #313244; color: #cdd6f4; border: 1px solid #45475a;"
            " border-radius: 6px; padding: 6px 10px; font-size: 13px; }"
            "QComboBox QAbstractItemView { background: #1e1e2e; color: #cdd6f4;"
            " selection-background-color: #45475a; }"
        )
        layout.addWidget(self._combo)

        # [DB] Hint informativo
        hint = QLabel(f"{len(self._db_class_names)} clase(s) disponible(s) en la base de datos")
        hint.setStyleSheet("color: #6c7086; font-size: 11px; font-style: italic;")
        layout.addWidget(hint)

        # — Color —
        lbl_color = QLabel("Color")
        lbl_color.setStyleSheet(
            "color: #89b4fa; font-size: 11px; font-weight: bold; letter-spacing: 1px;"
        )
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

        # — Botones —
        buttons = QDialogButtonBox(
            QDialogButtonBox.StandardButton.Ok | QDialogButtonBox.StandardButton.Cancel
        )
        buttons.button(QDialogButtonBox.StandardButton.Ok).setText("Añadir")
        buttons.button(QDialogButtonBox.StandardButton.Cancel).setText("Cancelar")
        buttons.accepted.connect(self._on_accept)
        buttons.rejected.connect(self.reject)
        layout.addWidget(buttons)

        self._color_preview.clicked.connect(self._pick_color)
        # [DB] Sincronizar color del combo al seleccionar clase existente
        self._combo.currentTextChanged.connect(self._on_combo_text_changed)
        self._combo.setFocus()

    # ------------------------------------------------------------------ #

    def _on_combo_text_changed(self, text: str) -> None:
        """Rellena el color automáticamente si se selecciona una clase de la BD."""
        cls = self._db_classes_by_name.get(text)
        if cls:
            self._color = cls.color
            self._apply_color_preview()
            self._color_label.setText(self._color)

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
        name = self._combo.currentText().strip()
        if not name:
            self._combo.lineEdit().setPlaceholderText("⚠ Elige o escribe un nombre")
            self._combo.setFocus()
            return

        # [DB] Validación preventiva: si el nombre NO está en la BD, confirmar
        if name not in self._db_class_names:
            reply = QMessageBox.question(
                self,
                "Clase no registrada",
                f"\u2018{name}\u2019 no existe en la base de datos.\n"
                "¿Deseas crearla como nueva categoría oficial?",
                QMessageBox.StandardButton.Yes | QMessageBox.StandardButton.No,
                QMessageBox.StandardButton.No,   # Botón por defecto: No (más seguro)
            )
            if reply == QMessageBox.StandardButton.No:
                return   # El usuario cancela → no cerrar el diálogo
            # [DB] Marcamos que esta clase es nueva y debe insertarse en BD
            self.new_class_confirmed = True

        self.accept()

    def get_result(self) -> tuple[str, str]:
        """Devuelve (nombre, color_hex) elegidos por el usuario."""
        return self._combo.currentText().strip(), self._color


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

        # [DB] Inicializar base de datos y crear tablas si no existen
        self._db = DatabaseManager()
        self._db.initialize_tables()
        # [DB] Sembrar proyecto y clases por defecto al arrancar
        self._db_seed_default_project()

        self._build_ui()
        self._overlay = BBoxOverlay(self.canvas.viewport())
        self._connect_signals()
        # [DB] Cargar clases de la BD en el panel lateral (sincronización inicial)
        self._db_load_classes_into_ui()
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

        # Página 1 — Analytics
        self.analytics_page = AnalyticsWidget(db=self._db, project_id=self._PROJECT_ID)
        self._stack.addWidget(self.analytics_page)

        # Página 2 — Train
        self.train_page = TrainWidget()
        self._stack.addWidget(self.train_page)

        # Página 3 — Inference
        self.inference_page = InferenceWidget()
        self._stack.addWidget(self.inference_page)

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
        self._btn_nav_analytics = self._make_nav_button("📊", "Analítica")
        self._btn_nav_train    = self._make_nav_button("🧠", "Train")
        self._btn_nav_inference = self._make_nav_button("👁", "Inferencia")
        self._btn_nav_labeling.setChecked(True)

        layout.addWidget(self._btn_nav_labeling)
        layout.addWidget(self._btn_nav_analytics)
        layout.addWidget(self._btn_nav_train)
        layout.addWidget(self._btn_nav_inference)
        layout.addStretch()

        self._nav_group = QButtonGroup(self)
        self._nav_group.addButton(self._btn_nav_labeling, 0)
        self._nav_group.addButton(self._btn_nav_analytics, 1)
        self._nav_group.addButton(self._btn_nav_train, 2)
        self._nav_group.addButton(self._btn_nav_inference, 3)

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
        self.btn_draw.clicked.connect(self._on_draw_clicked)
        self.btn_export.clicked.connect(self._on_export_yolo)  # [DB] Única conexión, sin duplicados

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
    # [DB] Lógica de siembra inicial (proyecto 'default' y clases YOLO)
    # ------------------------------------------------------------------ #

    def _db_seed_default_project(self) -> None:
        """Crea el proyecto 'animales' y sus clases YOLO si no existen aún.

        Diseño idempotente: si ya existen, no hace nada.
        ID de proyecto fijo: 'animales' (clave de negocio legible).
        """
        import uuid
        from datetime import datetime, timezone

        now = datetime.now(timezone.utc)
        # [DB] ID canónico del proyecto — debe coincidir en TODAS las tablas
        PROJECT_ID = "animales"

        # --- Proyecto ---
        existing_project = self._db.conn.execute(
            "SELECT id FROM projects WHERE id = ?",
            (PROJECT_ID,),
        ).fetchone()

        if existing_project is None:
            # [DB] El proyecto no existe → lo creamos
            self._db.conn.execute(
                """
                INSERT INTO projects (id, name, description, base_path, created_at, updated_at)
                VALUES (?, ?, ?, ?, ?, ?)
                """,
                (
                    PROJECT_ID,                             # id = 'animales'
                    "animales",                             # nombre del proyecto
                    "Proyecto de clasificación de animales",
                    "",                                     # base_path (sin carpeta fija aún)
                    now,
                    now,
                ),
            )

        # --- Clases YOLO (perros=0, gatos=1, caballos=2) ---
        # [DB] Definición canónica: (nombre, yolo_index, color_hex)
        _DEFAULT_CLASSES = [
            ("perros",   0, "#f38ba8"),
            ("gatos",    1, "#a6e3a1"),
            ("caballos", 2, "#89b4fa"),
        ]

        for class_name, yolo_idx, color in _DEFAULT_CLASSES:
            existing_cls = self._db.conn.execute(
                "SELECT id FROM classes WHERE project_id = ? AND yolo_index = ?",
                (PROJECT_ID, yolo_idx),
            ).fetchone()

            if existing_cls is None:
                # [DB] La clase no existe → la insertamos con un UUID propio
                self._db.conn.execute(
                    """
                    INSERT INTO classes (id, project_id, name, color, yolo_index)
                    VALUES (?, ?, ?, ?, ?)
                    """,
                    (
                        str(uuid.uuid4()),  # UUID único para la FK relacional
                        PROJECT_ID,
                        class_name,
                        color,
                        yolo_idx,
                    ),
                )

    # [DB] Constante de proyecto usada en upsert e imagen
    _PROJECT_ID = "animales"

    def _db_upsert_image(self, image_path: str, width: int, height: int) -> str:
        """Garantiza que la imagen esté registrada en la tabla images.

        Si ya existe (por file_path), devuelve su id sin modificarla.
        Si no existe, la inserta y devuelve el nuevo id.

        Returns:
            UUID de la fila en la tabla images.
        """
        import uuid
        from datetime import datetime, timezone

        # [DB] Buscar por ruta de archivo (clave de negocio)
        row = self._db.conn.execute(
            "SELECT id FROM images WHERE file_path = ?",
            (image_path,),
        ).fetchone()

        if row is not None:
            return row[0]   # Ya existe → devolvemos su UUID

        # [DB] No existe → insertar nueva fila
        new_id = str(uuid.uuid4())
        self._db.conn.execute(
            """
            INSERT INTO images
                (id, project_id, file_path, split, width, height, status, created_at)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?)
            """,
            (
                new_id,
                self._PROJECT_ID,   # [DB] 'animales' — consistente con el proyecto
                image_path,
                "train",            # split por defecto
                width,
                height,
                "annotated",
                datetime.now(timezone.utc),
            ),
        )
        return new_id

    def _db_resolve_class_uuid(
        self,
        label_class: "LabelClass",
    ) -> str | None:
        """Devuelve el UUID de BD para una clase, con confirmación interactiva.

        Flujo:
        1. Busca la clase en la BD por yolo_index dentro del proyecto 'animales'.
        2. Si existe, devuelve su UUID silenciosamente.
        3. Si NO existe, muestra QMessageBox.question al usuario:
           - 'Sí': la registra en la BD y devuelve el nuevo UUID.
           - 'No': devuelve None (el bbox se omite del guardado en BD).

        Args:
            label_class: Objeto LabelClass de la anotación actual.

        Returns:
            UUID (str) si la clase está o se ha creado, None si el usuario cancela.
        """
        import uuid

        # [DB] Lookup relacional: yolo_index dentro del proyecto 'animales'
        row = self._db.conn.execute(
            "SELECT id FROM classes WHERE project_id = ? AND yolo_index = ?",
            (self._PROJECT_ID, label_class.class_id),
        ).fetchone()

        if row is not None:
            return row[0]   # Clase ya registrada → devolvemos su UUID

        # [DB] La clase NO está en la BD → preguntamos al usuario antes de crear nada
        reply = QMessageBox.question(
            self,
            "Clase no registrada",
            f"La clase \u2018{label_class.name}\u2019 no existe en la base de datos.\n"
            "\u00bfDeseas crearla como una nueva categoría oficial?",
            QMessageBox.StandardButton.Yes | QMessageBox.StandardButton.No,
            QMessageBox.StandardButton.No,   # Botón por defecto: No (más seguro)
        )

        if reply == QMessageBox.StandardButton.No:
            # [DB] El usuario canceló → se omite este bbox del guardado en BD
            return None

        # [DB] El usuario aceptó → insertar con parámetros ? (sin SQL injection)
        new_uuid = str(uuid.uuid4())
        self._db.conn.execute(
            """
            INSERT INTO classes (id, project_id, name, color, yolo_index)
            VALUES (?, ?, ?, ?, ?)
            """,
            (
                new_uuid,
                self._PROJECT_ID,       # [DB] 'animales'
                label_class.name,
                label_class.color,
                label_class.class_id,
            ),
        )
        # [DB] Confirmación visual en la barra de estado
        self.status_bar.showMessage(
            f"✅ Clase \u2018{label_class.name}\u2019 dada de alta en la base de datos."
        )
        return new_uuid

    # ------------------------------------------------------------------ #
    # [DB] Carga inicial de clases desde DuckDB a la UI
    # ------------------------------------------------------------------ #

    def refresh_classes_from_db(self) -> None:
        """Lee la tabla classes desde DuckDB y sincroniza self._classes + el QListWidget.

        Debe llamarse siempre que algo pueda haber cambiado el estado de las clases:
        - Al arrancar la aplicación.
        - Al cargar una imagen o carpeta (_load_project).
        - Después de crear una nueva clase en _on_add_class.

        Es completamente idempotente: limpia primero y reconstruye desde cero.
        """
        rows = self._db.conn.execute(
            """
            SELECT id, name, color, yolo_index
            FROM   classes
            WHERE  project_id = ?
            ORDER  BY yolo_index
            """,
            (self._PROJECT_ID,),
        ).fetchall()

        self._classes.clear()
        self.class_list.clear()

        for _uuid, name, color, yolo_idx in rows:
            # [DB] Construimos LabelClass con yolo_index como class_id para el canvas
            lc = LabelClass(name=name, color=color, class_id=yolo_idx)
            self._classes.append(lc)
            self._add_class_to_list(lc)

        self._next_color_index = len(self._classes)
        if self._classes:
            self.status_bar.showMessage(
                f"✅ {len(self._classes)} clase(s) cargada(s) desde la base de datos."
            )

    # Alias interno para compatibilidad con el arranque inicial
    _db_load_classes_into_ui = refresh_classes_from_db

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
        # [DB] NO usamos project.classes: la BD es la fuente de verdad.
        # refresh_classes_from_db() se encarga de limpiar y recargar desde DuckDB.

        # Cargar galería
        self.gallery.load_images(project.image_paths)
        for img_path, ann in project.annotations.items():
            if ann.boxes:
                self.gallery.update_badge(img_path, len(ann.boxes))

        self.lbl_gallery_count.setText(f"{project.image_count} img")
        self._update_ui_state()

        # [DB] Refrescar clases desde la BD SIEMPRE al cargar un proyecto/imagen
        self.refresh_classes_from_db()

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

        # [DB] Upsert: registrar la imagen en la BD si no existe todavía
        self._db_upsert_image(image_path, w, h)

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
        # [DB] Pasamos las clases de la BD al diálogo para que rellene el combo
        dialog = AddClassDialog(
            suggested_color=suggested,
            db_classes=self._classes,
            parent=self,
        )
        if dialog.exec() != QDialog.DialogCode.Accepted:
            return
        name, color = dialog.get_result()
        if any(c.name == name for c in self._classes):
            self.status_bar.showMessage(f"La clase \u2018{name}\u2019 ya está activa en la UI.")
            return
        yolo_idx = len(self._classes)
        label_class = LabelClass(name=name, color=color, class_id=yolo_idx)
        self._classes.append(label_class)
        self._add_class_to_list(label_class)
        self.class_list.setCurrentRow(self.class_list.count() - 1)
        self._next_color_index += 1

        # [DB] Evitar duplicados: comprobar si el nombre YA existe en la BD
        if dialog.new_class_confirmed:
            import uuid
            existing_row = self._db.conn.execute(
                "SELECT id FROM classes WHERE project_id = ? AND name = ?",
                (self._PROJECT_ID, name),
            ).fetchone()

            if existing_row is not None:
                # El nombre ya existe en BD → reutilizamos ese UUID, no insertamos
                self.status_bar.showMessage(
                    f"ℹ️ Clase \u2018{name}\u2019 ya existía en BD (reutilizada, sin duplicar)."
                )
            else:
                # No existe → insertar con UUID nuevo
                self._db.conn.execute(
                    """
                    INSERT INTO classes (id, project_id, name, color, yolo_index)
                    VALUES (?, ?, ?, ?, ?)
                    """,
                    (
                        str(uuid.uuid4()),
                        self._PROJECT_ID,   # [DB] 'animales'
                        name,
                        color,
                        yolo_idx,
                    ),
                )
                self.status_bar.showMessage(
                    f"✅ Clase \u2018{name}\u2019 registrada en la BD y añadida a la UI."
                )
        else:
            # Clase existente seleccionada del combo → ya está en BD
            self.status_bar.showMessage(
                f"Clase \u2018{name}\u2019 añadida a la UI (ya existía en BD)."
            )

        # [DB] Refrescar la UI desde la BD para garantizar consistencia total
        self.refresh_classes_from_db()

        # [UX] Seleccionar automáticamente la clase para que el usuario pueda empezar a dibujar
        for i in range(self.class_list.count()):
            item = self.class_list.item(i)
            if isinstance(item, ClassListItem) and item.label_class.name == name:
                self.class_list.setCurrentItem(item)
                break

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

    def _db_persist_current_annotation(self) -> None:
        """Guarda en DuckDB el estado actual de anotaciones para la imagen visible.

        Diseño Database-First: se llama cada vez que se crea, elimina o modifica
        un bbox, en lugar de esperar a 'Exportar'. Usa replace=True para que la
        BD siempre refleje exactamente lo que hay en el canvas.

        Es un no-op seguro si no hay imagen cargada o la anotación está vacía.
        """
        if not self._annotation or not self._current_image_path:
            return

        w = self._annotation.image_width
        h = self._annotation.image_height

        # [DB] Upsert de la imagen para garantizar la FK
        image_db_id = self._db_upsert_image(self._current_image_path, w, h)

        # [DB] Mapear yolo_index → UUID de la tabla classes para cada bbox
        boxes_for_db: list[BoundingBox] = []
        for box in self._annotation.boxes:
            cls_row = self._db.conn.execute(
                "SELECT id FROM classes WHERE project_id = ? AND yolo_index = ?",
                (self._PROJECT_ID, box.label_class.class_id),
            ).fetchone()
            if cls_row is None:
                # Clase aún no persistida (no debería ocurrir con el nuevo flujo)
                continue
            boxes_for_db.append(
                BoundingBox(
                    x=box.x, y=box.y,
                    width=box.width, height=box.height,
                    label_class=LabelClass(
                        name=box.label_class.name,
                        color=box.label_class.color,
                        class_id=cls_row[0],   # UUID para la FK relacional
                    ),
                    id=box.id,
                )
            )

        # [DB] Construir el ImageAnnotation auxiliar y guardar (replace=True)
        annotation_for_db = ImageAnnotation(
            image_path=self._current_image_path,
            image_width=w,
            image_height=h,
            boxes=boxes_for_db,
        )
        try:
            self._db.save_image_annotation(
                image_id=image_db_id,
                annotation=annotation_for_db,
                replace=False,  # NO usamos replace=True; la BD debe acumular anotaciones.
            )
        except Exception as exc:
            # No interrumpimos el flujo de UI por un error de BD; sí lo notificamos
            self.status_bar.showMessage(f"⚠ Error al guardar en BD: {exc}")

    def _on_box_created(self, bbox: BoundingBox):
        if self._annotation:
            self._annotation.add_box(bbox)
            self._refresh_box_count()
            # [DB] Persistencia inmediata tras crear un bbox
            self._db_persist_current_annotation()

    def _on_box_deleted(self, box_id: str):
        if self._annotation:
            self._annotation.remove_box(box_id)
            self._refresh_box_count()
            # [DB] Borrar explícitamente el bbox de la BD
            try:
                self._db.conn.execute("DELETE FROM annotations WHERE id = ?", (box_id,))
            except Exception as e:
                self.status_bar.showMessage(f"⚠ Error al borrar en BD: {e}")

    def _on_clear_all(self):
        if not self._annotation:
            return
        reply = QMessageBox.question(
            self, "Confirmar", "¿Eliminar todos los bounding boxes?",
            QMessageBox.StandardButton.Yes | QMessageBox.StandardButton.No,
        )
        if reply == QMessageBox.StandardButton.Yes:
            # [DB] Borrar explícitamente los bboxes actuales de la BD
            try:
                for box in self._annotation.boxes:
                    self._db.conn.execute("DELETE FROM annotations WHERE id = ?", (box.id,))
            except Exception as e:
                self.status_bar.showMessage(f"⚠ Error al borrar en BD: {e}")

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
        # [DB] Persistencia inmediata tras cambio de clase en un bbox
        self._db_persist_current_annotation()

    def _on_overlay_delete(self, box_id: str):
        self._selected_item = None
        self.canvas.delete_box_by_id(box_id)

    # ------------------------------------------------------------------ #
    # Dibujo y exportar
    # ------------------------------------------------------------------ #

    def _on_draw_clicked(self):
        """Gestiona el clic en 'Dibujar bbox' con validación de clase activa.

        Flujo:
        1. Si el modo dibujo YA estaba activo → lo desactiva (toggle off).
        2. Si se intenta activar SIN clase seleccionada → aviso + no activa.
        3. Si hay clase seleccionada → activa el modo dibujo en el canvas.
        """
        # Estado actual: ¿estaba activo antes del clic?
        # btn_draw es checkable; Qt invierte el estado antes de emitir clicked,
        # así que self.btn_draw.isChecked() ya refleja el NUEVO estado deseado.
        want_draw = self.btn_draw.isChecked()

        if want_draw:
            # Verificar si hay clase seleccionada
            current_item = self.class_list.currentItem()
            if not current_item or not isinstance(current_item, ClassListItem):
                # Sin clase → mostrar aviso y revertir el estado del botón
                QMessageBox.warning(
                    self,
                    "Sin clase seleccionada",
                    "Debes seleccionar una clase en la lista de la derecha "
                    "(por ejemplo: perros, gatos, caballos) antes de dibujar.",
                )
                self.btn_draw.setChecked(False)   # revertir el toggle
                return

            # Activar modo dibujo con la clase activa
            self.canvas.set_draw_mode(True)
            self.btn_draw.setText("✏️  Dibujando...")
            self._overlay.hide()
            self.status_bar.showMessage(
                f"✏️ Modo dibujo activo — clase: {current_item.label_class.name}  "
                "| Arrastra para crear un bbox | Clic en el botón para salir"
            )
        else:
            # Desactivar modo dibujo
            self.canvas.set_draw_mode(False)
            self.btn_draw.setText("Dibujar bbox")

    def _on_export_yolo(self):
        """Exporta las anotaciones a archivos .txt YOLO.

        Genera el archivo completo basándose en TODO lo que hay en la BD para
        esa imagen, no sólo lo que hay en la sesión actual.
        """
        if not self._current_image_path:
            QMessageBox.information(self, "Sin imagen", "No hay imagen cargada.")
            return

        try:
            # 1. Recuperar info de imagen desde BD
            row = self._db.conn.execute(
                "SELECT id, width, height FROM images WHERE file_path = ?",
                (self._current_image_path,)
            ).fetchone()
            
            if not row:
                raise ValueError("La imagen no está registrada en la base de datos.")
            
            img_id, w, h = row
            
            # 2. Recuperar anotaciones de la BD
            db_boxes = self._db.get_annotations_for_image(img_id)
            if not db_boxes:
                QMessageBox.information(self, "Sin datos", "No hay anotaciones en BD que exportar.")
                return

            # 3. Construir ImageAnnotation completo
            full_annotation = ImageAnnotation(
                image_path=self._current_image_path,
                image_width=w,
                image_height=h,
            )

            for b in db_boxes:
                cls_row = self._db.conn.execute(
                    "SELECT name, color, yolo_index FROM classes WHERE id = ?",
                    (b["class_id"],)
                ).fetchone()
                if not cls_row:
                    continue
                c_name, c_color, c_yolo = cls_row

                # Revertir de coordenadas normalizadas YOLO a píxeles
                bw = b["width"] * w
                bh = b["height"] * h
                bx = (b["x_center"] * w) - (bw / 2)
                by = (b["y_center"] * h) - (bh / 2)

                bbox = BoundingBox(
                    x=bx, y=by, width=bw, height=bh,
                    label_class=LabelClass(name=c_name, color=c_color, class_id=c_yolo),
                    id=b["id"]
                )
                full_annotation.add_box(bbox)

            # 4. Exportar el .txt de YOLO con las anotaciones combinadas
            out_file = YoloExporter.export(full_annotation)
            
            # 5. Generar classes.txt con TODAS las clases del proyecto, en orden
            classes_rows = self._db.conn.execute(
                "SELECT name FROM classes WHERE project_id = ? ORDER BY yolo_index",
                (self._PROJECT_ID,)
            ).fetchall()
            
            classes_lines = [r[0] for r in classes_rows]
            from pathlib import Path
            out_classes = Path(self._current_image_path).parent / "classes.txt"
            out_classes.write_text("\n".join(classes_lines) + "\n", encoding="utf-8")

            QMessageBox.information(self, "Exportación completada",
                                    f"Generado archivo combinando anotaciones de la BD:\n{out_file}")
            self.status_bar.showMessage(f"Exportado desde BD: {out_file}")
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
        if index == 1:
            self.analytics_page.refresh_data()

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
        # [DB] Cerrar la conexión a DuckDB de forma limpia al salir
        self._db.close()
        event.accept()
