from __future__ import annotations

from PyQt6.QtWidgets import (
    QMainWindow, QWidget, QHBoxLayout, QVBoxLayout,
    QFrame, QPushButton, QAbstractButton, QStackedWidget, QButtonGroup, QStatusBar,
)
from PyQt6.QtCore import Qt, QSize, QRectF, pyqtSignal
from PyQt6.QtGui import QPainter, QColor, QFont

from app.database import DatabaseManager
from app.analytics import AnalyticsWidget
from app.yolo_engine import TrainWidget, InferenceWidget
from app.annotation_controller import AnnotationController
from app.pages.labeling_page import LabelingPage


class NavButton(QAbstractButton):
    """Botón del nav rail con icono grande y texto pequeño, pintado manualmente."""

    def __init__(self, icon: str, label: str, parent=None):
        super().__init__(parent)
        self.setCheckable(True)
        self._icon = icon
        self._label = label
        self.setFixedSize(130, 90)
        self.setAttribute(Qt.WidgetAttribute.WA_Hover, True)

    def paintEvent(self, event):
        p = QPainter(self)
        p.setRenderHint(QPainter.RenderHint.Antialiasing)
        w, h = self.width(), self.height()

        if self.isChecked():
            p.fillRect(0, 0, w, h, QColor("#1e1e2e"))
            p.fillRect(0, 0, 4, h, QColor("#89b4fa"))
            text_color = QColor("#89b4fa")
        elif self.underMouse():
            p.fillRect(0, 0, w, h, QColor("#1e1e2e"))
            text_color = QColor("#cdd6f4")
        else:
            text_color = QColor("#6c7086")

        p.setPen(text_color)

        # Icono grande
        p.setFont(QFont("Segoe UI Emoji", 26))
        p.drawText(QRectF(0, 6, w, 46), Qt.AlignmentFlag.AlignHCenter, self._icon)

        # Texto pequeño
        p.setFont(QFont("Segoe UI", 10, QFont.Weight.Bold))
        p.drawText(QRectF(0, 54, w, 28), Qt.AlignmentFlag.AlignHCenter, self._label)

        p.end()

    def sizeHint(self) -> QSize:
        return QSize(130, 90)


class MainWindow(QMainWindow):

    back_to_projects = pyqtSignal()

    def __init__(
        self,
        db: DatabaseManager | None = None,
        project_id: str | None = None,
        base_path: str | None = None,
        project_name: str | None = None,
    ):
        super().__init__()
        self.setWindowTitle(
            f"VisionHub — {project_name}" if project_name else "VisionHub Desktop"
        )
        self.setMinimumSize(1280, 720)
        self.resize(1500, 900)

        self._db = db if db is not None else DatabaseManager()
        self._db.initialize_tables()

        if project_id:
            self._project_id = project_id
        else:
            self._project_id = "animales"
            AnnotationController(self._db, self._project_id).seed_default_project()

        ctrl = AnnotationController(self._db, self._project_id)
        self._build_ui(ctrl)
        self._connect_signals()

        if project_id:
            self.labeling_page.load_project_from_db(project_id, base_path or "")

    # ------------------------------------------------------------------ #
    # UI
    # ------------------------------------------------------------------ #

    def _build_ui(self, ctrl: AnnotationController):
        central = QWidget()
        self.setCentralWidget(central)
        outer = QHBoxLayout(central)
        outer.setContentsMargins(0, 0, 0, 0)
        outer.setSpacing(0)

        outer.addWidget(self._build_nav_rail())

        self._stack = QStackedWidget()
        outer.addWidget(self._stack)

        self.labeling_page  = LabelingPage(self._db, self._project_id, ctrl)
        self.analytics_page = AnalyticsWidget(db=self._db, project_id=self._project_id)
        self.train_page     = TrainWidget(db=self._db, project_id=self._project_id)
        self.inference_page = InferenceWidget()

        self._stack.addWidget(self.labeling_page)   # índice 0
        self._stack.addWidget(self.analytics_page)  # índice 1
        self._stack.addWidget(self.train_page)       # índice 2
        self._stack.addWidget(self.inference_page)  # índice 3

        self.status_bar = QStatusBar()
        self.setStatusBar(self.status_bar)
        self.status_bar.showMessage("Abre una imagen o carpeta para comenzar.")

    def _build_nav_rail(self) -> QFrame:
        rail = QFrame()
        rail.setObjectName("nav_rail")
        rail.setFixedWidth(130)
        rail.setStyleSheet("""
            #nav_rail { background: #11111b; border-right: 1px solid #313244; }
        """)
        layout = QVBoxLayout(rail)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.setSpacing(0)

        # — Volver a proyectos arriba del todo —
        btn_back = QPushButton("←\nVolver a\nproyectos")
        btn_back.setFixedSize(130, 80)
        btn_back.setToolTip("Volver a proyectos")
        btn_back.setStyleSheet("""
            QPushButton {
                background: #313244;
                border: none;
                border-bottom: 2px solid #89b4fa;
                color: #89b4fa;
                font-size: 13px;
                font-weight: bold;
                padding: 0;
                min-width: 0;
            }
            QPushButton:hover  { background: #45475a; color: #b4befe; }
            QPushButton:pressed { background: #585b70; }
        """)
        btn_back.clicked.connect(self.back_to_projects)
        layout.addWidget(btn_back)

        # — Botones de navegación —
        layout.addSpacing(8)
        self._btn_nav_labeling  = self._make_nav_button("🏷", "Labeling")
        self._btn_nav_analytics = self._make_nav_button("📊", "Analítica")
        self._btn_nav_train     = self._make_nav_button("🧠", "Train")
        self._btn_nav_inference = self._make_nav_button("👁", "Inferencia")
        self._btn_nav_labeling.setChecked(True)

        layout.addWidget(self._btn_nav_labeling)
        layout.addWidget(self._btn_nav_analytics)
        layout.addWidget(self._btn_nav_train)
        layout.addWidget(self._btn_nav_inference)
        layout.addStretch()

        self._nav_group = QButtonGroup(self)
        self._nav_group.addButton(self._btn_nav_labeling,  0)
        self._nav_group.addButton(self._btn_nav_analytics, 1)
        self._nav_group.addButton(self._btn_nav_train,     2)
        self._nav_group.addButton(self._btn_nav_inference, 3)

        return rail

    def _make_nav_button(self, icon: str, label: str) -> NavButton:
        return NavButton(icon, label)

    # ------------------------------------------------------------------ #
    # Señales
    # ------------------------------------------------------------------ #

    def _connect_signals(self):
        self._nav_group.idClicked.connect(self._on_nav_changed)
        self.labeling_page.status_message.connect(self.status_bar.showMessage)
        self.labeling_page.dataset_exported.connect(self.train_page.set_yaml)
        self.train_page.dataset_export_requested.connect(self.labeling_page.on_export_dataset)

    # ------------------------------------------------------------------ #
    # Navegación entre páginas
    # ------------------------------------------------------------------ #

    def _on_nav_changed(self, index: int):
        self._stack.setCurrentIndex(index)
        if index == 1:
            self.analytics_page.refresh_data()

    def start_training_from_paths(self, paths: list[str]):
        """Genera un dataset dinámico a partir de las rutas de imágenes y arranca el entrenamiento YOLO."""
        from PyQt6.QtWidgets import QMessageBox
        
        project = self.labeling_page._project
        if not project:
            QMessageBox.warning(self, "Aviso", "No hay un proyecto cargado.")
            return
            
        annotations = {
            p: project.annotations[p] 
            for p in paths 
            if p in project.annotations and project.annotations[p].boxes
        }
        
        if not annotations:
            QMessageBox.warning(self, "Aviso", "Ninguna de las imágenes pendientes tiene anotaciones válidas.")
            return
            
        classes = self.labeling_page._classes
        if not classes:
            QMessageBox.warning(self, "Aviso", "No hay clases definidas en el proyecto.")
            return
            
        import tempfile
        from app.yolo_exporter import YoloExporter
        
        temp_dir = tempfile.mkdtemp(prefix="visionhub_train_")
        
        try:
            yaml_path, _ = YoloExporter.export_dataset(
                annotations=annotations,
                classes=classes,
                output_dir=temp_dir,
                val_ratio=0.2
            )
            self.train_page.set_yaml(yaml_path)
            self.train_page._toggle_training()
        except Exception as e:
            QMessageBox.critical(self, "Error", f"Error al generar dataset para entrenamiento: {e}")

    # ------------------------------------------------------------------ #
    # Eventos de ventana
    # ------------------------------------------------------------------ #

    def keyPressEvent(self, event):
        if self._stack.currentIndex() == 0:
            if not self.labeling_page.handle_key(event):
                super().keyPressEvent(event)
        else:
            super().keyPressEvent(event)

    def closeEvent(self, event):
        self.inference_page.stop_camera()
        self.labeling_page.on_close()
        self._db.close()
        event.accept()
