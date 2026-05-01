from __future__ import annotations

import uuid
from datetime import datetime, timezone
from pathlib import Path

from PyQt6.QtWidgets import (
    QMainWindow, QWidget, QVBoxLayout, QHBoxLayout, QGridLayout,
    QLabel, QPushButton, QFrame, QScrollArea, QFileDialog,
    QDialog, QLineEdit, QDialogButtonBox, QMessageBox, QSizePolicy, QInputDialog,
)
from PyQt6.QtCore import Qt, pyqtSignal, QSize
from PyQt6.QtGui import QPixmap, QColor, QPainter, QFont

from app.database import DatabaseManager


# ─────────────────────────────────────────────────────────────────────────────
# Tarjeta de proyecto
# ─────────────────────────────────────────────────────────────────────────────

class ProjectCard(QFrame):
    clicked = pyqtSignal(str, str, str)   # project_id, base_path, name
    edit_requested = pyqtSignal(str, str)    # project_id, project_name

    _BORDER_NORMAL = "#313244"
    _BORDER_HOVER  = "#89b4fa"
    _BG            = "#1e1e2e"

    def __init__(self, project: dict, parent=None):
        super().__init__(parent)
        self._project = project
        self.setFixedSize(260, 210)
        self.setCursor(Qt.CursorShape.PointingHandCursor)
        self._apply_style(hover=False)
        self._build()

    def _apply_style(self, hover: bool):
        border = self._BORDER_HOVER if hover else self._BORDER_NORMAL
        self.setStyleSheet(f"""
            ProjectCard {{
                background: {self._BG};
                border: 2px solid {border};
                border-radius: 10px;
            }}
        """)

    def _build(self):
        layout = QVBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.setSpacing(0)

        # Thumbnail
        thumb = QLabel()
        thumb.setFixedSize(260, 120)
        thumb.setAlignment(Qt.AlignmentFlag.AlignCenter)
        thumb.setStyleSheet("background: #11111b; border-radius: 8px 8px 0 0;")

        sample = self._project.get("sample_image")
        if sample and Path(sample).exists():
            px = QPixmap(sample).scaled(
                260, 120,
                Qt.AspectRatioMode.KeepAspectRatioByExpanding,
                Qt.TransformationMode.SmoothTransformation,
            )
            # Recortar al centro
            x = (px.width()  - 260) // 2
            y = (px.height() - 120) // 2
            thumb.setPixmap(px.copy(x, y, 260, 120))
        else:
            thumb.setText("Sin imágenes")
            thumb.setStyleSheet(
                "background: #11111b; color: #45475a; font-size: 13px;"
                "border-radius: 8px 8px 0 0;"
            )

        layout.addWidget(thumb)

        # Info
        info = QWidget()
        info.setStyleSheet("background: transparent;")
        info_layout = QVBoxLayout(info)
        info_layout.setContentsMargins(12, 10, 12, 10)
        info_layout.setSpacing(3)

        name_lbl = QLabel(self._project["name"])
        name_lbl.setStyleSheet("color: #cdd6f4; font-size: 14px; font-weight: bold;")
        name_lbl.setWordWrap(True)
        info_layout.addWidget(name_lbl)

        imgs  = self._project.get("image_count", 0)
        anots = self._project.get("annotation_count", 0)
        stats = QLabel(f"{imgs} imágenes · {anots} anotaciones")
        stats.setStyleSheet("color: #6c7086; font-size: 11px;")
        info_layout.addWidget(stats)

        last = self._project.get("last_activity") or self._project.get("updated_at")
        if last:
            mod_str = self._format_date(last)
            mod_lbl = QLabel(f"Modificado: {mod_str}")
            mod_lbl.setStyleSheet("color: #45475a; font-size: 10px;")
            info_layout.addWidget(mod_lbl)

        created = self._project.get("created_at")
        if created:
            cr_lbl = QLabel(f"Creado: {self._format_date(created)}")
            cr_lbl.setStyleSheet("color: #45475a; font-size: 10px;")
            info_layout.addWidget(cr_lbl)

        actions = QHBoxLayout()
        actions.addStretch()
        self._btn_edit = QPushButton("Editar")
        self._btn_edit.setCursor(Qt.CursorShape.PointingHandCursor)
        self._btn_edit.setFixedHeight(22)
        self._btn_edit.setStyleSheet(
            "QPushButton { background: transparent; color: #89b4fa; border: 1px solid #89b4fa; "
            "border-radius: 5px; padding: 0 8px; font-size: 11px; }"
            "QPushButton:hover { background: rgba(137,180,250,0.15); color: #b4d0f7; border-color: #b4d0f7; }"
        )
        self._btn_edit.clicked.connect(self._on_edit_clicked)
        actions.addWidget(self._btn_edit)
        info_layout.addLayout(actions)

        layout.addWidget(info)

    @staticmethod
    def _format_date(value) -> str:
        if isinstance(value, datetime):
            return value.strftime("%d/%m/%Y %H:%M")
        try:
            return str(value)[:16].replace("T", " ")
        except Exception:
            return ""

    def enterEvent(self, event):
        self._apply_style(hover=True)
        super().enterEvent(event)

    def leaveEvent(self, event):
        self._apply_style(hover=False)
        super().leaveEvent(event)

    def mousePressEvent(self, event):
        if event.button() == Qt.MouseButton.LeftButton:
            click_pos = event.position().toPoint()
            if self._btn_edit.geometry().contains(click_pos):
                super().mousePressEvent(event)
                return
            self.clicked.emit(
                self._project["id"],
                self._project.get("base_path", ""),
                self._project["name"],
            )
        super().mousePressEvent(event)

    def _on_edit_clicked(self):
        self.edit_requested.emit(self._project["id"], self._project["name"])


# ─────────────────────────────────────────────────────────────────────────────
# Tarjeta "Nuevo proyecto"
# ─────────────────────────────────────────────────────────────────────────────

class NewProjectCard(QFrame):
    clicked = pyqtSignal()

    def __init__(self, parent=None):
        super().__init__(parent)
        self.setFixedSize(260, 210)
        self.setCursor(Qt.CursorShape.PointingHandCursor)
        self._apply_style(hover=False)

        layout = QVBoxLayout(self)
        layout.setAlignment(Qt.AlignmentFlag.AlignCenter)

        plus = QLabel("+")
        plus.setAlignment(Qt.AlignmentFlag.AlignCenter)
        plus.setStyleSheet("color: #89b4fa; font-size: 48px; font-weight: 200;")
        layout.addWidget(plus)

        lbl = QLabel("Nuevo proyecto")
        lbl.setAlignment(Qt.AlignmentFlag.AlignCenter)
        lbl.setStyleSheet("color: #89b4fa; font-size: 14px;")
        layout.addWidget(lbl)

    def _apply_style(self, hover: bool):
        border = "#89b4fa" if hover else "#313244"
        bg     = "#1e1e2e" if hover else "#181825"
        self.setStyleSheet(f"""
            NewProjectCard {{
                background: {bg};
                border: 2px dashed {border};
                border-radius: 10px;
            }}
        """)

    def enterEvent(self, event):
        self._apply_style(hover=True)
        super().enterEvent(event)

    def leaveEvent(self, event):
        self._apply_style(hover=False)
        super().leaveEvent(event)

    def mousePressEvent(self, event):
        if event.button() == Qt.MouseButton.LeftButton:
            self.clicked.emit()
        super().mousePressEvent(event)


# ─────────────────────────────────────────────────────────────────────────────
# Diálogo "Nuevo proyecto"
# ─────────────────────────────────────────────────────────────────────────────

class NewProjectDialog(QDialog):
    def __init__(self, parent=None):
        super().__init__(parent)
        self.setWindowTitle("Nuevo proyecto")
        self.setFixedWidth(420)
        self.setStyleSheet("""
            QDialog { background: #1e1e2e; }
            QLabel  { color: #cdd6f4; }
            QLineEdit {
                background: #313244; color: #cdd6f4;
                border: 1px solid #45475a; border-radius: 6px;
                padding: 6px 10px; font-size: 13px;
            }
            QPushButton {
                background: #89b4fa; color: #1e1e2e;
                border: none; border-radius: 6px;
                padding: 8px 20px; font-weight: bold;
            }
            QPushButton:hover { background: #b4d0f7; }
            QPushButton[flat="true"] {
                background: transparent; color: #6c7086; font-weight: normal;
            }
        """)

        layout = QVBoxLayout(self)
        layout.setSpacing(12)
        layout.setContentsMargins(24, 24, 24, 24)

        layout.addWidget(QLabel("Nombre del proyecto"))
        self.name_edit = QLineEdit()
        self.name_edit.setPlaceholderText("ej. Detección de coches")
        layout.addWidget(self.name_edit)

        folder_label = QLabel("Carpeta de imágenes  <span style='color:#6c7086;font-size:11px'>(opcional)</span>")
        folder_label.setTextFormat(Qt.TextFormat.RichText)
        layout.addWidget(folder_label)
        folder_row = QHBoxLayout()
        self.folder_edit = QLineEdit()
        self.folder_edit.setPlaceholderText("Se puede añadir después desde Etiquetado…")
        self.folder_edit.setReadOnly(True)
        folder_row.addWidget(self.folder_edit)
        btn_browse = QPushButton("Examinar")
        btn_browse.setFixedWidth(90)
        btn_browse.clicked.connect(self._browse)
        folder_row.addWidget(btn_browse)
        layout.addLayout(folder_row)

        buttons = QDialogButtonBox(
            QDialogButtonBox.StandardButton.Ok | QDialogButtonBox.StandardButton.Cancel
        )
        buttons.accepted.connect(self._accept)
        buttons.rejected.connect(self.reject)
        layout.addWidget(buttons)

    def _browse(self):
        folder = QFileDialog.getExistingDirectory(self, "Seleccionar carpeta de imágenes")
        if folder:
            self.folder_edit.setText(folder)
            if not self.name_edit.text():
                self.name_edit.setText(Path(folder).name)

    def _accept(self):
        if not self.name_edit.text().strip():
            QMessageBox.warning(self, "Campo requerido", "Introduce un nombre para el proyecto.")
            return
        self.accept()

    def get_result(self) -> tuple[str, str]:
        return self.name_edit.text().strip(), self.folder_edit.text()


# ─────────────────────────────────────────────────────────────────────────────
# Ventana principal del selector
# ─────────────────────────────────────────────────────────────────────────────

class ProjectSelectorWindow(QMainWindow):
    project_selected = pyqtSignal(str, str, str)   # project_id, base_path, name

    def __init__(self):
        super().__init__()
        self.setWindowTitle("VisionHub Desktop")
        self.setMinimumSize(900, 600)
        self.resize(1100, 700)

        self._db = DatabaseManager()
        self._db.initialize_tables()

        self._build_ui()
        self._load_projects()

    def _build_ui(self):
        central = QWidget()
        central.setStyleSheet("background: #181825;")
        self.setCentralWidget(central)

        root = QVBoxLayout(central)
        root.setContentsMargins(48, 40, 48, 40)
        root.setSpacing(0)

        # Header
        header = QHBoxLayout()

        title_col = QVBoxLayout()
        title_col.setSpacing(4)
        title = QLabel("VisionHub Desktop")
        title.setStyleSheet("color: #cdd6f4; font-size: 28px; font-weight: bold;")
        subtitle = QLabel("Selecciona un proyecto para continuar")
        subtitle.setStyleSheet("color: #6c7086; font-size: 14px;")
        title_col.addWidget(title)
        title_col.addWidget(subtitle)
        header.addLayout(title_col)
        header.addStretch()

        root.addLayout(header)
        root.addSpacing(32)

        # Scroll area con grid de tarjetas
        scroll = QScrollArea()
        scroll.setWidgetResizable(True)
        scroll.setFrameShape(QFrame.Shape.NoFrame)
        scroll.setStyleSheet("QScrollArea { background: transparent; border: none; }")

        self._cards_widget = QWidget()
        self._cards_widget.setStyleSheet("background: transparent;")
        self._grid = QGridLayout(self._cards_widget)
        self._grid.setSpacing(20)
        self._grid.setAlignment(Qt.AlignmentFlag.AlignTop | Qt.AlignmentFlag.AlignLeft)

        scroll.setWidget(self._cards_widget)
        root.addWidget(scroll, stretch=1)

    def _load_projects(self):
        # Limpiar grid
        while self._grid.count():
            item = self._grid.takeAt(0)
            if item.widget():
                item.widget().deleteLater()

        projects = self._db.get_all_projects_with_stats()

        col, cols = 0, 3

        # Tarjeta "Nuevo proyecto" siempre primera
        new_card = NewProjectCard()
        new_card.clicked.connect(self._on_new_project)
        self._grid.addWidget(new_card, 0, 0)
        col = 1

        for p in projects:
            row, c = divmod(col, cols)
            card = ProjectCard(p)
            card.clicked.connect(self.project_selected)
            card.edit_requested.connect(self._on_edit_project_name)
            self._grid.addWidget(card, row, c)
            col += 1

    def _on_new_project(self):
        dialog = NewProjectDialog(self)
        if dialog.exec() != QDialog.DialogCode.Accepted:
            return

        name, folder = dialog.get_result()
        project_id = self._db.create_project(name=name, base_path=folder)
        self.project_selected.emit(project_id, folder, name)

    def _on_edit_project_name(self, project_id: str, current_name: str):
        new_name, ok = QInputDialog.getText(
            self,
            "Editar nombre del proyecto",
            "Nuevo nombre:",
            QLineEdit.EchoMode.Normal,
            current_name,
        )
        if not ok:
            return
        new_name = new_name.strip()
        if not new_name:
            QMessageBox.warning(self, "Nombre inválido", "El nombre no puede estar vacío.")
            return
        if new_name == current_name:
            return
        try:
            self._db.rename_project(project_id, new_name)
            self._load_projects()
        except Exception as e:
            QMessageBox.critical(
                self,
                "Error al renombrar",
                f"No se pudo renombrar el proyecto.\n\nDetalle: {e}",
            )

    def closeEvent(self, event):
        self._db.close()
        super().closeEvent(event)
