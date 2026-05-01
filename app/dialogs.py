from __future__ import annotations

from PyQt6.QtWidgets import (
    QDialog, QDialogButtonBox, QVBoxLayout, QHBoxLayout,
    QLabel, QPushButton, QListWidgetItem, QWidget, QComboBox,
    QMessageBox, QSizePolicy,
)
from PyQt6.QtCore import Qt, QSize
from PyQt6.QtGui import QColor, QCursor

from PyQt6.QtWidgets import QColorDialog

from app.annotation import LabelClass


class AddClassDialog(QDialog):
    """Diálogo que sugiere las clases existentes en la BD mediante un QComboBox."""

    new_class_confirmed: bool = False

    def __init__(
        self,
        suggested_color: str,
        db_classes: list[LabelClass],
        parent=None,
    ):
        super().__init__(parent)
        self.setWindowTitle("Seleccionar / nueva etiqueta")
        self.setFixedWidth(360)
        self.setModal(True)
        self._color = suggested_color
        self._db_class_names: list[str] = [c.name for c in db_classes]
        self._db_classes_by_name: dict[str, LabelClass] = {c.name: c for c in db_classes}

        layout = QVBoxLayout(self)
        layout.setSpacing(14)
        layout.setContentsMargins(20, 20, 20, 20)

        lbl_name = QLabel("Etiqueta")
        lbl_name.setStyleSheet(
            "color: #89b4fa; font-size: 11px; font-weight: bold; letter-spacing: 1px;"
        )
        layout.addWidget(lbl_name)

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

        hint = QLabel(f"{len(self._db_class_names)} clase(s) disponible(s) en la base de datos")
        hint.setStyleSheet("color: #6c7086; font-size: 11px; font-style: italic;")
        layout.addWidget(hint)

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

        buttons = QDialogButtonBox(
            QDialogButtonBox.StandardButton.Ok | QDialogButtonBox.StandardButton.Cancel
        )
        buttons.button(QDialogButtonBox.StandardButton.Ok).setText("Añadir")
        buttons.button(QDialogButtonBox.StandardButton.Cancel).setText("Cancelar")
        buttons.accepted.connect(self._on_accept)
        buttons.rejected.connect(self.reject)
        layout.addWidget(buttons)

        self._color_preview.clicked.connect(self._pick_color)
        self._combo.currentTextChanged.connect(self._on_combo_text_changed)
        self._combo.setFocus()

    def _on_combo_text_changed(self, text: str) -> None:
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

        if name not in self._db_class_names:
            reply = QMessageBox.question(
                self,
                "Clase no registrada",
                f"‘{name}’ no existe en la base de datos.\n"
                "¿Deseas crearla como nueva categoría oficial?",
                QMessageBox.StandardButton.Yes | QMessageBox.StandardButton.No,
                QMessageBox.StandardButton.No,
            )
            if reply == QMessageBox.StandardButton.No:
                return
            self.new_class_confirmed = True

        self.accept()

    def get_result(self) -> tuple[str, str]:
        """Devuelve (nombre, color_hex) elegidos por el usuario."""
        return self._combo.currentText().strip(), self._color


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
