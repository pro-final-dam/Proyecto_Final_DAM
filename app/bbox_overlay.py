from __future__ import annotations

from PyQt6.QtWidgets import QWidget, QHBoxLayout, QLabel, QPushButton, QFrame, QMenu
from PyQt6.QtCore import Qt, pyqtSignal
from PyQt6.QtGui import QCursor

from app.annotation import LabelClass


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
