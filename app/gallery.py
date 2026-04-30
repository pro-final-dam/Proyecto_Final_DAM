from __future__ import annotations
from pathlib import Path

from PyQt6.QtWidgets import QListWidget, QListWidgetItem
from PyQt6.QtCore import Qt, pyqtSignal, QSize
from PyQt6.QtGui import QPixmap, QIcon, QColor

THUMB_W, THUMB_H = 60, 45


class GalleryWidget(QListWidget):
    """Lista lateral de imágenes con thumbnails y badge de anotaciones."""

    image_selected = pyqtSignal(str)   # ruta absoluta de la imagen

    def __init__(self, parent=None):
        super().__init__(parent)
        self.setIconSize(QSize(THUMB_W, THUMB_H))
        self.setSpacing(2)
        self.setUniformItemSizes(True)
        self.setStyleSheet("""
            QListWidget {
                background: #11111b;
                border: none;
                outline: none;
            }
            QListWidget::item {
                color: #6c7086;
                padding: 4px 6px;
                border-radius: 4px;
                font-size: 11px;
            }
            QListWidget::item:selected {
                background: #313244;
                color: #cdd6f4;
            }
            QListWidget::item:hover {
                background: #1e1e2e;
                color: #cdd6f4;
            }
        """)
        self._loading = False
        self.currentItemChanged.connect(self._on_current_changed)

    def load_images(self, image_paths: list[str]):
        self._loading = True
        self.clear()
        for path in image_paths:
            item = QListWidgetItem()
            item.setData(Qt.ItemDataRole.UserRole, path)
            item.setText(Path(path).name)
            pixmap = QPixmap(path)
            if not pixmap.isNull():
                pixmap = pixmap.scaled(
                    THUMB_W, THUMB_H,
                    Qt.AspectRatioMode.KeepAspectRatio,
                    Qt.TransformationMode.SmoothTransformation,
                )
            item.setIcon(QIcon(pixmap))
            self.addItem(item)
        self._loading = False

    def set_current_by_path(self, image_path: str):
        self._loading = True
        for i in range(self.count()):
            item = self.item(i)
            if item.data(Qt.ItemDataRole.UserRole) == image_path:
                self.setCurrentItem(item)
                self.scrollToItem(item)
                break
        self._loading = False

    def update_badge(self, image_path: str, box_count: int):
        """Muestra el número de bboxes junto al nombre y colorea el item."""
        for i in range(self.count()):
            item = self.item(i)
            if item.data(Qt.ItemDataRole.UserRole) == image_path:
                name = Path(image_path).name
                item.setText(f"{name}  [{box_count}]" if box_count > 0 else name)
                item.setForeground(
                    QColor("#a6e3a1") if box_count > 0 else QColor("#6c7086")
                )
                break

    def _on_current_changed(self, current: QListWidgetItem, previous: QListWidgetItem):
        if current and not self._loading:
            self.image_selected.emit(current.data(Qt.ItemDataRole.UserRole))
