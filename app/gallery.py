from __future__ import annotations
from pathlib import Path

from PyQt6.QtWidgets import QListWidget, QListWidgetItem, QMenu
from PyQt6.QtCore import Qt, pyqtSignal, QSize, QPoint
from PyQt6.QtGui import QPixmap, QIcon, QColor

THUMB_W, THUMB_H = 60, 45

_ROLE_PATH  = Qt.ItemDataRole.UserRole
_ROLE_SPLIT = Qt.ItemDataRole.UserRole + 1
_ROLE_COUNT = Qt.ItemDataRole.UserRole + 2

_SPLIT_PREFIX = {'train': '[T]', 'val': '[V]', 'unassigned': ''}
_SPLIT_COLOR  = {'train': '#89b4fa', 'val': '#f9e2af'}  # azul / amarillo


class GalleryWidget(QListWidget):
    """Lista lateral de imágenes con thumbnails, badge de anotaciones y selector de split."""

    image_selected = pyqtSignal(str)        # ruta absoluta de la imagen
    split_changed  = pyqtSignal(str, str)   # (image_path, nuevo_split)

    def __init__(self, parent=None):
        super().__init__(parent)
        self.setIconSize(QSize(THUMB_W, THUMB_H))
        self.setSpacing(2)
        self.setUniformItemSizes(True)
        self.setContextMenuPolicy(Qt.ContextMenuPolicy.CustomContextMenu)
        self.customContextMenuRequested.connect(self._show_context_menu)
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

    # ------------------------------------------------------------------ #
    # Carga
    # ------------------------------------------------------------------ #

    def load_images(self, image_paths: list[str]):
        self._loading = True
        self.clear()
        for path in image_paths:
            item = QListWidgetItem()
            item.setData(_ROLE_PATH,  path)
            item.setData(_ROLE_SPLIT, 'unassigned')
            item.setData(_ROLE_COUNT, 0)
            pixmap = QPixmap(path)
            if not pixmap.isNull():
                pixmap = pixmap.scaled(
                    THUMB_W, THUMB_H,
                    Qt.AspectRatioMode.KeepAspectRatio,
                    Qt.TransformationMode.SmoothTransformation,
                )
            item.setIcon(QIcon(pixmap))
            self._refresh_item(item)
            self.addItem(item)
        self._loading = False

    def set_current_by_path(self, image_path: str):
        self._loading = True
        for i in range(self.count()):
            item = self.item(i)
            if item.data(_ROLE_PATH) == image_path:
                self.setCurrentItem(item)
                self.scrollToItem(item)
                break
        self._loading = False

    # ------------------------------------------------------------------ #
    # Actualización de badges
    # ------------------------------------------------------------------ #

    def update_badge(self, image_path: str, box_count: int):
        """Actualiza el número de bboxes visible junto al nombre."""
        for i in range(self.count()):
            item = self.item(i)
            if item.data(_ROLE_PATH) == image_path:
                item.setData(_ROLE_COUNT, box_count)
                self._refresh_item(item)
                break

    def update_split(self, image_path: str, split: str):
        """Actualiza el badge de split ([T]/[V]) de una imagen."""
        for i in range(self.count()):
            item = self.item(i)
            if item.data(_ROLE_PATH) == image_path:
                item.setData(_ROLE_SPLIT, split)
                self._refresh_item(item)
                break

    def _refresh_item(self, item: QListWidgetItem):
        """Recalcula texto y color del item a partir de sus datos internos."""
        path  = item.data(_ROLE_PATH)  or ''
        split = item.data(_ROLE_SPLIT) or 'unassigned'
        count = item.data(_ROLE_COUNT) or 0

        name   = Path(path).name
        prefix = _SPLIT_PREFIX.get(split, '')
        badge  = f'  [{count}]' if count > 0 else ''
        item.setText(f'{prefix} {name}{badge}' if prefix else f'{name}{badge}')

        if split in _SPLIT_COLOR:
            item.setForeground(QColor(_SPLIT_COLOR[split]))
        elif count > 0:
            item.setForeground(QColor('#a6e3a1'))
        else:
            item.setForeground(QColor('#6c7086'))

    # ------------------------------------------------------------------ #
    # Menú contextual de split
    # ------------------------------------------------------------------ #

    def _show_context_menu(self, pos: QPoint):
        item = self.itemAt(pos)
        if not item:
            return

        image_path    = item.data(_ROLE_PATH)
        current_split = item.data(_ROLE_SPLIT) or 'unassigned'

        menu = QMenu(self)
        menu.setStyleSheet("""
            QMenu {
                background: #1e1e2e;
                color: #cdd6f4;
                border: 1px solid #45475a;
                border-radius: 6px;
            }
            QMenu::item { padding: 6px 20px; }
            QMenu::item:selected { background: #313244; }
            QMenu::separator { height: 1px; background: #45475a; margin: 4px 0; }
        """)

        act_train = menu.addAction('🔵  Train')
        act_val   = menu.addAction('🟡  Validación')
        menu.addSeparator()
        act_unset = menu.addAction('✕  Sin asignar')

        for act, split in [(act_train, 'train'), (act_val, 'val'), (act_unset, 'unassigned')]:
            act.setCheckable(True)
            act.setChecked(current_split == split)

        chosen = menu.exec(self.mapToGlobal(pos))

        mapping = {act_train: 'train', act_val: 'val', act_unset: 'unassigned'}
        if chosen in mapping:
            self.split_changed.emit(image_path, mapping[chosen])

    # ------------------------------------------------------------------ #
    # Señal de selección
    # ------------------------------------------------------------------ #

    def _on_current_changed(self, current: QListWidgetItem, previous: QListWidgetItem):
        if current and not self._loading:
            self.image_selected.emit(current.data(_ROLE_PATH))
