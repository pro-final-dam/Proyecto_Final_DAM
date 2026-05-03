from __future__ import annotations
from pathlib import Path

from PyQt6.QtWidgets import QListWidget, QListWidgetItem, QMenu, QStyle, QStyledItemDelegate
from PyQt6.QtCore import Qt, pyqtSignal, QSize, QPoint, QRect, QThread
from PyQt6.QtGui import QPixmap, QIcon, QColor, QImage, QPainter, QFont

THUMB_W, THUMB_H = 64, 64   # miniaturas cuadradas

_ROLE_PATH  = Qt.ItemDataRole.UserRole
_ROLE_SPLIT = Qt.ItemDataRole.UserRole + 1
_ROLE_COUNT = Qt.ItemDataRole.UserRole + 2

_SPLIT_COLOR = {
    'train': '#89b4fa',   # azul
    'val':   '#f9e2af',   # amarillo
    'test':  '#cba6f7',   # malva
}
_SPLIT_LETTER = {'train': 'T', 'val': 'V', 'test': 'P'}


class _GalleryDelegate(QStyledItemDelegate):
    """Dibuja solo la miniatura + tag de split + badge de conteo."""

    def sizeHint(self, option, index) -> QSize:
        return QSize(THUMB_W + 8, THUMB_H + 8)

    def paint(self, painter: QPainter, option, index) -> None:
        painter.save()
        rect: QRect = option.rect

        # Fondo
        selected = bool(option.state & QStyle.StateFlag.State_Selected)
        hovered  = bool(option.state & QStyle.StateFlag.State_MouseOver)
        if selected:
            painter.fillRect(rect, QColor('#313244'))
        elif hovered:
            painter.fillRect(rect, QColor('#1e1e2e'))
        else:
            painter.fillRect(rect, QColor('#11111b'))

        # Miniatura centrada
        icon = index.data(Qt.ItemDataRole.DecorationRole)
        if icon and not icon.isNull():
            pix: QPixmap = icon.pixmap(THUMB_W, THUMB_H)
            if not pix.isNull():
                px = rect.x() + (rect.width()  - pix.width())  // 2
                py = rect.y() + (rect.height() - pix.height()) // 2
                painter.drawPixmap(px, py, pix)

        # Tag de split — pequeño pill en esquina inferior derecha
        split = index.data(_ROLE_SPLIT) or 'unassigned'
        split_color = _SPLIT_COLOR.get(split)
        split_letter = _SPLIT_LETTER.get(split, '')
        if split_color and split_letter:
            tag_w, tag_h = 14, 14
            tag_x = rect.right()  - tag_w - 3
            tag_y = rect.bottom() - tag_h - 3
            tag_rect = QRect(tag_x, tag_y, tag_w, tag_h)
            painter.setRenderHint(QPainter.RenderHint.Antialiasing, True)
            painter.setBrush(QColor(split_color))
            painter.setPen(Qt.PenStyle.NoPen)
            painter.drawRoundedRect(tag_rect, 4, 4)
            font = QFont()
            font.setPixelSize(9)
            font.setBold(True)
            painter.setFont(font)
            painter.setPen(QColor('#11111b'))
            painter.drawText(tag_rect, Qt.AlignmentFlag.AlignCenter, split_letter)

        painter.restore()


class _ThumbnailLoader(QThread):
    thumbnail_ready = pyqtSignal(str, QImage)

    def __init__(self, paths: list[str], parent=None):
        super().__init__(parent)
        self._paths = paths
        self._cancelled = False

    def cancel(self) -> None:
        self._cancelled = True

    def run(self) -> None:
        for path in self._paths:
            if self._cancelled:
                return
            img = QImage(path)
            if not img.isNull():
                img = img.scaled(
                    THUMB_W, THUMB_H,
                    Qt.AspectRatioMode.KeepAspectRatio,
                    Qt.TransformationMode.SmoothTransformation,
                )
            self.thumbnail_ready.emit(path, img)


class GalleryWidget(QListWidget):
    image_selected = pyqtSignal(str)
    split_changed  = pyqtSignal(str, str)

    def __init__(self, parent=None):
        super().__init__(parent)
        self._thumb_loader: _ThumbnailLoader | None = None
        self._index: dict[str, QListWidgetItem] = {}
        self._active_filter: str | None = None
        self.setItemDelegate(_GalleryDelegate(self))
        self.setIconSize(QSize(THUMB_W, THUMB_H))
        self.setSpacing(2)
        self.setUniformItemSizes(True)
        self.setMouseTracking(True)
        self.setContextMenuPolicy(Qt.ContextMenuPolicy.CustomContextMenu)
        self.customContextMenuRequested.connect(self._show_context_menu)
        self.setStyleSheet("""
            QListWidget {
                background: #11111b;
                border: none;
                outline: none;
            }
            QListWidget::item { border-radius: 4px; }
        """)
        self._loading = False
        self.currentItemChanged.connect(self._on_current_changed)

    # ------------------------------------------------------------------ #
    # Carga
    # ------------------------------------------------------------------ #

    def load_images(self, image_paths: list[str]) -> None:
        self._cancel_loader()
        self._loading = True
        self.clear()
        self._index.clear()
        for path in image_paths:
            item = QListWidgetItem()
            item.setData(_ROLE_PATH,  path)
            item.setData(_ROLE_SPLIT, 'unassigned')
            item.setData(_ROLE_COUNT, 0)
            self.addItem(item)
            self._index[path] = item
        self._loading = False

        if image_paths:
            self._thumb_loader = _ThumbnailLoader(image_paths)
            self._thumb_loader.thumbnail_ready.connect(self._on_thumbnail_ready)
            self._thumb_loader.start()

    def _cancel_loader(self) -> None:
        if self._thumb_loader is not None:
            try:
                self._thumb_loader.thumbnail_ready.disconnect(self._on_thumbnail_ready)
            except Exception:
                pass
            self._thumb_loader.cancel()
            self._thumb_loader = None

    def _on_thumbnail_ready(self, path: str, image: QImage) -> None:
        item = self._index.get(path)
        if item is None or image.isNull():
            return
        item.setIcon(QIcon(QPixmap.fromImage(image)))

    def set_current_by_path(self, image_path: str) -> None:
        item = self._index.get(image_path)
        if item is not None:
            self._loading = True
            self.setCurrentItem(item)
            self.scrollToItem(item)
            self._loading = False

    # ------------------------------------------------------------------ #
    # Actualización de datos
    # ------------------------------------------------------------------ #

    def update_badge(self, image_path: str, box_count: int) -> None:
        item = self._index.get(image_path)
        if item is not None:
            item.setData(_ROLE_COUNT, box_count)
            self.update(self.indexFromItem(item))

    def set_split_filter(self, split: str | None) -> None:
        self._active_filter = split
        for path, item in self._index.items():
            if split is None:
                item.setHidden(False)
            else:
                item.setHidden((item.data(_ROLE_SPLIT) or 'unassigned') != split)

    def update_split(self, image_path: str, split: str) -> None:
        item = self._index.get(image_path)
        if item is not None:
            item.setData(_ROLE_SPLIT, split)
            if self._active_filter is not None:
                item.setHidden(split != self._active_filter)
            self.update(self.indexFromItem(item))

    # ------------------------------------------------------------------ #
    # Menú contextual de split
    # ------------------------------------------------------------------ #

    def _show_context_menu(self, pos: QPoint) -> None:
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
        act_test  = menu.addAction('🟣  Pruebas')
        menu.addSeparator()
        act_unset = menu.addAction('✕  Sin asignar')

        for act, split in [(act_train, 'train'), (act_val, 'val'), (act_test, 'test'), (act_unset, 'unassigned')]:
            act.setCheckable(True)
            act.setChecked(current_split == split)

        chosen = menu.exec(self.mapToGlobal(pos))

        mapping = {act_train: 'train', act_val: 'val', act_test: 'test', act_unset: 'unassigned'}
        if chosen in mapping:
            self.split_changed.emit(image_path, mapping[chosen])

    # ------------------------------------------------------------------ #
    # Señal de selección
    # ------------------------------------------------------------------ #

    def _on_current_changed(self, current: QListWidgetItem, previous: QListWidgetItem) -> None:
        if current and not self._loading:
            self.image_selected.emit(current.data(_ROLE_PATH))
