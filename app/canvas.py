from __future__ import annotations
from PyQt6.QtWidgets import (
    QGraphicsView, QGraphicsScene, QGraphicsRectItem,
    QGraphicsTextItem, QGraphicsItem, QGraphicsPixmapItem,
)
from PyQt6.QtCore import Qt, QRectF, QPointF, pyqtSignal, QPoint
from PyQt6.QtGui import QPixmap, QPen, QColor, QBrush, QFont, QCursor, QPainter

from app.annotation import BoundingBox, LabelClass


class BBoxItem(QGraphicsRectItem):
    """Rectángulo de anotación interactivo con etiqueta de clase."""

    def __init__(self, bbox: BoundingBox, parent=None):
        super().__init__(parent)
        self.bbox = bbox
        self._apply_style()

        self.setRect(QRectF(bbox.x, bbox.y, bbox.width, bbox.height))
        self.setFlags(
            QGraphicsItem.GraphicsItemFlag.ItemIsSelectable |
            QGraphicsItem.GraphicsItemFlag.ItemIsMovable |
            QGraphicsItem.GraphicsItemFlag.ItemSendsGeometryChanges
        )
        self.setAcceptHoverEvents(True)

        self._label = QGraphicsTextItem(bbox.label_class.name, self)
        self._label.setDefaultTextColor(QColor(bbox.label_class.color))
        font = QFont("Segoe UI", 9, QFont.Weight.Bold)
        self._label.setFont(font)
        self._update_label_pos()

    def _apply_style(self):
        color = QColor(self.bbox.label_class.color)
        pen = QPen(color, 1)
        self.setPen(pen)
        fill = QColor(color)
        fill.setAlpha(55)
        self.setBrush(QBrush(fill))

    def update_class(self, new_class: LabelClass):
        self.bbox.label_class = new_class
        self._apply_style()
        self._label.setPlainText(new_class.name)
        self._label.setDefaultTextColor(QColor(new_class.color))

    def _update_label_pos(self):
        r = self.rect()
        self._label.setPos(r.x() + 2, r.y() - 20)

    def hoverEnterEvent(self, event):
        pen = self.pen()
        pen.setWidth(2)
        self.setPen(pen)
        self.setCursor(QCursor(Qt.CursorShape.SizeAllCursor))
        super().hoverEnterEvent(event)

    def hoverLeaveEvent(self, event):
        pen = self.pen()
        pen.setWidth(1)
        self.setPen(pen)
        self.unsetCursor()
        super().hoverLeaveEvent(event)

    def itemChange(self, change, value):
        if change == QGraphicsItem.GraphicsItemChange.ItemPositionHasChanged:
            r = self.rect()
            pos = self.pos()
            self.bbox.x = r.x() + pos.x()
            self.bbox.y = r.y() + pos.y()
            self._update_label_pos()
        return super().itemChange(change, value)

    def sync_bbox(self):
        r = self.rect()
        pos = self.pos()
        self.bbox.x = r.x() + pos.x()
        self.bbox.y = r.y() + pos.y()
        self.bbox.width = r.width()
        self.bbox.height = r.height()


class AnnotationCanvas(QGraphicsView):
    """Canvas principal para visualizar imagen y dibujar bounding boxes."""

    box_created = pyqtSignal(object)   # BoundingBox
    box_deleted = pyqtSignal(str)      # box id
    box_selected = pyqtSignal(object)  # BBoxItem or None
    view_changed = pyqtSignal()
    status_message = pyqtSignal(str)

    def __init__(self, parent=None):
        super().__init__(parent)
        self._scene = QGraphicsScene(self)
        self.setScene(self._scene)

        self.setRenderHint(QPainter.RenderHint.Antialiasing)
        self.setDragMode(QGraphicsView.DragMode.ScrollHandDrag)
        self.setTransformationAnchor(QGraphicsView.ViewportAnchor.AnchorUnderMouse)
        self.setResizeAnchor(QGraphicsView.ViewportAnchor.AnchorViewCenter)

        self._pixmap_item: QGraphicsPixmapItem | None = None
        self._img_width = 0
        self._img_height = 0

        self._drawing = False
        self._draw_mode = False
        self._origin: QPointF | None = None
        self._current_rect: QGraphicsRectItem | None = None
        self._active_class: LabelClass | None = None
        self._bbox_items: dict[str, BBoxItem] = {}
        self._fit_zoom: float = 1.0   # zoom en el último fitInView
        self._user_zoomed: bool = False

        self._scene.selectionChanged.connect(self._on_selection_changed)

    # ------------------------------------------------------------------ #
    # API pública
    # ------------------------------------------------------------------ #

    def load_image(self, path: str) -> bool:
        pixmap = QPixmap(path)
        if pixmap.isNull():
            return False
        self._scene.clear()
        self._bbox_items.clear()
        self._pixmap_item = self._scene.addPixmap(pixmap)
        self._img_width = pixmap.width()
        self._img_height = pixmap.height()
        self._scene.setSceneRect(QRectF(pixmap.rect()))
        self._user_zoomed = False
        super().fitInView(self._scene.sceneRect(), Qt.AspectRatioMode.KeepAspectRatio)
        self._fit_zoom = self.transform().m11()
        self.view_changed.emit()
        return True

    def set_draw_mode(self, enabled: bool):
        self._draw_mode = enabled
        if enabled:
            self.setDragMode(QGraphicsView.DragMode.NoDrag)
            self.setCursor(QCursor(Qt.CursorShape.CrossCursor))
            self.status_message.emit("Modo dibujo: clic y arrastra para crear un bounding box")
        else:
            self.setDragMode(QGraphicsView.DragMode.ScrollHandDrag)
            self.unsetCursor()
            self.status_message.emit("Modo navegación: rueda para zoom, arrastra para mover")

    def set_active_class(self, label_class: LabelClass | None):
        self._active_class = label_class

    def add_bbox_item(self, bbox: BoundingBox):
        item = BBoxItem(bbox)
        self._scene.addItem(item)
        self._bbox_items[bbox.id] = item

    def delete_selected(self):
        for item in list(self._scene.selectedItems()):
            if isinstance(item, BBoxItem):
                self._scene.removeItem(item)
                del self._bbox_items[item.bbox.id]
                self.box_deleted.emit(item.bbox.id)

    def delete_box_by_id(self, box_id: str):
        item = self._bbox_items.get(box_id)
        if item:
            self._scene.removeItem(item)
            del self._bbox_items[box_id]
            self.box_deleted.emit(box_id)

    def change_box_class(self, box_id: str, new_class: LabelClass):
        item = self._bbox_items.get(box_id)
        if item:
            item.update_class(new_class)

    def clear_all_boxes(self):
        for item in list(self._bbox_items.values()):
            self._scene.removeItem(item)
        self._bbox_items.clear()

    def clear_image(self):
        """Elimina la imagen actual y resetea el estado visual del canvas."""
        self._scene.clear()
        self._bbox_items.clear()
        self._pixmap_item = None
        self._img_width = 0
        self._img_height = 0
        self._user_zoomed = False
        self.resetTransform()
        self._fit_zoom = 1.0
        self.view_changed.emit()

    def get_image_size(self) -> tuple[int, int]:
        return self._img_width, self._img_height

    def has_image(self) -> bool:
        return self._pixmap_item is not None

    def get_zoom_level(self) -> float:
        """Devuelve el factor de escala actual (1.0 = 100%)."""
        return self.transform().m11()

    def set_zoom_level(self, factor: float):
        """Establece un zoom absoluto sin acumular transformaciones."""
        current = self.transform().m11()
        if current > 0:
            self.scale(factor / current, factor / current)

    def get_viewport_normalized_rect(self) -> QRectF | None:
        """Rectángulo del viewport en coordenadas normalizadas (0-1) respecto a la imagen."""
        if not self._pixmap_item:
            return None
        scene_rect = self.mapToScene(self.viewport().rect()).boundingRect()
        img = self._scene.sceneRect()
        if img.width() == 0 or img.height() == 0:
            return None
        return QRectF(
            (scene_rect.x() - img.x()) / img.width(),
            (scene_rect.y() - img.y()) / img.height(),
            scene_rect.width()  / img.width(),
            scene_rect.height() / img.height(),
        )

    def get_item_viewport_top_center(self, item: BBoxItem) -> QPoint:
        scene_rect = item.sceneBoundingRect()
        top_center = QPointF(scene_rect.center().x(), scene_rect.top())
        return self.mapFromScene(top_center)

    # ------------------------------------------------------------------ #
    # Slots internos
    # ------------------------------------------------------------------ #

    def _on_selection_changed(self):
        items = self._scene.selectedItems()
        selected = next((i for i in items if isinstance(i, BBoxItem)), None)
        self.box_selected.emit(selected)

    # ------------------------------------------------------------------ #
    # Eventos
    # ------------------------------------------------------------------ #

    def mousePressEvent(self, event):
        if self._draw_mode and event.button() == Qt.MouseButton.LeftButton:
            if not self._active_class:
                self.status_message.emit("Selecciona una clase antes de dibujar")
                return
            self._drawing = True
            self._origin = self.mapToScene(event.pos())
            pen = QPen(QColor(self._active_class.color), 2, Qt.PenStyle.DashLine)
            self._current_rect = self._scene.addRect(
                QRectF(self._origin, self._origin), pen
            )
        else:
            super().mousePressEvent(event)

    def mouseMoveEvent(self, event):
        if self._drawing and self._current_rect and self._origin:
            current = self.mapToScene(event.pos())
            self._current_rect.setRect(QRectF(self._origin, current).normalized())
        else:
            super().mouseMoveEvent(event)

    def mouseReleaseEvent(self, event):
        if self._drawing and event.button() == Qt.MouseButton.LeftButton:
            self._drawing = False
            if self._current_rect and self._origin and self._active_class:
                current = self.mapToScene(event.pos())
                rect = QRectF(self._origin, current).normalized()
                self._scene.removeItem(self._current_rect)
                self._current_rect = None
                bbox = BoundingBox(
                    x=rect.x(), y=rect.y(),
                    width=rect.width(), height=rect.height(),
                    label_class=self._active_class,
                )
                if bbox.is_valid:
                    self.add_bbox_item(bbox)
                    self.box_created.emit(bbox)
        else:
            super().mouseReleaseEvent(event)
            self.view_changed.emit()

    def fit_view(self):
        """Ajusta la vista a la imagen y marca que no hay zoom manual."""
        if not self._pixmap_item:
            return
        self._user_zoomed = False
        super().fitInView(self._scene.sceneRect(), Qt.AspectRatioMode.KeepAspectRatio)
        self._fit_zoom = self.transform().m11()
        self.view_changed.emit()

    def get_fit_zoom(self) -> float:
        return self._fit_zoom

    def scale(self, sx: float, sy: float):
        super().scale(sx, sy)
        self._user_zoomed = True
        self.view_changed.emit()

    _MAX_ZOOM = 15.0   # 1500%

    def wheelEvent(self, event):
        factor = 1.15 if event.angleDelta().y() > 0 else 1 / 1.15
        current = self.transform().m11()
        new_zoom = current * factor
        if new_zoom > self._MAX_ZOOM:
            factor = self._MAX_ZOOM / current
        elif new_zoom < self._fit_zoom:
            factor = self._fit_zoom / current
        self.scale(factor, factor)

    def scrollContentsBy(self, dx: int, dy: int):
        super().scrollContentsBy(dx, dy)
        self.view_changed.emit()

    def keyPressEvent(self, event):
        if event.key() == Qt.Key.Key_Delete:
            self.delete_selected()
        else:
            super().keyPressEvent(event)

    def resizeEvent(self, event):
        super().resizeEvent(event)
        if self._pixmap_item and not self._user_zoomed:
            super().fitInView(self._scene.sceneRect(), Qt.AspectRatioMode.KeepAspectRatio)
            self._fit_zoom = self.transform().m11()
            self.view_changed.emit()
