# VisionHub Desktop — Guía para Claude Code

## Descripción del proyecto

Herramienta de escritorio en Python/PyQt6 para etiquetar imágenes con bounding boxes y exportar en formato YOLO. MVP funcional.

## Stack técnico

- **Python 3.11+** con type hints modernos (`X | Y`, `list[X]`)
- **PyQt6** — widgets, signals/slots, QSS
- **QGraphicsView / QGraphicsScene** — canvas principal
- **QGraphicsRectItem** — bounding boxes interactivos
- Sin base de datos, sin entrenamiento

## Arquitectura

```
app/annotation.py    # Modelos de datos puros (dataclasses). Sin PyQt6.
app/canvas.py        # Vista gráfica. Emite señales, no modifica modelos.
app/main_window.py   # Orquesta UI + lógica. Conecta canvas <-> modelos.
app/styles.py        # QSS puro. Sin lógica.
app/yolo_exporter.py # I/O puro. Sin PyQt6.
main.py              # QApplication + arranque.
```

## Convenciones de código

- Nombres en español para variables de dominio (clase, etiqueta, anotacion)
- Nombres en inglés para infraestructura PyQt (item, widget, layout, scene)
- Sin comentarios obvios; solo comentarios donde el WHY no es evidente
- Type hints en todas las firmas de funciones públicas
- Señales PyQt6: `snake_case`, sufijo descriptivo (`box_created`, `box_deleted`)

## Cómo ejecutar

```bash
pip install -r requirements.txt
python main.py
```

## Tareas pendientes para próximas fases

### Fase 2 — Multi-imagen
- [ ] Panel de galería (lista de imágenes en carpeta)
- [ ] Navegación anterior/siguiente con atajos de teclado
- [ ] Persistencia de anotaciones en JSON por proyecto
- [ ] Importar etiquetas YOLO existentes (.txt)

### Fase 3 — UX avanzado
- [ ] Redimensionar bboxes con handles en las esquinas
- [ ] Copiar/pegar bboxes entre imágenes
- [ ] Undo/Redo (QUndoStack)
- [ ] Miniatura de navegación (overview)
- [ ] Zoom a resolución 1:1

### Fase 4 — Integración YOLO
- [ ] Exportar dataset completo (images/ + labels/)
- [ ] Generar data.yaml para YOLOv8
- [ ] Vista previa de inferencia con modelo entrenado
- [ ] Estadísticas de dataset (distribución de clases)

## Puntos de extensión conocidos

- `AnnotationCanvas.add_bbox_item()` — punto de entrada para cargar anotaciones externas
- `YoloExporter` — añadir métodos para otros formatos (COCO JSON, Pascal VOC XML)
- `MainWindow._build_left_panel()` — área para más herramientas (polígono, punto, etc.)
- `CLASS_COLORS` en `styles.py` — ampliar paleta

## Patrones a seguir al añadir código

1. Modelos de datos en `annotation.py` — sin dependencias PyQt6
2. Lógica visual en `canvas.py` — emitir señales en lugar de llamar directamente a `main_window`
3. Conectar señales en `MainWindow._connect_signals()`
4. Estado de botones en `MainWindow._update_ui_state()`
5. QSS en `styles.py` — no inline en widgets

## Comandos útiles para Claude Code

```bash
# Instalar dependencias
pip install -r requirements.txt

# Ejecutar la aplicación
python main.py

# Verificar imports sin abrir ventana
python -c "from app.main_window import MainWindow; print('OK')"
```
