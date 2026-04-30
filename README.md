# VisionHub Desktop — MVP Etiquetado

Herramienta de escritorio para etiquetar imágenes con bounding boxes y exportar en formato YOLO.

## Requisitos

- Python 3.11+
- PyQt6

## Instalación

```bash
pip install -r requirements.txt
```

## Ejecución

```bash
python main.py
```

## Flujo de trabajo

1. **Cargar imagen** — clic en "Cargar imagen" o `Ctrl+O`
2. **Crear clase** — escribe el nombre en el panel derecho y pulsa `+`
3. **Seleccionar clase** activa en la lista
4. **Activar modo dibujo** — botón "Dibujar bbox" o tecla `D`
5. **Dibujar** — clic y arrastra sobre la imagen
6. **Eliminar** bbox — selecciónalo y pulsa `Del` o el botón "Eliminar"
7. **Exportar** — clic en "Exportar YOLO" genera `<imagen>.txt` y `classes.txt`

## Controles

| Acción | Atajo |
|--------|-------|
| Modo dibujo on/off | `D` |
| Ajustar vista | `F` |
| Eliminar seleccionado | `Del` |
| Zoom | Rueda del ratón |

## Formato de salida YOLO

```
<class_id> <x_center> <y_center> <width> <height>
```
Coordenadas normalizadas entre 0 y 1.

## Estructura del proyecto

```
visionhub/
├── main.py               # Punto de entrada
├── requirements.txt
├── README.md
├── app/
│   ├── __init__.py
│   ├── main_window.py    # Ventana principal y lógica UI
│   ├── canvas.py         # QGraphicsView con dibujo de bboxes
│   ├── annotation.py     # Modelos de datos (BoundingBox, LabelClass)
│   ├── styles.py         # Tema oscuro QSS + paleta de colores
│   └── yolo_exporter.py  # Exportación YOLO
└── assets/               # Iconos y recursos (futuro)
```
