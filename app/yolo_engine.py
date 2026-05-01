import cv2
import numpy as np
from PyQt6.QtWidgets import (
    QWidget, QVBoxLayout, QHBoxLayout, QPushButton, QLabel, 
    QTextEdit, QProgressBar, QFileDialog, QFormLayout, QSpinBox
)
from PyQt6.QtCore import Qt, QThread, pyqtSignal
from PyQt6.QtGui import QImage, QPixmap

try:
    from ultralytics import YOLO
    HAS_YOLO = True
except ImportError:
    HAS_YOLO = False


class YoloTrainerWorker(QThread):
    """Ejecuta el entrenamiento de YOLO en un hilo secundario."""
    log_msg = pyqtSignal(str)
    finished_training = pyqtSignal(bool, str)

    def __init__(self, dataset_yaml: str, epochs: int, imgsz: int, parent=None):
        super().__init__(parent)
        self.dataset_yaml = dataset_yaml
        self.epochs = epochs
        self.imgsz = imgsz
        self._is_running = True

    def run(self):
        if not HAS_YOLO:
            self.log_msg.emit("Error: librería ultralytics no encontrada.")
            self.finished_training.emit(False, "Error dependencias")
            return

        try:
            self.log_msg.emit("Cargando modelo yolov8n.pt...")
            model = YOLO('yolov8n.pt')
            self.log_msg.emit(f"Iniciando entrenamiento ({self.epochs} epochs)...")
            
            # TODO: Redirigir stdout/stderr de YOLO es complejo,
            # pero YOLO imprimirá por consola. 
            # Aquí llamamos al entrenamiento.
            results = model.train(
                data=self.dataset_yaml,
                epochs=self.epochs,
                imgsz=self.imgsz,
                device='', # auto
                verbose=True
            )
            
            self.log_msg.emit("¡Entrenamiento finalizado con éxito!")
            self.finished_training.emit(True, "OK")
        except Exception as e:
            self.log_msg.emit(f"Error durante el entrenamiento: {str(e)}")
            self.finished_training.emit(False, str(e))

    def stop(self):
        self._is_running = False
        self.quit()
        self.wait()


class TrainWidget(QWidget):
    """Panel de configuración y ejecución de entrenamiento YOLO."""

    def __init__(self, parent=None):
        super().__init__(parent)
        self.worker = None
        self._build_ui()

    def _build_ui(self):
        layout = QVBoxLayout(self)
        layout.setContentsMargins(20, 20, 20, 20)
        layout.setSpacing(15)

        title = QLabel("Entrenamiento YOLO")
        title.setStyleSheet("color: #cdd6f4; font-size: 24px; font-weight: bold;")
        layout.addWidget(title)

        # Formulario de configuración
        form = QFormLayout()
        
        self.yaml_label = QLabel("No seleccionado")
        self.yaml_label.setStyleSheet("color: #a6e3a1;")
        btn_yaml = QPushButton("Seleccionar dataset.yaml")
        btn_yaml.clicked.connect(self._select_yaml)
        
        yaml_layout = QHBoxLayout()
        yaml_layout.addWidget(btn_yaml)
        yaml_layout.addWidget(self.yaml_label)
        form.addRow("Dataset YAML:", yaml_layout)

        self.spin_epochs = QSpinBox()
        self.spin_epochs.setRange(1, 1000)
        self.spin_epochs.setValue(10)
        form.addRow("Epochs:", self.spin_epochs)

        self.spin_imgsz = QSpinBox()
        self.spin_imgsz.setRange(320, 1280)
        self.spin_imgsz.setSingleStep(32)
        self.spin_imgsz.setValue(640)
        form.addRow("Image Size:", self.spin_imgsz)

        layout.addLayout(form)

        # Controles
        self.btn_train = QPushButton("▶ Iniciar Entrenamiento")
        self.btn_train.setObjectName("btn_primary")
        self.btn_train.clicked.connect(self._toggle_training)
        layout.addWidget(self.btn_train)

        # Logs
        layout.addWidget(QLabel("Consola:"))
        self.log_console = QTextEdit()
        self.log_console.setReadOnly(True)
        self.log_console.setStyleSheet("background: #11111b; color: #a6adc8; font-family: monospace;")
        layout.addWidget(self.log_console)

    def _select_yaml(self):
        path, _ = QFileDialog.getOpenFileName(self, "Seleccionar YAML", "", "YAML (*.yaml *.yml)")
        if path:
            self.yaml_label.setText(path)

    def _toggle_training(self):
        if self.worker is not None and self.worker.isRunning():
            self.worker.stop()
            self.worker = None
            self.btn_train.setText("▶ Iniciar Entrenamiento")
            self.log_console.append("Entrenamiento detenido por el usuario.")
            return

        yaml_path = self.yaml_label.text()
        if yaml_path == "No seleccionado":
            self.log_console.append("⚠ Error: Selecciona un archivo dataset.yaml primero.")
            return

        self.btn_train.setText("⏹ Detener Entrenamiento")
        self.log_console.clear()
        self.worker = YoloTrainerWorker(
            dataset_yaml=yaml_path,
            epochs=self.spin_epochs.value(),
            imgsz=self.spin_imgsz.value()
        )
        self.worker.log_msg.connect(self.log_console.append)
        self.worker.finished_training.connect(self._on_training_finished)
        self.worker.start()

    def _on_training_finished(self, success: bool, msg: str):
        self.btn_train.setText("▶ Iniciar Entrenamiento")
        self.worker = None


class YoloInferenceWorker(QThread):
    """Hilo de inferencia en tiempo real para capturar la cámara y predecir."""
    frame_ready = pyqtSignal(QImage)
    error_msg = pyqtSignal(str)

    def __init__(self, model_path: str, parent=None):
        super().__init__(parent)
        self.model_path = model_path
        self._is_running = True

    def run(self):
        if not HAS_YOLO:
            self.error_msg.emit("Error: librería ultralytics no encontrada.")
            return

        try:
            model = YOLO(self.model_path)
            cap = cv2.VideoCapture(0)  # Webcam por defecto
            
            if not cap.isOpened():
                self.error_msg.emit("Error: No se pudo acceder a la cámara.")
                return

            while self._is_running:
                ret, frame = cap.read()
                if not ret:
                    continue

                # Inferencia
                results = model(frame, verbose=False)
                
                # Dibujar resultados (anotación nativa de ultralytics)
                annotated_frame = results[0].plot()
                
                # Convertir BGR (OpenCV) a RGB (Qt)
                rgb_image = cv2.cvtColor(annotated_frame, cv2.COLOR_BGR2RGB)
                h, w, ch = rgb_image.shape
                bytes_per_line = ch * w
                qt_img = QImage(rgb_image.data, w, h, bytes_per_line, QImage.Format.Format_RGB888)
                
                self.frame_ready.emit(qt_img)
                
            cap.release()
        except Exception as e:
            self.error_msg.emit(f"Error de inferencia: {str(e)}")

    def stop(self):
        self._is_running = False
        self.quit()
        self.wait()


class InferenceWidget(QWidget):
    """Panel de Demostración de Inferencia en Tiempo Real."""

    def __init__(self, parent=None):
        super().__init__(parent)
        self.worker = None
        self._build_ui()

    def _build_ui(self):
        layout = QVBoxLayout(self)
        layout.setContentsMargins(20, 20, 20, 20)
        layout.setSpacing(15)

        title = QLabel("Demo de Inferencia en Tiempo Real")
        title.setStyleSheet("color: #cdd6f4; font-size: 24px; font-weight: bold;")
        layout.addWidget(title)

        # Controles
        ctrl_layout = QHBoxLayout()
        self.btn_load_model = QPushButton("Cargar Modelo (.pt)")
        self.btn_load_model.clicked.connect(self._load_model)
        ctrl_layout.addWidget(self.btn_load_model)

        self.lbl_model = QLabel("Modelo: yolov8n.pt (por defecto)")
        self.lbl_model.setStyleSheet("color: #a6e3a1;")
        ctrl_layout.addWidget(self.lbl_model)
        
        self.btn_start = QPushButton("▶ Iniciar Cámara")
        self.btn_start.setObjectName("btn_primary")
        self.btn_start.clicked.connect(self._toggle_inference)
        ctrl_layout.addWidget(self.btn_start)
        
        layout.addLayout(ctrl_layout)

        # Visor de cámara
        self.lbl_camera = QLabel("Cámara apagada")
        self.lbl_camera.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.lbl_camera.setStyleSheet("background: #11111b; border: 2px solid #45475a; border-radius: 8px;")
        self.lbl_camera.setMinimumSize(640, 480)
        layout.addWidget(self.lbl_camera, stretch=1)
        
        self.model_path = "yolov8n.pt"

    def _load_model(self):
        path, _ = QFileDialog.getOpenFileName(self, "Seleccionar Modelo", "", "PyTorch Models (*.pt)")
        if path:
            self.model_path = path
            self.lbl_model.setText(f"Modelo: {path}")

    def _toggle_inference(self):
        if self.worker is not None and self.worker.isRunning():
            self.worker.stop()
            self.worker = None
            self.btn_start.setText("▶ Iniciar Cámara")
            self.lbl_camera.setText("Cámara apagada")
            return

        self.btn_start.setText("⏹ Detener Cámara")
        self.worker = YoloInferenceWorker(model_path=self.model_path)
        self.worker.frame_ready.connect(self._update_frame)
        self.worker.error_msg.connect(self.lbl_camera.setText)
        self.worker.start()

    def _update_frame(self, image: QImage):
        pixmap = QPixmap.fromImage(image)
        # Escalar manteniendo la proporción
        scaled_pixmap = pixmap.scaled(
            self.lbl_camera.size(), 
            Qt.AspectRatioMode.KeepAspectRatio, 
            Qt.TransformationMode.SmoothTransformation
        )
        self.lbl_camera.setPixmap(scaled_pixmap)
