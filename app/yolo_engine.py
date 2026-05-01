from __future__ import annotations

import cv2
import uuid
from datetime import datetime, timezone
from pathlib import Path

from PyQt6.QtCore import QThread, Qt, pyqtSignal
from PyQt6.QtGui import QImage, QPainter, QPen, QPixmap
from PyQt6.QtWidgets import (
    QFileDialog,
    QFormLayout,
    QHBoxLayout,
    QLabel,
    QListWidget,
    QPushButton,
    QProgressBar,
    QSpinBox,
    QTextEdit,
    QVBoxLayout,
    QWidget,
)

try:
    from ultralytics import YOLO
    HAS_YOLO = True
except ImportError:
    HAS_YOLO = False


class MetricsPlotWidget(QWidget):
    """Gráfico simple en memoria para métricas de entrenamiento."""

    def __init__(self, parent=None):
        super().__init__(parent)
        self.setMinimumHeight(180)
        self._series: dict[str, list[float]] = {
            "map50": [],
            "box_loss": [],
        }

    def clear(self) -> None:
        for k in self._series:
            self._series[k].clear()
        self.update()

    def append(self, map50: float | None, box_loss: float | None) -> None:
        self._series["map50"].append(float(map50) if map50 is not None else 0.0)
        self._series["box_loss"].append(float(box_loss) if box_loss is not None else 0.0)
        self.update()

    def paintEvent(self, event):
        del event
        p = QPainter(self)
        p.setRenderHint(QPainter.RenderHint.Antialiasing, True)
        rect = self.rect().adjusted(10, 10, -10, -10)
        p.fillRect(rect, Qt.GlobalColor.black)

        p.setPen(QPen(Qt.GlobalColor.darkGray, 1))
        p.drawRect(rect)

        self._draw_series(p, rect, self._series["map50"], QPen(Qt.GlobalColor.green, 2), 1.0)

        loss = self._series["box_loss"]
        loss_max = max(loss) if loss else 1.0
        self._draw_series(p, rect, loss, QPen(Qt.GlobalColor.yellow, 2), loss_max if loss_max > 0 else 1.0)

    def _draw_series(self, painter: QPainter, rect, values: list[float], pen: QPen, max_value: float):
        if len(values) < 2:
            return
        painter.setPen(pen)
        n = len(values)
        for i in range(1, n):
            x1 = rect.left() + (i - 1) * rect.width() / (n - 1)
            x2 = rect.left() + i * rect.width() / (n - 1)
            y1 = rect.bottom() - (values[i - 1] / max_value) * rect.height()
            y2 = rect.bottom() - (values[i] / max_value) * rect.height()
            painter.drawLine(int(x1), int(y1), int(x2), int(y2))


class YoloTrainerWorker(QThread):
    """Ejecuta entrenamiento YOLO con callbacks de métricas en vivo."""

    log_msg = pyqtSignal(str)
    epoch_metrics = pyqtSignal(dict)
    finished_training = pyqtSignal(bool, str, dict)

    def __init__(
        self,
        *,
        dataset_yaml: str,
        epochs: int,
        imgsz: int,
        batch: int,
        patience: int,
        model_name: str,
        run_project_dir: str,
        run_name: str,
        resume: bool = False,
        resume_checkpoint: str = "",
        parent=None,
    ):
        super().__init__(parent)
        self.dataset_yaml = dataset_yaml
        self.epochs = epochs
        self.imgsz = imgsz
        self.batch = batch
        self.patience = patience
        self.model_name = model_name
        self.run_project_dir = run_project_dir
        self.run_name = run_name
        self.resume = resume
        self.resume_checkpoint = resume_checkpoint
        self._stop_requested = False

    def _as_float(self, d: dict, *keys: str) -> float | None:
        for k in keys:
            if k in d:
                try:
                    return float(d[k])
                except Exception:
                    return None
        return None

    def run(self):
        if not HAS_YOLO:
            self.finished_training.emit(False, "Librería ultralytics no encontrada.", {})
            return

        run_dir = str(Path(self.run_project_dir) / self.run_name)
        info = {"run_dir": run_dir, "weights_path": "", "metrics_path": ""}
        try:
            model = YOLO(self.resume_checkpoint if (self.resume and self.resume_checkpoint) else self.model_name)

            def on_fit_epoch_end(trainer):
                metrics = dict(getattr(trainer, "metrics", {}) or {})
                payload = {
                    "epoch": int(getattr(trainer, "epoch", -1)) + 1,
                    "map50": self._as_float(metrics, "metrics/mAP50(B)", "metrics/mAP50"),
                    "map50_95": self._as_float(metrics, "metrics/mAP50-95(B)", "metrics/mAP50-95"),
                    "precision": self._as_float(metrics, "metrics/precision(B)", "metrics/precision"),
                    "recall": self._as_float(metrics, "metrics/recall(B)", "metrics/recall"),
                    "box_loss": self._as_float(metrics, "train/box_loss", "val/box_loss"),
                }
                self.epoch_metrics.emit(payload)
                if self._stop_requested:
                    setattr(trainer, "stop", True)

            model.add_callback("on_fit_epoch_end", on_fit_epoch_end)

            self.log_msg.emit(f"Iniciando entrenamiento en {run_dir}")
            train_kwargs = {
                "data": self.dataset_yaml,
                "epochs": self.epochs,
                "imgsz": self.imgsz,
                "batch": self.batch,
                "patience": self.patience,
                "project": self.run_project_dir,
                "name": self.run_name,
                "plots": True,
                "save_period": 1,
                "verbose": True,
            }
            if self.resume:
                train_kwargs["resume"] = True

            results = model.train(**train_kwargs)
            save_dir = Path(getattr(results, "save_dir", run_dir))
            info["run_dir"] = str(save_dir)

            best_pt = save_dir / "weights" / "best.pt"
            last_pt = save_dir / "weights" / "last.pt"
            info["weights_path"] = str(best_pt if best_pt.exists() else last_pt)

            metrics_csv = save_dir / "results.csv"
            if metrics_csv.exists():
                info["metrics_path"] = str(metrics_csv)

            if self._stop_requested:
                self.finished_training.emit(False, "Entrenamiento detenido por el usuario.", info)
            else:
                self.finished_training.emit(True, "Entrenamiento finalizado.", info)
        except Exception as e:
            self.finished_training.emit(False, str(e), info)

    def stop(self):
        self._stop_requested = True
        self.log_msg.emit("Solicitud de parada enviada. Se detendrá al finalizar la época actual.")


class TrainWidget(QWidget):
    """Panel de entrenamiento YOLO con métricas en vivo y persistencia."""

    dataset_export_requested = pyqtSignal()

    def __init__(self, *, db=None, project_id: str = "animales", parent=None):
        super().__init__(parent)
        self._db = db
        self._project_id = project_id
        self.worker: YoloTrainerWorker | None = None
        self._current_training_id: str | None = None
        self._last_run_info: dict = {}
        self._build_ui()

    def _build_ui(self):
        layout = QVBoxLayout(self)
        layout.setContentsMargins(20, 20, 20, 20)
        layout.setSpacing(12)

        title = QLabel("Entrenamiento YOLO")
        title.setStyleSheet("color: #cdd6f4; font-size: 24px; font-weight: bold;")
        layout.addWidget(title)

        form = QFormLayout()

        self.yaml_label = QLabel("No seleccionado")
        self.yaml_label.setStyleSheet("color: #a6e3a1;")
        self.yaml_label.setWordWrap(True)

        btn_yaml = QPushButton("Seleccionar dataset.yaml")
        btn_yaml.clicked.connect(self._select_yaml)

        self.btn_export_dataset = QPushButton("Generar Dataset")
        self.btn_export_dataset.setObjectName("btn_primary")
        self.btn_export_dataset.clicked.connect(self.dataset_export_requested)

        yaml_layout = QHBoxLayout()
        yaml_layout.addWidget(self.btn_export_dataset)
        yaml_layout.addWidget(btn_yaml)
        yaml_layout.addWidget(self.yaml_label, stretch=1)
        form.addRow("Dataset YAML:", yaml_layout)

        self.spin_epochs = QSpinBox()
        self.spin_epochs.setRange(1, 1000)
        self.spin_epochs.setValue(50)
        form.addRow("Epochs:", self.spin_epochs)

        self.spin_imgsz = QSpinBox()
        self.spin_imgsz.setRange(320, 1280)
        self.spin_imgsz.setSingleStep(32)
        self.spin_imgsz.setValue(640)
        form.addRow("Image Size:", self.spin_imgsz)

        self.spin_batch = QSpinBox()
        self.spin_batch.setRange(1, 256)
        self.spin_batch.setValue(16)
        form.addRow("Batch size:", self.spin_batch)

        self.spin_patience = QSpinBox()
        self.spin_patience.setRange(0, 300)
        self.spin_patience.setValue(50)
        form.addRow("Patience:", self.spin_patience)

        layout.addLayout(form)

        self.lbl_kpi = QLabel("Epoch: - | mAP50: - | Loss: -")
        self.lbl_kpi.setStyleSheet("color: #cdd6f4; font-size: 12px;")
        layout.addWidget(self.lbl_kpi)

        self.progress_epoch = QProgressBar()
        self.progress_epoch.setMinimum(0)
        self.progress_epoch.setValue(0)
        self.progress_epoch.setFormat("Época 0/0")
        layout.addWidget(self.progress_epoch)

        self.plot = MetricsPlotWidget()
        layout.addWidget(self.plot)

        layout.addWidget(QLabel("Historial de épocas:"))
        self.epoch_list = QListWidget()
        self.epoch_list.setStyleSheet(
            "background: #11111b; color: #cdd6f4; border: 1px solid #313244;"
        )
        self.epoch_list.setMinimumHeight(120)
        layout.addWidget(self.epoch_list)

        self.btn_train = QPushButton("▶ Iniciar Entrenamiento")
        self.btn_train.setObjectName("btn_primary")
        self.btn_train.clicked.connect(self._toggle_training)
        layout.addWidget(self.btn_train)

        layout.addWidget(QLabel("Consola:"))
        self.log_console = QTextEdit()
        self.log_console.setReadOnly(True)
        self.log_console.setStyleSheet(
            "background: #11111b; color: #a6adc8; font-family: monospace;"
        )
        layout.addWidget(self.log_console)

    def set_yaml(self, path: str) -> None:
        self.yaml_label.setText(path)

    def _select_yaml(self):
        path, _ = QFileDialog.getOpenFileName(self, "Seleccionar YAML", "", "YAML (*.yaml *.yml)")
        if path:
            self.yaml_label.setText(path)

    def _toggle_training(self):
        if self.worker is not None and self.worker.isRunning():
            self.btn_train.setEnabled(False)
            self.worker.stop()
            return

        yaml_path = self.yaml_label.text()
        if yaml_path == "No seleccionado":
            self.log_console.append("Selecciona un dataset.yaml o genera el dataset primero.")
            return

        self.plot.clear()
        self.log_console.clear()
        self.epoch_list.clear()
        self.progress_epoch.setMaximum(self.spin_epochs.value())
        self.progress_epoch.setValue(0)
        self.progress_epoch.setFormat(f"Época 0/{self.spin_epochs.value()}")
        self.btn_train.setText("⏹ Detener Entrenamiento")

        now = datetime.now(timezone.utc)
        self._current_training_id = str(uuid.uuid4())
        run_name = f"train_{now.strftime('%Y%m%d_%H%M%S')}_{self._project_id}"
        run_project_dir = str(Path(yaml_path).resolve().parent / "runs")
        self._last_run_info = {"run_dir": str(Path(run_project_dir) / run_name)}

        if self._db is not None:
            try:
                self._db.create_training_run(
                    training_id=self._current_training_id,
                    project_id=self._project_id,
                    model_name="yolov8n.pt",
                    dataset_yaml_path=yaml_path,
                    epochs=self.spin_epochs.value(),
                    image_size=self.spin_imgsz.value(),
                    batch_size=self.spin_batch.value(),
                    status="running",
                )
            except Exception as e:
                self.log_console.append(f"No se pudo registrar training en BD: {e}")

        self.worker = YoloTrainerWorker(
            dataset_yaml=yaml_path,
            epochs=self.spin_epochs.value(),
            imgsz=self.spin_imgsz.value(),
            batch=self.spin_batch.value(),
            patience=self.spin_patience.value(),
            model_name="yolov8n.pt",
            run_project_dir=run_project_dir,
            run_name=run_name,
        )
        self.worker.log_msg.connect(self.log_console.append)
        self.worker.epoch_metrics.connect(self._on_epoch_metrics)
        self.worker.finished_training.connect(self._on_training_finished)
        self.worker.finished.connect(self._on_worker_done)
        self.worker.start()

    def _on_epoch_metrics(self, data: dict):
        epoch = data.get("epoch", "-")
        map50 = data.get("map50")
        box_loss = data.get("box_loss")
        self.plot.append(map50, box_loss)
        try:
            current_epoch = int(epoch)
            total = self.spin_epochs.value()
            self.progress_epoch.setMaximum(total)
            self.progress_epoch.setValue(min(current_epoch, total))
            self.progress_epoch.setFormat(f"Época {current_epoch}/{total}")
        except Exception:
            pass

        epoch_line = (
            f"Epoch {epoch} | "
            f"mAP50={map50 if map50 is not None else '-'} | "
            f"loss={box_loss if box_loss is not None else '-'}"
        )
        self.epoch_list.addItem(epoch_line)
        self.epoch_list.scrollToBottom()
        self.lbl_kpi.setText(
            f"Epoch: {epoch} | "
            f"mAP50: {map50 if map50 is not None else '-'} | "
            f"Loss: {box_loss if box_loss is not None else '-'}"
        )

    def _on_training_finished(self, success: bool, msg: str, info: dict):
        self._last_run_info = info or self._last_run_info
        self.log_console.append(msg)

        if self._db is not None and self._current_training_id is not None:
            try:
                status = "finished" if success else "failed"
                if "detenido" in msg.lower():
                    status = "stopped"
                self._db.update_training_run(
                    training_id=self._current_training_id,
                    status=status,
                    metrics_path=self._last_run_info.get("metrics_path", ""),
                    weights_path=self._last_run_info.get("weights_path", ""),
                )
            except Exception as e:
                self.log_console.append(f"No se pudo actualizar training en BD: {e}")

        if self._last_run_info.get("weights_path"):
            self.log_console.append(f"Pesos: {self._last_run_info['weights_path']}")
        if self._last_run_info.get("metrics_path"):
            self.log_console.append(f"Métricas: {self._last_run_info['metrics_path']}")
        if self._last_run_info.get("run_dir"):
            self.log_console.append(f"Run dir: {self._last_run_info['run_dir']}")

    def _on_worker_done(self):
        self.worker = None
        self.btn_train.setEnabled(True)
        self.btn_train.setText("▶ Iniciar Entrenamiento")


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
            cap = cv2.VideoCapture(0)

            if not cap.isOpened():
                self.error_msg.emit("Error: No se pudo acceder a la cámara.")
                return

            while self._is_running:
                ret, frame = cap.read()
                if not ret:
                    continue

                results = model(frame, verbose=False)
                annotated_frame = results[0].plot()

                rgb_image = cv2.cvtColor(annotated_frame, cv2.COLOR_BGR2RGB)
                h, w, ch = rgb_image.shape
                qt_img = QImage(rgb_image.data, w, h, ch * w, QImage.Format.Format_RGB888)
                self.frame_ready.emit(qt_img)

            cap.release()
        except Exception as e:
            self.error_msg.emit(f"Error de inferencia: {str(e)}")

    def stop(self):
        self._is_running = False
        self.quit()
        self.wait()


class InferenceWidget(QWidget):
    """Panel de demostración de inferencia en tiempo real."""

    def __init__(self, parent=None):
        super().__init__(parent)
        self.worker: YoloInferenceWorker | None = None
        self._build_ui()

    def _build_ui(self):
        layout = QVBoxLayout(self)
        layout.setContentsMargins(20, 20, 20, 20)
        layout.setSpacing(15)

        title = QLabel("Demo de Inferencia en Tiempo Real")
        title.setStyleSheet("color: #cdd6f4; font-size: 24px; font-weight: bold;")
        layout.addWidget(title)

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

        self.lbl_camera = QLabel("Cámara apagada")
        self.lbl_camera.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.lbl_camera.setStyleSheet(
            "background: #11111b; border: 2px solid #45475a; border-radius: 8px;"
        )
        self.lbl_camera.setMinimumSize(640, 480)
        layout.addWidget(self.lbl_camera, stretch=1)

        self.model_path = "yolov8n.pt"

    def stop_camera(self) -> None:
        if self.worker is not None and self.worker.isRunning():
            self.worker.stop()
            self.worker = None
            self.btn_start.setText("▶ Iniciar Cámara")
            self.lbl_camera.setText("Cámara apagada")

    def hideEvent(self, event):
        self.stop_camera()
        super().hideEvent(event)

    def _load_model(self):
        path, _ = QFileDialog.getOpenFileName(self, "Seleccionar Modelo", "", "PyTorch Models (*.pt)")
        if path:
            self.model_path = path
            self.lbl_model.setText(f"Modelo: {path}")

    def _toggle_inference(self):
        if self.worker is not None and self.worker.isRunning():
            self.stop_camera()
            return

        self.btn_start.setText("⏹ Detener Cámara")
        self.worker = YoloInferenceWorker(model_path=self.model_path)
        self.worker.frame_ready.connect(self._update_frame)
        self.worker.error_msg.connect(self.lbl_camera.setText)
        self.worker.start()

    def _update_frame(self, image: QImage):
        pixmap = QPixmap.fromImage(image)
        scaled_pixmap = pixmap.scaled(
            self.lbl_camera.size(),
            Qt.AspectRatioMode.KeepAspectRatio,
            Qt.TransformationMode.SmoothTransformation,
        )
        self.lbl_camera.setPixmap(scaled_pixmap)
