from __future__ import annotations

import cv2
import csv
import json
import platform
import re
import shutil
import subprocess
import uuid
from datetime import datetime, timezone
from pathlib import Path

from PyQt6.QtCore import QThread, Qt, pyqtSignal
from PyQt6.QtGui import QImage, QPainter, QPen, QPixmap
from PyQt6.QtWidgets import (
    QCheckBox,
    QComboBox,
    QDoubleSpinBox,
    QFileDialog,
    QFormLayout,
    QFrame,
    QGroupBox,
    QHeaderView,
    QHBoxLayout,
    QLabel,
    QListWidget,
    QListWidgetItem,
    QPushButton,
    QProgressBar,
    QMessageBox,
    QTabWidget,
    QTableWidget,
    QTableWidgetItem,
    QSpinBox,
    QStackedWidget,
    QTextEdit,
    QVBoxLayout,
    QWidget,
)

try:
    from ultralytics import YOLO
    HAS_YOLO = True
except ImportError:
    HAS_YOLO = False

try:
    import torch
    HAS_TORCH = True
except ImportError:
    torch = None
    HAS_TORCH = False


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
        learning_rate: float = 0.01,
        device: str = "",
        augment: bool = True,
        augmentation_params: dict | None = None,
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
        self.learning_rate = learning_rate
        self.device = device
        self.augment = augment
        self.augmentation_params = augmentation_params or {}
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
                "lr0": self.learning_rate,
                "augment": self.augment,
            }
            if self.device:
                train_kwargs["device"] = self.device
            train_kwargs.update(self.augmentation_params)
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
    """Panel profesional de entrenamiento con versiones, configuracion y resultados."""

    dataset_export_requested = pyqtSignal(str)

    def __init__(self, *, db=None, project_id: str = "animales", project_base_path: str = "", parent=None):
        super().__init__(parent)
        self._db = db
        self._project_id = project_id
        self._project_base_path = project_base_path
        self.worker: YoloTrainerWorker | None = None
        self._current_training_id: str | None = None
        self._selected_training: dict | None = None
        self._last_run_info: dict = {}
        self._build_ui()
        self._load_training_runs()

    def _build_ui(self):
        root = QHBoxLayout(self)
        root.setContentsMargins(0, 0, 0, 0)
        root.setSpacing(0)

        self.sidebar = QFrame()
        self.sidebar.setFixedWidth(285)
        self.sidebar.setStyleSheet("QFrame { background: #181825; border-right: 1px solid #313244; }")
        side = QVBoxLayout(self.sidebar)
        side.setContentsMargins(14, 16, 14, 14)
        side.setSpacing(10)

        title = QLabel("Model Trainings")
        title.setStyleSheet("color: #cdd6f4; font-size: 20px; font-weight: bold;")
        side.addWidget(title)

        self.btn_new_run = QPushButton("Crear version")
        self.btn_new_run.setObjectName("btn_primary")
        self.btn_new_run.clicked.connect(self._new_training_version)
        side.addWidget(self.btn_new_run)

        self.training_list = QListWidget()
        self.training_list.setStyleSheet(
            "QListWidget { background: #11111b; color: #cdd6f4; border: 1px solid #313244; }"
            "QListWidget::item { padding: 8px; border-bottom: 1px solid #1e1e2e; }"
            "QListWidget::item:selected { background: #0f766e; color: white; }"
        )
        self.training_list.currentItemChanged.connect(self._on_training_selected)
        side.addWidget(self.training_list, stretch=1)

        self.btn_resume_run = QPushButton("Reentrenar seleccionado")
        self.btn_resume_run.clicked.connect(lambda: self._toggle_training(resume_selected=True))
        side.addWidget(self.btn_resume_run)

        self.btn_delete_run = QPushButton("Eliminar seleccionado")
        self.btn_delete_run.clicked.connect(self._delete_selected_training)
        side.addWidget(self.btn_delete_run)

        root.addWidget(self.sidebar)

        main = QVBoxLayout()
        main.setContentsMargins(18, 16, 18, 16)
        main.setSpacing(12)
        root.addLayout(main, stretch=1)

        header = QHBoxLayout()
        self.lbl_title = QLabel("Entrenamiento YOLO")
        self.lbl_title.setStyleSheet("color: #cdd6f4; font-size: 24px; font-weight: bold;")
        header.addWidget(self.lbl_title)
        header.addStretch()
        self.lbl_status = QLabel("Listo")
        self.lbl_status.setStyleSheet("color: #a6e3a1; font-size: 12px;")
        header.addWidget(self.lbl_status)
        main.addLayout(header)

        self.tabs = QTabWidget()
        self.tabs.setStyleSheet(
            "QTabWidget::pane { border: 1px solid #313244; background: #1e1e2e; }"
            "QTabBar::tab { background: #313244; color: #cdd6f4; padding: 10px 22px; }"
            "QTabBar::tab:selected { background: #0f9f8f; color: white; }"
        )
        self.tabs.addTab(self._build_settings_tab(), "Configuracion")
        self.tabs.addTab(self._build_results_tab(), "Resultados")
        main.addWidget(self.tabs, stretch=1)

    def _panel(self, title: str) -> QGroupBox:
        box = QGroupBox(title)
        box.setStyleSheet(
            "QGroupBox { color: #cdd6f4; font-weight: bold; border: 1px solid #313244; "
            "border-radius: 8px; margin-top: 10px; padding-top: 12px; background: #1e1e2e; }"
            "QGroupBox::title { subcontrol-origin: margin; left: 10px; padding: 0 4px; }"
            "QLabel { color: #cdd6f4; }"
        )
        return box

    def _build_settings_tab(self) -> QWidget:
        page = QWidget()
        outer = QHBoxLayout(page)
        outer.setContentsMargins(12, 12, 12, 12)
        outer.setSpacing(14)

        left = QVBoxLayout()
        right = QVBoxLayout()
        outer.addLayout(left, stretch=1)
        outer.addLayout(right, stretch=1)

        dataset_box = self._panel("Dataset")
        dataset_layout = QVBoxLayout(dataset_box)
        self.btn_select_yaml = QPushButton("Seleccionar data.yaml")
        self.btn_select_yaml.clicked.connect(self._select_yaml)
        dataset_layout.addWidget(self.btn_select_yaml)
        self.yaml_label = QLabel("No seleccionado")
        self.yaml_label.setWordWrap(True)
        self.yaml_label.setStyleSheet("color: #a6e3a1;")
        dataset_layout.addWidget(self.yaml_label)
        left.addWidget(dataset_box)

        network_box = self._panel("Red neuronal")
        network_form = QFormLayout(network_box)
        self.combo_model = QComboBox()
        self.combo_model.addItems(["yolov8n.pt", "yolov8s.pt", "yolov8m.pt", "yolo11n.pt", "yolo11s.pt"])
        network_form.addRow("Modelo base:", self.combo_model)
        self.spin_imgsz = QSpinBox()
        self.spin_imgsz.setRange(320, 1536)
        self.spin_imgsz.setSingleStep(32)
        self.spin_imgsz.setValue(640)
        network_form.addRow("Tamano imagen:", self.spin_imgsz)
        self.combo_device = QComboBox()
        self._populate_device_options()
        network_form.addRow("Dispositivo:", self.combo_device)

        config_box = self._panel("Configuracion avanzada")
        config_form = QFormLayout(config_box)
        self.chk_deterministic = QCheckBox("Use deterministic algorithms")
        self.chk_deterministic.setStyleSheet("color: #cdd6f4;")
        config_form.addRow(self.chk_deterministic)
        self.chk_random_seed = QCheckBox("Random Seed")
        self.chk_random_seed.setStyleSheet("color: #cdd6f4;")
        self.spin_seed = QSpinBox()
        self.spin_seed.setRange(0, 999999)
        self.spin_seed.setValue(42)
        self.spin_seed.setEnabled(False)
        self.chk_random_seed.toggled.connect(self.spin_seed.setEnabled)
        config_form.addRow(self.chk_random_seed, self.spin_seed)
        left.addWidget(network_box)
        left.addWidget(config_box)

        train_box = self._panel("Parametros de entrenamiento")
        train_form = QFormLayout(train_box)
        self.spin_epochs = QSpinBox()
        self.spin_epochs.setRange(1, 1000)
        self.spin_epochs.setValue(50)
        train_form.addRow("Epocas:", self.spin_epochs)
        self.spin_batch = QSpinBox()
        self.spin_batch.setRange(1, 256)
        self.spin_batch.setValue(16)
        train_form.addRow("Batch size:", self.spin_batch)
        self.spin_patience = QSpinBox()
        self.spin_patience.setRange(0, 300)
        self.spin_patience.setValue(50)
        train_form.addRow("Paciencia:", self.spin_patience)
        self.spin_lr = QDoubleSpinBox()
        self.spin_lr.setDecimals(5)
        self.spin_lr.setRange(0.00001, 1.0)
        self.spin_lr.setSingleStep(0.0001)
        self.spin_lr.setValue(0.01)
        train_form.addRow("Learning rate:", self.spin_lr)
        right.addWidget(train_box)

        aug_box = self._panel("Augmentacion")
        aug_layout = QFormLayout(aug_box)
        self.chk_augment = QCheckBox("Activar augmentacion de imagenes")
        self.chk_augment.setChecked(True)
        self.chk_augment.setStyleSheet("color: #cdd6f4;")
        self.chk_augment.toggled.connect(self._set_augmentation_controls_enabled)
        aug_layout.addRow(self.chk_augment)

        self.spin_aug_percent = QSpinBox()
        self.spin_aug_percent.setRange(0, 100)
        self.spin_aug_percent.setValue(50)
        self.spin_aug_percent.setSuffix(" %")
        aug_layout.addRow("Porcentaje de imagenes:", self.spin_aug_percent)

        self.chk_rotate = QCheckBox("Rotate Step")
        self.chk_rotate.setStyleSheet("color: #cdd6f4;")
        self.combo_rotate = QComboBox()
        self.combo_rotate.addItems(["15", "30", "45", "90"])
        self.combo_rotate.setCurrentText("90")
        aug_layout.addRow(self.chk_rotate, self.combo_rotate)

        self.chk_mirror = QCheckBox("Mirror")
        self.chk_mirror.setStyleSheet("color: #cdd6f4;")
        self.combo_mirror = QComboBox()
        self.combo_mirror.addItems(["Horizontal", "Vertical", "Ambos"])
        aug_layout.addRow(self.chk_mirror, self.combo_mirror)

        self.chk_brightness = QCheckBox("Brightness Variation")
        self.chk_brightness.setChecked(True)
        self.chk_brightness.setStyleSheet("color: #cdd6f4;")
        self.spin_brightness = QSpinBox()
        self.spin_brightness.setRange(0, 100)
        self.spin_brightness.setValue(20)
        self.spin_brightness.setSuffix(" %")
        aug_layout.addRow(self.chk_brightness, self.spin_brightness)

        self.chk_brightness_spot = QCheckBox("Brightness Variation Spot")
        self.chk_brightness_spot.setChecked(True)
        self.chk_brightness_spot.setStyleSheet("color: #cdd6f4;")
        self.spin_brightness_spot = QSpinBox()
        self.spin_brightness_spot.setRange(0, 100)
        self.spin_brightness_spot.setValue(20)
        self.spin_brightness_spot.setSuffix(" %")
        aug_layout.addRow(self.chk_brightness_spot, self.spin_brightness_spot)

        self.chk_contrast = QCheckBox("Contrast Variation")
        self.chk_contrast.setChecked(True)
        self.chk_contrast.setStyleSheet("color: #cdd6f4;")
        self.spin_contrast = QSpinBox()
        self.spin_contrast.setRange(0, 100)
        self.spin_contrast.setValue(20)
        self.spin_contrast.setSuffix(" %")
        aug_layout.addRow(self.chk_contrast, self.spin_contrast)

        self.chk_saturation = QCheckBox("Saturation Variation")
        self.chk_saturation.setChecked(True)
        self.chk_saturation.setStyleSheet("color: #cdd6f4;")
        self.spin_saturation = QSpinBox()
        self.spin_saturation.setRange(0, 100)
        self.spin_saturation.setValue(20)
        self.spin_saturation.setSuffix(" %")
        aug_layout.addRow(self.chk_saturation, self.spin_saturation)

        self._set_augmentation_controls_enabled(True)
        right.addWidget(aug_box)

        actions = QHBoxLayout()
        self.btn_train = QPushButton("Iniciar nueva version")
        self.btn_train.setObjectName("btn_primary")
        self.btn_train.clicked.connect(lambda: self._toggle_training(resume_selected=False))
        actions.addWidget(self.btn_train)
        self.btn_stop = QPushButton("Detener")
        self.btn_stop.setEnabled(False)
        self.btn_stop.clicked.connect(self._stop_training)
        actions.addWidget(self.btn_stop)
        right.addLayout(actions)
        right.addStretch()
        left.addStretch()
        return page

    def _build_results_tab(self) -> QWidget:
        page = QWidget()
        layout = QVBoxLayout(page)
        layout.setContentsMargins(12, 12, 12, 12)
        layout.setSpacing(10)

        self.lbl_kpi = QLabel("Epoca: - | mAP50: - | mAP50-95: - | Precision: - | Recall: - | Loss: -")
        self.lbl_kpi.setStyleSheet("color: #cdd6f4; font-size: 12px;")
        layout.addWidget(self.lbl_kpi)

        self.progress_epoch = QProgressBar()
        self.progress_epoch.setMinimum(0)
        self.progress_epoch.setValue(0)
        self.progress_epoch.setFormat("Epoca 0/0")
        layout.addWidget(self.progress_epoch)

        self.plot = MetricsPlotWidget()
        layout.addWidget(self.plot)

        self.metrics_table = QTableWidget(0, 6)
        self.metrics_table.setHorizontalHeaderLabels(["Epoch", "mAP50", "mAP50-95", "Precision", "Recall", "Box loss"])
        self.metrics_table.horizontalHeader().setSectionResizeMode(QHeaderView.ResizeMode.Stretch)
        self.metrics_table.setStyleSheet("background: #11111b; color: #cdd6f4; border: 1px solid #313244;")
        layout.addWidget(self.metrics_table, stretch=1)

        layout.addWidget(QLabel("Consola:"))
        self.log_console = QTextEdit()
        self.log_console.setReadOnly(True)
        self.log_console.setStyleSheet("background: #11111b; color: #a6adc8; font-family: Consolas, monospace;")
        layout.addWidget(self.log_console, stretch=1)
        return page

    def set_yaml(self, path: str) -> None:
        self.yaml_label.setText(path)
        if self._selected_training and path and path != "No seleccionado":
            self._persist_training_config(self._selected_training["id"], dataset_yaml_path=path)

    def _select_yaml(self):
        path, _ = QFileDialog.getOpenFileName(self, "Seleccionar YAML", "", "YAML (*.yaml *.yml)")
        if path:
            self.set_yaml(path)

    def _project_base_dir(self) -> Path:
        if self._project_base_path:
            return Path(self._project_base_path)
        if self._db is not None:
            try:
                row = self._db.conn.execute(
                    "SELECT base_path FROM projects WHERE id = ?",
                    (self._project_id,),
                ).fetchone()
                if row and row[0]:
                    return Path(row[0])
            except Exception:
                pass
        return Path.cwd().parent

    def _training_dir(self, training_id: str) -> Path:
        safe_id = re.sub(r"[^A-Za-z0-9_.-]+", "_", str(training_id)).strip("._") or "train"
        return self._project_base_dir() / "trainings" / safe_id

    def _dataset_dir(self, training_id: str) -> Path:
        return self._training_dir(training_id) / "dataset"

    def _config_path(self, training_id: str) -> Path:
        return self._training_dir(training_id) / "config.json"

    def _existing_yaml_for_training(self, training_id: str) -> str:
        for candidate in (
            self._dataset_dir(training_id) / "data.yaml",
            self._training_dir(training_id) / "data.yaml",
        ):
            if candidate.exists():
                return str(candidate)
        return ""

    def _export_dataset_for_current_training(self) -> None:
        training_id = self._ensure_training_version()
        if not training_id:
            return
        output_dir = self._dataset_dir(training_id)
        output_dir.mkdir(parents=True, exist_ok=True)
        self.dataset_export_requested.emit(str(output_dir))
        yaml_path = self._existing_yaml_for_training(training_id)
        if yaml_path:
            self.set_yaml(yaml_path)
            self._persist_training_config(training_id, dataset_yaml_path=yaml_path)
            self.log_console.append(f"Dataset guardado en: {output_dir}")

    def _ensure_training_version(self) -> str:
        if self._selected_training:
            return str(self._selected_training["id"])
        if self._db is None:
            training_id = str(uuid.uuid4())
            self._current_training_id = training_id
            return training_id
        self._new_training_version()
        return str(self._selected_training["id"]) if self._selected_training else ""

    def _training_config(self, training_id: str, dataset_yaml_path: str = "", run_dir: str = "") -> dict:
        return {
            "training_id": training_id,
            "project_id": self._project_id,
            "model_name": self.combo_model.currentText(),
            "dataset_yaml_path": dataset_yaml_path or ("" if self.yaml_label.text() == "No seleccionado" else self.yaml_label.text()),
            "epochs": self.spin_epochs.value(),
            "image_size": self.spin_imgsz.value(),
            "batch_size": self.spin_batch.value(),
            "patience": self.spin_patience.value(),
            "learning_rate": self.spin_lr.value(),
            "device": self._current_device(),
            "augment": self.chk_augment.isChecked(),
            "augmentation_params": self._augmentation_train_params(),
            "configuration_params": self._configuration_train_params(),
            "ui_options": {
                "aug_percent": self.spin_aug_percent.value(),
                "rotate": self.chk_rotate.isChecked(),
                "rotate_step": self.combo_rotate.currentText(),
                "mirror": self.chk_mirror.isChecked(),
                "mirror_mode": self.combo_mirror.currentText(),
                "brightness": self.chk_brightness.isChecked(),
                "brightness_value": self.spin_brightness.value(),
                "brightness_spot": self.chk_brightness_spot.isChecked(),
                "brightness_spot_value": self.spin_brightness_spot.value(),
                "contrast": self.chk_contrast.isChecked(),
                "contrast_value": self.spin_contrast.value(),
                "saturation": self.chk_saturation.isChecked(),
                "saturation_value": self.spin_saturation.value(),
                "deterministic": self.chk_deterministic.isChecked(),
                "random_seed": self.chk_random_seed.isChecked(),
                "seed": self.spin_seed.value(),
            },
            "run_dir": run_dir,
            "updated_at": datetime.now(timezone.utc).isoformat(),
        }

    def _persist_training_config(self, training_id: str, dataset_yaml_path: str = "", run_dir: str = "") -> dict:
        config = self._training_config(training_id, dataset_yaml_path, run_dir)
        config_path = self._config_path(training_id)
        config_path.parent.mkdir(parents=True, exist_ok=True)
        config_path.write_text(json.dumps(config, indent=2, ensure_ascii=False), encoding="utf-8")
        if self._db is not None:
            try:
                self._db.conn.execute(
                    """
                    UPDATE trainings
                    SET dataset_yaml_path = COALESCE(NULLIF(?, ''), dataset_yaml_path),
                        model_name = ?,
                        epochs = ?,
                        image_size = ?,
                        batch_size = ?,
                        run_dir = COALESCE(NULLIF(?, ''), run_dir),
                        config_path = ?,
                        config_json = ?
                    WHERE id = ? AND project_id = ?
                    """,
                    (
                        config["dataset_yaml_path"],
                        config["model_name"],
                        config["epochs"],
                        config["image_size"],
                        config["batch_size"],
                        run_dir,
                        str(config_path),
                        json.dumps(config, ensure_ascii=False),
                        training_id,
                        self._project_id,
                    ),
                )
            except Exception as e:
                self.log_console.append(f"No se pudo guardar config de training en BD: {e}")
        return config

    def _apply_training_config(self, run: dict) -> None:
        config = {}
        raw = run.get("config_json") or ""
        if raw:
            try:
                config = json.loads(raw)
            except Exception:
                config = {}
        if not config and run.get("config_path"):
            path = Path(run["config_path"])
            if path.exists():
                try:
                    config = json.loads(path.read_text(encoding="utf-8"))
                except Exception:
                    config = {}
        if not config:
            return

        idx = self.combo_model.findText(config.get("model_name") or "")
        if idx >= 0:
            self.combo_model.setCurrentIndex(idx)
        for widget, key in (
            (self.spin_epochs, "epochs"),
            (self.spin_imgsz, "image_size"),
            (self.spin_batch, "batch_size"),
            (self.spin_patience, "patience"),
        ):
            if config.get(key) is not None:
                widget.setValue(int(config[key]))
        if config.get("learning_rate") is not None:
            self.spin_lr.setValue(float(config["learning_rate"]))
        if config.get("device") is not None:
            for idx in range(self.combo_device.count()):
                if self.combo_device.itemData(idx) == config["device"]:
                    self.combo_device.setCurrentIndex(idx)
                    break
        if config.get("augment") is not None:
            self.chk_augment.setChecked(bool(config["augment"]))
        ui = config.get("ui_options") or {}
        if ui:
            self.spin_aug_percent.setValue(int(ui.get("aug_percent", self.spin_aug_percent.value())))
            self.chk_rotate.setChecked(bool(ui.get("rotate", self.chk_rotate.isChecked())))
            self.combo_rotate.setCurrentText(str(ui.get("rotate_step", self.combo_rotate.currentText())))
            self.chk_mirror.setChecked(bool(ui.get("mirror", self.chk_mirror.isChecked())))
            self.combo_mirror.setCurrentText(str(ui.get("mirror_mode", self.combo_mirror.currentText())))
            self.chk_brightness.setChecked(bool(ui.get("brightness", self.chk_brightness.isChecked())))
            self.spin_brightness.setValue(int(ui.get("brightness_value", self.spin_brightness.value())))
            self.chk_brightness_spot.setChecked(bool(ui.get("brightness_spot", self.chk_brightness_spot.isChecked())))
            self.spin_brightness_spot.setValue(int(ui.get("brightness_spot_value", self.spin_brightness_spot.value())))
            self.chk_contrast.setChecked(bool(ui.get("contrast", self.chk_contrast.isChecked())))
            self.spin_contrast.setValue(int(ui.get("contrast_value", self.spin_contrast.value())))
            self.chk_saturation.setChecked(bool(ui.get("saturation", self.chk_saturation.isChecked())))
            self.spin_saturation.setValue(int(ui.get("saturation_value", self.spin_saturation.value())))
            self.chk_deterministic.setChecked(bool(ui.get("deterministic", self.chk_deterministic.isChecked())))
            self.chk_random_seed.setChecked(bool(ui.get("random_seed", self.chk_random_seed.isChecked())))
            self.spin_seed.setValue(int(ui.get("seed", self.spin_seed.value())))

    def _current_device(self) -> str:
        value = self.combo_device.currentData()
        return "" if value in (None, "auto") else str(value)

    def _set_augmentation_controls_enabled(self, enabled: bool) -> None:
        widgets = [
            self.spin_aug_percent,
            self.chk_rotate, self.combo_rotate,
            self.chk_mirror, self.combo_mirror,
            self.chk_brightness, self.spin_brightness,
            self.chk_brightness_spot, self.spin_brightness_spot,
            self.chk_contrast, self.spin_contrast,
            self.chk_saturation, self.spin_saturation,
        ]
        for widget in widgets:
            widget.setEnabled(enabled)

    def _augmentation_train_params(self) -> dict:
        if not self.chk_augment.isChecked():
            return {
                "degrees": 0.0,
                "fliplr": 0.0,
                "flipud": 0.0,
                "hsv_h": 0.0,
                "hsv_s": 0.0,
                "hsv_v": 0.0,
            }

        probability = self.spin_aug_percent.value() / 100.0
        params = {
            "hsv_h": 0.015,
            "hsv_s": 0.0,
            "hsv_v": 0.0,
            "degrees": 0.0,
            "fliplr": 0.0,
            "flipud": 0.0,
        }

        if self.chk_rotate.isChecked():
            params["degrees"] = float(self.combo_rotate.currentText())

        if self.chk_mirror.isChecked():
            mirror = self.combo_mirror.currentText()
            if mirror in ("Horizontal", "Ambos"):
                params["fliplr"] = probability
            if mirror in ("Vertical", "Ambos"):
                params["flipud"] = probability

        if self.chk_brightness.isChecked() or self.chk_brightness_spot.isChecked():
            value = max(self.spin_brightness.value(), self.spin_brightness_spot.value())
            params["hsv_v"] = min(value / 100.0, 1.0)

        if self.chk_saturation.isChecked():
            params["hsv_s"] = min(self.spin_saturation.value() / 100.0, 1.0)

        return params

    def _augmentation_summary(self) -> str:
        if not self.chk_augment.isChecked():
            return "Augmentacion desactivada"
        parts = [f"prob={self.spin_aug_percent.value()}%"]
        if self.chk_rotate.isChecked():
            parts.append(f"rotate=±{self.combo_rotate.currentText()} grados")
        if self.chk_mirror.isChecked():
            parts.append(f"mirror={self.combo_mirror.currentText()}")
        if self.chk_brightness.isChecked():
            parts.append(f"brightness={self.spin_brightness.value()}%")
        if self.chk_brightness_spot.isChecked():
            parts.append(f"brightness_spot={self.spin_brightness_spot.value()}%")
        if self.chk_contrast.isChecked():
            parts.append(f"contrast={self.spin_contrast.value()}% (visual, no directo YOLO)")
        if self.chk_saturation.isChecked():
            parts.append(f"saturation={self.spin_saturation.value()}%")
        return "Augmentacion: " + ", ".join(parts)

    def _configuration_train_params(self) -> dict:
        params = {"deterministic": self.chk_deterministic.isChecked()}
        if self.chk_random_seed.isChecked():
            params["seed"] = self.spin_seed.value()
        return params

    def _populate_device_options(self) -> None:
        self.combo_device.clear()
        self.combo_device.addItem("Auto - elegir automaticamente", "auto")

        cpu_name = platform.processor() or platform.machine() or "CPU"
        self.combo_device.addItem(f"CPU - {cpu_name}", "cpu")

        if HAS_TORCH and torch.cuda.is_available():
            for idx in range(torch.cuda.device_count()):
                try:
                    name = torch.cuda.get_device_name(idx)
                except Exception:
                    name = f"CUDA device {idx}"
                self.combo_device.addItem(f"GPU {idx} - {name}", str(idx))
        else:
            for idx, name in enumerate(self._nvidia_smi_gpu_names()):
                self.combo_device.addItem(
                    f"GPU {idx} - {name} (detectada, PyTorch sin CUDA)",
                    f"unavailable:{idx}",
                )
                item = self.combo_device.model().item(self.combo_device.count() - 1)
                if item is not None:
                    item.setEnabled(False)

    def _nvidia_smi_gpu_names(self) -> list[str]:
        try:
            output = subprocess.check_output(
                [
                    "nvidia-smi",
                    "--query-gpu=name",
                    "--format=csv,noheader",
                ],
                text=True,
                stderr=subprocess.DEVNULL,
                timeout=3,
            )
        except Exception:
            return []
        return [line.strip() for line in output.splitlines() if line.strip()]

    def _new_training_version(self):
        if self._db is None:
            self._selected_training = None
            self.training_list.clearSelection()
            self.lbl_title.setText("Nueva version de entrenamiento")
            self.tabs.setCurrentIndex(0)
            return

        training_id = str(uuid.uuid4())
        yaml_path = "" if self.yaml_label.text() == "No seleccionado" else self.yaml_label.text()
        train_dir = self._training_dir(training_id)
        train_dir.mkdir(parents=True, exist_ok=True)
        config = self._training_config(training_id, dataset_yaml_path=yaml_path)
        config_path = self._config_path(training_id)
        try:
            config_path.write_text(json.dumps(config, indent=2, ensure_ascii=False), encoding="utf-8")
            self._db.create_training_run(
                training_id=training_id,
                project_id=self._project_id,
                model_name=self.combo_model.currentText(),
                dataset_yaml_path=yaml_path,
                epochs=self.spin_epochs.value(),
                image_size=self.spin_imgsz.value(),
                batch_size=self.spin_batch.value(),
                status="pending",
                run_dir=str(train_dir / "runs"),
                config_path=str(config_path),
                config_json=json.dumps(config, ensure_ascii=False),
            )
        except Exception as e:
            self.tabs.setCurrentIndex(1)
            self.log_console.append(f"No se pudo crear la version: {e}")
            return

        self._load_training_runs(select_id=training_id)
        self.lbl_title.setText("Nueva version de entrenamiento")
        self.tabs.setCurrentIndex(0)

    def _delete_selected_training(self):
        if not self._selected_training:
            QMessageBox.information(self, "Sin seleccion", "Selecciona un entrenamiento para eliminar.")
            return
        if self.worker is not None and self.worker.isRunning():
            QMessageBox.warning(
                self,
                "Entrenamiento activo",
                "No puedes eliminar una version mientras hay un entrenamiento en curso.",
            )
            return

        run = self._selected_training
        reply = QMessageBox.question(
            self,
            "Eliminar entrenamiento",
            (
                "?Eliminar esta version de entrenamiento?\n\n"
                f"{self._run_label(run)}\n\n"
                "Se borrara el registro de la base de datos y tambien los archivos "
                "fisicos del run si existen."
            ),
            QMessageBox.StandardButton.Yes | QMessageBox.StandardButton.No,
            QMessageBox.StandardButton.No,
        )
        if reply != QMessageBox.StandardButton.Yes:
            return

        try:
            deleted_paths = self._delete_training_files(run)
            if self._db is not None:
                self._db.delete_training_run(run["id"], self._project_id)
            self._selected_training = None
            self._load_training_runs()
            self.lbl_title.setText("Entrenamiento YOLO")
            self.log_console.append("Version de entrenamiento eliminada.")
            if deleted_paths:
                self.log_console.append("Archivos eliminados:")
                for path in deleted_paths:
                    self.log_console.append(f" - {path}")
        except Exception as e:
            QMessageBox.critical(
                self,
                "Error al eliminar",
                f"No se pudo eliminar el entrenamiento.\n\nDetalle: {e}",
            )

    def _delete_training_files(self, run: dict) -> list[str]:
        targets: set[Path] = set()
        for key in ("weights_path", "metrics_path", "config_path"):
            value = run.get(key)
            if value:
                path = Path(value)
                if path.exists():
                    targets.add(path)
                    run_dir = self._run_dir_from_training_file(path)
                    if run_dir:
                        targets.add(run_dir)
                    train_dir = self._train_dir_from_training_file(path)
                    if train_dir:
                        targets.add(train_dir)
        yaml_path = run.get("dataset_yaml_path")
        if yaml_path:
            train_dir = self._train_dir_from_training_file(Path(yaml_path))
            if train_dir and train_dir.exists():
                targets.add(train_dir)

        deleted: list[str] = []
        for target in sorted(targets, key=lambda p: len(p.parts), reverse=True):
            if not target.exists():
                continue
            if target.is_dir():
                if self._is_safe_training_run_dir(target) or self._is_safe_training_dir(target):
                    shutil.rmtree(target)
                    deleted.append(str(target))
            elif target.is_file():
                if self._is_safe_training_file(target):
                    target.unlink()
                    deleted.append(str(target))
        return deleted

    def _run_dir_from_training_file(self, path: Path) -> Path | None:
        parts = path.parts
        if "weights" in parts:
            return path.parent.parent
        if path.name == "results.csv":
            return path.parent
        return None

    def _train_dir_from_training_file(self, path: Path) -> Path | None:
        resolved = path.resolve()
        parts = resolved.parts
        if "trainings" not in parts:
            return None
        idx = parts.index("trainings")
        if len(parts) <= idx + 1:
            return None
        return Path(*parts[:idx + 2])

    def _is_safe_training_run_dir(self, path: Path) -> bool:
        try:
            resolved = path.resolve()
        except Exception:
            return False
        return (
            resolved.exists()
            and resolved.is_dir()
            and resolved.parent.name == "runs"
            and (resolved / "weights").exists()
        )

    def _is_safe_training_dir(self, path: Path) -> bool:
        try:
            resolved = path.resolve()
        except Exception:
            return False
        return (
            resolved.exists()
            and resolved.is_dir()
            and resolved.parent.name == "trainings"
            and ((resolved / "config.json").exists() or (resolved / "dataset" / "data.yaml").exists())
        )

    def _is_safe_training_file(self, path: Path) -> bool:
        try:
            resolved = path.resolve()
        except Exception:
            return False
        return (
            resolved.exists()
            and resolved.is_file()
            and (
                resolved.parent.name == "weights"
                or resolved.name == "results.csv"
                or (resolved.name == "config.json" and resolved.parent.parent.name == "trainings")
            )
        )

    def _toggle_training(self, resume_selected: bool = False):
        if self.worker is not None and self.worker.isRunning():
            self._stop_training()
            return

        now = datetime.now(timezone.utc)
        use_selected_pending = (
            not resume_selected
            and self._selected_training is not None
            and self._selected_training.get("status") == "pending"
        )
        self._current_training_id = (
            self._selected_training["id"] if use_selected_pending else str(uuid.uuid4())
        )

        output_dir = self._dataset_dir(self._current_training_id)
        output_dir.mkdir(parents=True, exist_ok=True)
        self.dataset_export_requested.emit(str(output_dir))
        yaml_path = self._existing_yaml_for_training(self._current_training_id) or self.yaml_label.text()
        if yaml_path == "No seleccionado":
            self.tabs.setCurrentIndex(1)
            self.log_console.append("No se pudo generar el data.yaml para este entrenamiento.")
            return
        self.set_yaml(yaml_path)

        resume_checkpoint = ""
        if resume_selected and self._selected_training:
            resume_checkpoint = self._selected_training.get("weights_path") or ""
            if not resume_checkpoint or not Path(resume_checkpoint).exists():
                self.tabs.setCurrentIndex(1)
                self.log_console.append("No hay pesos disponibles para reentrenar este modelo. Se iniciara una nueva version.")
                resume_selected = False

        self._prepare_live_results()
        self.tabs.setCurrentIndex(1)
        self.btn_train.setEnabled(False)
        self.btn_resume_run.setEnabled(False)
        self.btn_stop.setEnabled(True)
        self.lbl_status.setText("Entrenando...")

        run_prefix = "resume" if resume_selected else "train"
        run_name = f"{run_prefix}_{now.strftime('%Y%m%d_%H%M%S')}_{self._project_id}"
        train_dir = self._training_dir(self._current_training_id)
        run_project_dir = str(train_dir / "runs")
        model_name = self.combo_model.currentText()
        self._last_run_info = {"run_dir": str(Path(run_project_dir) / run_name)}
        config = self._persist_training_config(
            self._current_training_id,
            dataset_yaml_path=yaml_path,
            run_dir=self._last_run_info["run_dir"],
        )

        if self._db is not None and not use_selected_pending:
            try:
                self._db.create_training_run(
                    training_id=self._current_training_id,
                    project_id=self._project_id,
                    model_name=model_name,
                    dataset_yaml_path=yaml_path,
                    epochs=self.spin_epochs.value(),
                    image_size=self.spin_imgsz.value(),
                    batch_size=self.spin_batch.value(),
                    status="running",
                    run_dir=self._last_run_info["run_dir"],
                    config_path=str(self._config_path(self._current_training_id)),
                    config_json=json.dumps(config, ensure_ascii=False),
                )
            except Exception as e:
                self.log_console.append(f"No se pudo registrar training en BD: {e}")
        elif self._db is not None and use_selected_pending:
            try:
                self._db.conn.execute(
                    """
                    UPDATE trainings
                    SET model_name = ?,
                        dataset_yaml_path = ?,
                        epochs = ?,
                        image_size = ?,
                        batch_size = ?,
                        status = 'running',
                        started_at = ?,
                        run_dir = ?,
                        config_path = ?,
                        config_json = ?
                    WHERE id = ?
                    """,
                    (
                        model_name,
                        yaml_path,
                        self.spin_epochs.value(),
                        self.spin_imgsz.value(),
                        self.spin_batch.value(),
                        now,
                        self._last_run_info["run_dir"],
                        str(self._config_path(self._current_training_id)),
                        json.dumps(config, ensure_ascii=False),
                        self._current_training_id,
                    ),
                )
            except Exception as e:
                self.log_console.append(f"No se pudo actualizar la version en BD: {e}")

        self.log_console.append(f"Run: {run_name}")
        self.log_console.append(f"Modelo base: {model_name}")
        self.log_console.append(f"Dataset: {yaml_path}")
        self.log_console.append(
            f"Parametros: epochs={self.spin_epochs.value()}, imgsz={self.spin_imgsz.value()}, "
            f"batch={self.spin_batch.value()}, patience={self.spin_patience.value()}, lr={self.spin_lr.value()}"
        )
        aug_params = self._augmentation_train_params()
        config_params = self._configuration_train_params()
        self.log_console.append(self._augmentation_summary())
        self.log_console.append(f"Parametros YOLO augment: {aug_params}")
        self.log_console.append(f"Configuracion avanzada: {config_params}")
        if resume_selected:
            self.log_console.append(f"Reentrenando desde: {resume_checkpoint}")

        self.worker = YoloTrainerWorker(
            dataset_yaml=yaml_path,
            epochs=self.spin_epochs.value(),
            imgsz=self.spin_imgsz.value(),
            batch=self.spin_batch.value(),
            patience=self.spin_patience.value(),
            model_name=model_name,
            run_project_dir=run_project_dir,
            run_name=run_name,
            learning_rate=self.spin_lr.value(),
            device=self._current_device(),
            augment=self.chk_augment.isChecked(),
            augmentation_params={**aug_params, **config_params},
            resume=resume_selected,
            resume_checkpoint=resume_checkpoint,
        )
        self.worker.log_msg.connect(self.log_console.append)
        self.worker.epoch_metrics.connect(self._on_epoch_metrics)
        self.worker.finished_training.connect(self._on_training_finished)
        self.worker.finished.connect(self._on_worker_done)
        self.worker.start()

    def _prepare_live_results(self):
        self.plot.clear()
        self.log_console.clear()
        self.metrics_table.setRowCount(0)
        self.progress_epoch.setMaximum(self.spin_epochs.value())
        self.progress_epoch.setValue(0)
        self.progress_epoch.setFormat(f"Epoca 0/{self.spin_epochs.value()}")
        self.lbl_kpi.setText("Epoca: - | mAP50: - | mAP50-95: - | Precision: - | Recall: - | Loss: -")

    def _stop_training(self):
        if self.worker is not None and self.worker.isRunning():
            self.btn_stop.setEnabled(False)
            self.worker.stop()

    def _on_epoch_metrics(self, data: dict):
        epoch = data.get("epoch", "-")
        map50 = data.get("map50")
        map50_95 = data.get("map50_95")
        precision = data.get("precision")
        recall = data.get("recall")
        box_loss = data.get("box_loss")
        self.plot.append(map50, box_loss)
        try:
            current_epoch = int(epoch)
            total = self.spin_epochs.value()
            self.progress_epoch.setMaximum(total)
            self.progress_epoch.setValue(min(current_epoch, total))
            self.progress_epoch.setFormat(f"Epoca {current_epoch}/{total}")
        except Exception:
            pass

        row = self.metrics_table.rowCount()
        self.metrics_table.insertRow(row)
        values = [epoch, map50, map50_95, precision, recall, box_loss]
        for col, value in enumerate(values):
            text = "-" if value is None else (f"{value:.5f}" if isinstance(value, float) else str(value))
            self.metrics_table.setItem(row, col, QTableWidgetItem(text))
        self.metrics_table.scrollToBottom()

        self.log_console.append(
            f"Epoch {epoch}: mAP50={self._fmt(map50)}, mAP50-95={self._fmt(map50_95)}, "
            f"precision={self._fmt(precision)}, recall={self._fmt(recall)}, box_loss={self._fmt(box_loss)}"
        )
        self.lbl_kpi.setText(
            f"Epoca: {epoch} | mAP50: {self._fmt(map50)} | mAP50-95: {self._fmt(map50_95)} | "
            f"Precision: {self._fmt(precision)} | Recall: {self._fmt(recall)} | Loss: {self._fmt(box_loss)}"
        )

    def _on_training_finished(self, success: bool, msg: str, info: dict):
        self._last_run_info = info or self._last_run_info
        self.log_console.append(msg)

        summary = self._summarize_metrics_csv(self._last_run_info.get("metrics_path", ""))
        if summary:
            self.log_console.append(
                f"Mejor Epoca: {summary['best_epoch']} | "
                f"best mAP50={self._fmt(summary.get('best_map50'))} | "
                f"best mAP50-95={self._fmt(summary.get('best_map50_95'))}"
            )
            if summary.get("final_epoch"):
                self.log_console.append(f"Epoca final registrada: {summary['final_epoch']}")

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
                    run_dir=self._last_run_info.get("run_dir", ""),
                    config_path=str(self._config_path(self._current_training_id)),
                    config_json=json.dumps(
                        self._training_config(
                            self._current_training_id,
                            dataset_yaml_path="" if self.yaml_label.text() == "No seleccionado" else self.yaml_label.text(),
                            run_dir=self._last_run_info.get("run_dir", ""),
                        ),
                        ensure_ascii=False,
                    ),
                )
            except Exception as e:
                self.log_console.append(f"No se pudo actualizar training en BD: {e}")

        if self._last_run_info.get("weights_path"):
            self.log_console.append(f"Pesos: {self._last_run_info['weights_path']}")
        if self._last_run_info.get("metrics_path"):
            self.log_console.append(f"Metricas: {self._last_run_info['metrics_path']}")
        if self._last_run_info.get("run_dir"):
            self.log_console.append(f"Run dir: {self._last_run_info['run_dir']}")
        self.lbl_status.setText("Finalizado" if success else "Revisar resultado")

    def _on_worker_done(self):
        self.worker = None
        self.btn_train.setEnabled(True)
        self.btn_resume_run.setEnabled(True)
        self.btn_stop.setEnabled(False)
        self._load_training_runs()

    def _load_training_runs(self, select_id: str | None = None):
        self.training_list.clear()
        if self._db is None:
            return
        rows = self._db.conn.execute(
            """
            SELECT id, model_name, dataset_yaml_path, epochs, image_size, batch_size,
                   status, metrics_path, weights_path, run_dir, config_path, config_json,
                   started_at, finished_at
            FROM trainings
            WHERE project_id = ?
            ORDER BY COALESCE(started_at, finished_at) DESC
            """,
            (self._project_id,),
        ).fetchall()
        keys = [
            "id", "model_name", "dataset_yaml_path", "epochs", "image_size", "batch_size",
            "status", "metrics_path", "weights_path", "run_dir", "config_path", "config_json",
            "started_at", "finished_at",
        ]
        for row in rows:
            run = dict(zip(keys, row))
            item = QListWidgetItem(self._run_label(run))
            item.setData(Qt.ItemDataRole.UserRole, run)
            self.training_list.addItem(item)
            if select_id and run.get("id") == select_id:
                self.training_list.setCurrentItem(item)

    def _run_label(self, run: dict) -> str:
        started = str(run.get("started_at") or "")[:19].replace("T", " ")
        if not started:
            started = "Version pendiente"
        model = run.get("model_name") or "modelo"
        status = run.get("status") or "pending"
        return f"{started}\n{model} - {run.get('image_size')}px - {run.get('epochs')} ep - {status}"

    def _on_training_selected(self, current: QListWidgetItem, previous: QListWidgetItem):
        del previous
        if not current:
            return
        self._selected_training = current.data(Qt.ItemDataRole.UserRole)
        run = self._selected_training
        self.lbl_title.setText(f"Version: {run.get('model_name')} - {run.get('status')}")
        yaml_path = run.get("dataset_yaml_path") or self._existing_yaml_for_training(run["id"])
        if yaml_path:
            self.yaml_label.setText(yaml_path)
            if not run.get("dataset_yaml_path"):
                self._persist_training_config(run["id"], dataset_yaml_path=yaml_path)
                run["dataset_yaml_path"] = yaml_path
        if run.get("epochs"):
            self.spin_epochs.setValue(int(run["epochs"]))
        if run.get("image_size"):
            self.spin_imgsz.setValue(int(run["image_size"]))
        if run.get("batch_size"):
            self.spin_batch.setValue(int(run["batch_size"]))
        idx = self.combo_model.findText(run.get("model_name") or "")
        if idx >= 0:
            self.combo_model.setCurrentIndex(idx)
        self._apply_training_config(run)
        self._show_run_results(run)

    def _show_run_results(self, run: dict):
        self.tabs.setCurrentIndex(1)
        self.plot.clear()
        self.metrics_table.setRowCount(0)
        self.log_console.clear()
        self.log_console.append(f"Training ID: {run.get('id')}")
        self.log_console.append(f"Estado: {run.get('status')}")
        if run.get("weights_path"):
            self.log_console.append(f"Pesos: {run.get('weights_path')}")
        if run.get("dataset_yaml_path"):
            self.log_console.append(f"Dataset YAML: {run.get('dataset_yaml_path')}")
        if run.get("config_path"):
            self.log_console.append(f"Config: {run.get('config_path')}")
        if run.get("run_dir"):
            self.log_console.append(f"Run dir: {run.get('run_dir')}")
        if run.get("metrics_path"):
            self.log_console.append(f"Metricas: {run.get('metrics_path')}")
            self._load_metrics_table(run.get("metrics_path"))
        else:
            self.log_console.append("Esta version aun no tiene metricas registradas.")

    def _load_metrics_table(self, metrics_path: str):
        path = Path(metrics_path)
        if not path.exists():
            self.log_console.append("El archivo results.csv no existe en disco.")
            return
        try:
            with path.open("r", encoding="utf-8", newline="") as f:
                rows = list(csv.DictReader(f))
        except Exception as e:
            self.log_console.append(f"No se pudieron leer metricas: {e}")
            return
        for r in rows:
            payload = self._metrics_row_payload(r)
            self._append_metric_row(payload)
            self.plot.append(payload.get("map50"), payload.get("box_loss"))
        summary = self._summarize_metrics_rows(rows)
        if summary:
            self.log_console.append(
                f"Mejor Epoca: {summary['best_epoch']} | "
                f"best mAP50={self._fmt(summary.get('best_map50'))} | "
                f"best mAP50-95={self._fmt(summary.get('best_map50_95'))}"
            )

    def _append_metric_row(self, payload: dict):
        row = self.metrics_table.rowCount()
        self.metrics_table.insertRow(row)
        values = [
            payload.get("epoch"), payload.get("map50"), payload.get("map50_95"),
            payload.get("precision"), payload.get("recall"), payload.get("box_loss"),
        ]
        for col, value in enumerate(values):
            text = "-" if value is None else (f"{value:.5f}" if isinstance(value, float) else str(value))
            self.metrics_table.setItem(row, col, QTableWidgetItem(text))

    def _metrics_row_payload(self, row: dict) -> dict:
        return {
            "epoch": self._csv_number(row, "epoch", as_int=True),
            "map50": self._csv_number(row, "metrics/mAP50(B)", "metrics/mAP50"),
            "map50_95": self._csv_number(row, "metrics/mAP50-95(B)", "metrics/mAP50-95"),
            "precision": self._csv_number(row, "metrics/precision(B)", "metrics/precision"),
            "recall": self._csv_number(row, "metrics/recall(B)", "metrics/recall"),
            "box_loss": self._csv_number(row, "train/box_loss", "val/box_loss"),
        }

    def _summarize_metrics_csv(self, metrics_path: str) -> dict:
        path = Path(metrics_path) if metrics_path else None
        if not path or not path.exists():
            return {}
        try:
            with path.open("r", encoding="utf-8", newline="") as f:
                return self._summarize_metrics_rows(list(csv.DictReader(f)))
        except Exception:
            return {}

    def _summarize_metrics_rows(self, rows: list[dict]) -> dict:
        best = None
        for row in rows:
            payload = self._metrics_row_payload(row)
            score = payload.get("map50")
            if score is None:
                continue
            if best is None or score > best.get("best_map50", -1):
                best = {
                    "best_epoch": payload.get("epoch"),
                    "best_map50": score,
                    "best_map50_95": payload.get("map50_95"),
                }
        if best and rows:
            best["final_epoch"] = self._metrics_row_payload(rows[-1]).get("epoch")
        return best or {}

    def _csv_number(self, row: dict, *keys: str, as_int: bool = False):
        for key in keys:
            if key in row and str(row[key]).strip() != "":
                try:
                    value = float(row[key])
                    return int(value) if as_int else value
                except Exception:
                    return None
        return None

    def _fmt(self, value) -> str:
        if value is None:
            return "-"
        try:
            return f"{float(value):.4f}"
        except Exception:
            return str(value)


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


class YoloImageInferenceWorker(QThread):
    result_ready = pyqtSignal(QImage)
    error_msg = pyqtSignal(str)

    def __init__(self, model_path: str, image_path: str, parent=None):
        super().__init__(parent)
        self.model_path = model_path
        self.image_path = image_path

    def run(self):
        if not HAS_YOLO:
            self.error_msg.emit("Error: librería ultralytics no encontrada.")
            return
        try:
            model = YOLO(self.model_path)
            results = model(self.image_path, verbose=False)
            annotated = results[0].plot()
            rgb = cv2.cvtColor(annotated, cv2.COLOR_BGR2RGB)
            h, w, ch = rgb.shape
            qt_img = QImage(rgb.data.tobytes(), w, h, ch * w, QImage.Format.Format_RGB888)
            self.result_ready.emit(qt_img)
        except Exception as e:
            self.error_msg.emit(f"Error: {str(e)}")


class InferenceWidget(QWidget):

    def __init__(self, *, db=None, project_id: str = "", parent=None):
        super().__init__(parent)
        self._db = db
        self._project_id = project_id
        self.worker: YoloInferenceWorker | None = None
        self._img_worker: YoloImageInferenceWorker | None = None
        self.model_path = "yolov8n.pt"
        self._build_ui()
        self._load_trained_models()

    def _build_ui(self):
        layout = QVBoxLayout(self)
        layout.setContentsMargins(20, 20, 20, 20)
        layout.setSpacing(12)

        title = QLabel("Demo de Inferencia")
        title.setStyleSheet("color: #cdd6f4; font-size: 24px; font-weight: bold;")
        layout.addWidget(title)

        # --- Selector de modelo ---
        model_box = QGroupBox("Modelo")
        model_box.setStyleSheet(
            "QGroupBox { color: #cdd6f4; font-weight: bold; border: 1px solid #313244; "
            "border-radius: 8px; margin-top: 10px; padding-top: 12px; background: #1e1e2e; }"
            "QGroupBox::title { subcontrol-origin: margin; left: 10px; padding: 0 4px; }"
            "QLabel { color: #cdd6f4; }"
        )
        model_vlay = QVBoxLayout(model_box)

        combo_row = QHBoxLayout()
        self.combo_models = QComboBox()
        self.combo_models.setStyleSheet(
            "QComboBox { background: #11111b; color: #cdd6f4; border: 1px solid #313244; padding: 4px; }"
        )
        self.combo_models.currentIndexChanged.connect(self._on_combo_model_changed)
        combo_row.addWidget(self.combo_models, stretch=1)

        btn_refresh = QPushButton("↺")
        btn_refresh.setFixedWidth(32)
        btn_refresh.setToolTip("Actualizar lista de modelos entrenados")
        btn_refresh.setStyleSheet(
            "QPushButton { background: #313244; color: #cdd6f4; border: none; border-radius: 4px; font-size: 16px; }"
            "QPushButton:hover { background: #45475a; }"
        )
        btn_refresh.clicked.connect(self._load_trained_models)
        combo_row.addWidget(btn_refresh)
        model_vlay.addLayout(combo_row)

        self.btn_load_file = QPushButton("Cargar desde archivo (.pt)...")
        self.btn_load_file.setStyleSheet(
            "QPushButton { background: transparent; color: #89b4fa; border: none; "
            "text-align: left; padding: 2px 0; font-size: 11px; }"
            "QPushButton:hover { color: #cdd6f4; }"
        )
        self.btn_load_file.clicked.connect(self._load_model_from_file)
        model_vlay.addWidget(self.btn_load_file)

        self.lbl_model = QLabel(f"Activo: {self.model_path}")
        self.lbl_model.setStyleSheet("color: #a6e3a1; font-size: 11px;")
        model_vlay.addWidget(self.lbl_model)
        layout.addWidget(model_box)

        # --- Toggle de modo ---
        _toggle_style = (
            "QPushButton { background: #313244; color: #cdd6f4; border: none;"
            " border-radius: 6px; padding: 8px 20px; font-size: 13px; }"
            "QPushButton:checked { background: #0f9f8f; color: white; }"
            "QPushButton:hover:!checked { background: #45475a; }"
        )
        mode_row = QHBoxLayout()
        self.btn_mode_camera = QPushButton("📷  Cámara en tiempo real")
        self.btn_mode_camera.setCheckable(True)
        self.btn_mode_camera.setChecked(True)
        self.btn_mode_camera.setStyleSheet(_toggle_style)
        self.btn_mode_camera.clicked.connect(lambda: self._set_mode("camera"))

        self.btn_mode_image = QPushButton("🖼  Imagen")
        self.btn_mode_image.setCheckable(True)
        self.btn_mode_image.setStyleSheet(_toggle_style)
        self.btn_mode_image.clicked.connect(lambda: self._set_mode("image"))

        mode_row.addWidget(self.btn_mode_camera)
        mode_row.addWidget(self.btn_mode_image)
        mode_row.addStretch()
        layout.addLayout(mode_row)

        # --- Página cámara ---
        _display_style = (
            "background: #11111b; border: 2px solid #45475a; border-radius: 8px; color: #6c7086;"
        )
        self._page_camera = QWidget()
        cam_lay = QVBoxLayout(self._page_camera)
        cam_lay.setContentsMargins(0, 0, 0, 0)
        cam_lay.setSpacing(8)

        self.lbl_camera = QLabel("Cámara apagada")
        self.lbl_camera.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.lbl_camera.setStyleSheet(_display_style)
        self.lbl_camera.setMinimumSize(640, 420)
        cam_lay.addWidget(self.lbl_camera, stretch=1)

        self.btn_start = QPushButton("▶ Iniciar Cámara")
        self.btn_start.setObjectName("btn_primary")
        self.btn_start.clicked.connect(self._toggle_camera)
        cam_lay.addWidget(self.btn_start)

        # --- Página imagen ---
        self._page_image = QWidget()
        img_lay = QVBoxLayout(self._page_image)
        img_lay.setContentsMargins(0, 0, 0, 0)
        img_lay.setSpacing(8)

        self.lbl_image = QLabel("Sin imagen")
        self.lbl_image.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.lbl_image.setStyleSheet(_display_style)
        self.lbl_image.setMinimumSize(640, 420)
        img_lay.addWidget(self.lbl_image, stretch=1)

        img_ctrl = QHBoxLayout()
        self.btn_load_image = QPushButton("📂 Cargar imagen")
        self.btn_load_image.setObjectName("btn_primary")
        self.btn_load_image.clicked.connect(self._load_and_infer_image)
        self.lbl_image_status = QLabel("")
        self.lbl_image_status.setStyleSheet("color: #a6adc8; font-size: 11px;")
        img_ctrl.addWidget(self.btn_load_image)
        img_ctrl.addWidget(self.lbl_image_status, stretch=1)
        img_lay.addLayout(img_ctrl)

        # --- Stack ---
        self.display_stack = QStackedWidget()
        self.display_stack.addWidget(self._page_camera)  # 0
        self.display_stack.addWidget(self._page_image)   # 1
        layout.addWidget(self.display_stack, stretch=1)

    def _load_trained_models(self) -> None:
        self.combo_models.blockSignals(True)
        self.combo_models.clear()
        self.combo_models.addItem("yolov8n.pt (por defecto)", "yolov8n.pt")

        if self._db is not None and self._project_id:
            try:
                rows = self._db.conn.execute(
                    """
                    SELECT model_name, weights_path, started_at
                    FROM trainings
                    WHERE project_id = ? AND status = 'finished'
                      AND weights_path IS NOT NULL AND weights_path <> ''
                    ORDER BY started_at DESC
                    """,
                    (self._project_id,),
                ).fetchall()
                for model_name, weights_path, started_at in rows:
                    if not Path(weights_path).exists():
                        continue
                    date = str(started_at or "")[:10]
                    label = f"{model_name}  —  {date}  [{Path(weights_path).name}]"
                    self.combo_models.addItem(label, weights_path)
            except Exception:
                pass

        self.combo_models.blockSignals(False)

    def _on_combo_model_changed(self, index: int) -> None:
        path = self.combo_models.itemData(index)
        if path:
            self.model_path = path
            self.lbl_model.setText(f"Activo: {Path(path).name}")

    def _load_model_from_file(self) -> None:
        path, _ = QFileDialog.getOpenFileName(self, "Seleccionar Modelo", "", "PyTorch Models (*.pt)")
        if path:
            self.model_path = path
            self.lbl_model.setText(f"Activo: {Path(path).name}")
            self.combo_models.blockSignals(True)
            self.combo_models.setCurrentIndex(-1)
            self.combo_models.blockSignals(False)

    def _set_mode(self, mode: str) -> None:
        if mode == "camera":
            self.btn_mode_camera.setChecked(True)
            self.btn_mode_image.setChecked(False)
            self.display_stack.setCurrentIndex(0)
        else:
            self.stop_camera()
            self.btn_mode_camera.setChecked(False)
            self.btn_mode_image.setChecked(True)
            self.display_stack.setCurrentIndex(1)

    def stop_camera(self) -> None:
        if self.worker is not None and self.worker.isRunning():
            self.worker.stop()
            self.worker = None
            self.btn_start.setText("▶ Iniciar Cámara")
            self.lbl_camera.setText("Cámara apagada")

    def hideEvent(self, event):
        self.stop_camera()
        super().hideEvent(event)

    def _toggle_camera(self) -> None:
        if self.worker is not None and self.worker.isRunning():
            self.stop_camera()
            return
        self.btn_start.setText("⏹ Detener Cámara")
        self.worker = YoloInferenceWorker(model_path=self.model_path)
        self.worker.frame_ready.connect(self._update_camera_frame)
        self.worker.error_msg.connect(self.lbl_camera.setText)
        self.worker.start()

    def _update_camera_frame(self, image: QImage) -> None:
        pixmap = QPixmap.fromImage(image)
        scaled = pixmap.scaled(
            self.lbl_camera.size(),
            Qt.AspectRatioMode.KeepAspectRatio,
            Qt.TransformationMode.SmoothTransformation,
        )
        self.lbl_camera.setPixmap(scaled)

    def _load_and_infer_image(self) -> None:
        path, _ = QFileDialog.getOpenFileName(
            self, "Seleccionar imagen", "",
            "Imágenes (*.jpg *.jpeg *.png *.bmp *.tiff *.webp)"
        )
        if not path:
            return
        self.lbl_image_status.setText("Procesando...")
        self.btn_load_image.setEnabled(False)
        self._img_worker = YoloImageInferenceWorker(model_path=self.model_path, image_path=path)
        self._img_worker.result_ready.connect(self._show_image_result)
        self._img_worker.error_msg.connect(self._on_image_error)
        self._img_worker.finished.connect(lambda: self.btn_load_image.setEnabled(True))
        self._img_worker.start()

    def _show_image_result(self, image: QImage) -> None:
        pixmap = QPixmap.fromImage(image)
        scaled = pixmap.scaled(
            self.lbl_image.size(),
            Qt.AspectRatioMode.KeepAspectRatio,
            Qt.TransformationMode.SmoothTransformation,
        )
        self.lbl_image.setPixmap(scaled)
        self.lbl_image_status.setText("Listo")

    def _on_image_error(self, msg: str) -> None:
        self.lbl_image.setText(msg)
        self.lbl_image_status.setText("Error")
