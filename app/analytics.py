import pyqtgraph as pg
from PyQt6.QtWidgets import (
    QWidget, QVBoxLayout, QHBoxLayout, QPushButton, QLabel, QFrame, QGridLayout, QComboBox, QScrollArea, QSizePolicy
)
from PyQt6.QtCore import Qt, QRectF
from PyQt6.QtGui import QColor, QPainter, QBrush, QPen, QFont
import math

from app.database import DatabaseManager

class PieChartItem(pg.GraphicsObject):
    def __init__(self, data, colors, show_text=True):
        super().__init__()
        self.data = data
        self.colors = [QColor(c) for c in colors]
        self.show_text = show_text
        self.rect = QRectF(-40, -40, 80, 80)

    def boundingRect(self):
        padding = 10
        return self.rect.adjusted(-padding, -padding, padding, padding)

    def paint(self, p: QPainter, *args):
        p.setRenderHint(QPainter.RenderHint.Antialiasing)
        total = sum(self.data)
        if total == 0:
            p.setBrush(QBrush(QColor("#585b70")))  # Gris neutro visible para que no parezca un hueco
            p.setPen(QPen(QColor("#1e1e2e"), 1))
            p.drawEllipse(self.rect)
            return

        start_angle = 0
        for val, color in zip(self.data, self.colors):
            if val == 0:
                continue
            span_angle = (val / total) * 360
            p.setBrush(QBrush(color))
            p.setPen(QPen(QColor("#1e1e2e"), 1))
            p.drawPie(self.rect, int(start_angle * 16), int(span_angle * 16))
            
            if self.show_text:
                mid_angle = start_angle + span_angle / 2
                rad = math.radians(mid_angle)
                radius = self.rect.width() / 2
                tx = math.cos(rad) * (radius * 0.6) + self.rect.center().x()
                ty = -math.sin(rad) * (radius * 0.6) + self.rect.center().y()
                
                p.save()
                p.translate(tx, ty)
                p.scale(1, -1)
                
                p.setPen(QPen(QColor("#1e1e2e")))
                font = p.font()
                font.setPointSize(max(6, int(radius * 0.25)))
                font.setBold(True)
                p.setFont(font)
                
                text_rect = QRectF(-15, -10, 30, 20)
                p.drawText(text_rect, Qt.AlignmentFlag.AlignCenter, str(val))
                
                p.restore()
            
            start_angle += span_angle

class StackedBarWidget(QWidget):
    def __init__(self, data, colors, parent=None):
        super().__init__(parent)
        self.data = data
        self.colors = [QColor(c) for c in colors]
        self.setFixedHeight(30)
        self.setMinimumWidth(150)
        
    def paintEvent(self, event):
        p = QPainter(self)
        p.setRenderHint(QPainter.RenderHint.Antialiasing)
        total = sum(self.data)
        rect = self.rect()
        
        if total == 0:
            p.setBrush(QColor("#45475a"))
            p.setPen(Qt.PenStyle.NoPen)
            p.drawRect(rect)
            return
            
        current_x = 0
        width = rect.width()
        
        for val, color in zip(self.data, self.colors):
            if val == 0:
                continue
            w = (val / total) * width
            p.setBrush(color)
            p.setPen(Qt.PenStyle.NoPen)
            segment_rect = QRectF(current_x, 0, w, rect.height())
            p.drawRect(segment_rect)
            
            p.setPen(QColor("#1e1e2e"))
            font = p.font()
            font.setPointSize(9)
            font.setBold(True)
            p.setFont(font)
            p.drawText(segment_rect, Qt.AlignmentFlag.AlignCenter, str(val))
            
            current_x += w

class AnalyticsWidget(QWidget):
    """Módulo de Inteligencia y Analítica: Visualización de métricas refactorizado."""

    def __init__(self, db: DatabaseManager, project_id: str, parent=None):
        super().__init__(parent)
        self.db = db
        self.project_id = project_id
        self._build_ui()
        self._populate_filters()
        self.refresh_data()

    def _build_ui(self):
        layout = QVBoxLayout(self)
        layout.setContentsMargins(20, 20, 20, 20)
        layout.setSpacing(15)

        # Header
        header = QHBoxLayout()
        title = QLabel("Dashboard de Analítica")
        title.setStyleSheet("color: #cdd6f4; font-size: 24px; font-weight: bold;")
        header.addWidget(title)
        
        self.btn_train_pending = QPushButton("Entrenar Pendientes")
        self.btn_train_pending.setObjectName("btn_primary")
        self.btn_train_pending.setEnabled(False)
        self.btn_train_pending.clicked.connect(self._on_train_pending_clicked)
        header.addWidget(self.btn_train_pending, alignment=Qt.AlignmentFlag.AlignRight)
        
        self.btn_refresh = QPushButton("↻ Refrescar")
        self.btn_refresh.setObjectName("btn_primary")
        self.btn_refresh.clicked.connect(self.refresh_data)
        header.addWidget(self.btn_refresh, alignment=Qt.AlignmentFlag.AlignRight)
        
        layout.addLayout(header)

        # Filtros
        filters_frame = QFrame()
        filters_frame.setStyleSheet("background: #1e1e2e; border-radius: 8px;")
        filters_layout = QHBoxLayout(filters_frame)
        
        lbl_proj = QLabel("Proyecto:")
        lbl_proj.setStyleSheet("color: #cdd6f4; font-weight: bold;")
        self.combo_project = QComboBox()
        
        lbl_class = QLabel("Clase:")
        lbl_class.setStyleSheet("color: #cdd6f4; font-weight: bold;")
        self.combo_class = QComboBox()
        
        lbl_status = QLabel("Estado:")
        lbl_status.setStyleSheet("color: #cdd6f4; font-weight: bold;")
        self.combo_status = QComboBox()
        
        for w in (lbl_proj, self.combo_project, lbl_class, self.combo_class, lbl_status, self.combo_status):
            filters_layout.addWidget(w)
        filters_layout.addStretch()
        
        layout.addWidget(filters_frame)

        # Dataset Overview (Pies)
        lbl_pies = QLabel("Dataset Overview (Pies)")
        lbl_pies.setStyleSheet("color: #cdd6f4; font-size: 16px; font-weight: bold;")
        lbl_pies.setAlignment(Qt.AlignmentFlag.AlignCenter)
        layout.addWidget(lbl_pies)

        self.pies_scroll = QScrollArea()
        self.pies_scroll.setWidgetResizable(True)
        self.pies_scroll.setFixedHeight(220)
        self.pies_scroll.setStyleSheet("QScrollArea { border: none; background-color: transparent; }")
        self.pies_container = QWidget()
        self.pies_container.setStyleSheet("background-color: transparent;")
        self.pies_layout = QHBoxLayout(self.pies_container)
        self.pies_layout.setAlignment(Qt.AlignmentFlag.AlignLeft)
        self.pies_scroll.setWidget(self.pies_container)
        layout.addWidget(self.pies_scroll)

        # Class Breakdown
        lbl_bd = QLabel("Class Breakdown (Pies & Bars)")
        lbl_bd.setStyleSheet("color: #cdd6f4; font-size: 16px; font-weight: bold;")
        layout.addWidget(lbl_bd)
        
        # Header Row
        header_row = QWidget()
        header_row.setStyleSheet("background-color: #313244; border-radius: 4px;")
        h_layout = QHBoxLayout(header_row)
        h_layout.setContentsMargins(10, 5, 10, 5)
        headers = [
            ("Class Name", 2),
            ("Total\nImages", 1),
            ("Total\nInstances", 1),
            ("Label Status\n(check/X)", 1),
            ("Label Health", 1),
            ("Instance Distribution", 3),
            ("Image Distribution", 3)
        ]
        for text, stretch in headers:
            lbl = QLabel(text)
            lbl.setStyleSheet("color: #bac2de; font-weight: bold; font-size: 12px;")
            lbl.setAlignment(Qt.AlignmentFlag.AlignCenter)
            h_layout.addWidget(lbl, stretch)
        layout.addWidget(header_row)

        self.bd_scroll = QScrollArea()
        self.bd_scroll.setWidgetResizable(True)
        self.bd_scroll.setStyleSheet("QScrollArea { border: none; background-color: transparent; }")
        self.bd_container = QWidget()
        self.bd_container.setStyleSheet("background-color: transparent;")
        self.bd_layout = QVBoxLayout(self.bd_container)
        self.bd_layout.setAlignment(Qt.AlignmentFlag.AlignTop)
        self.bd_layout.setSpacing(5)
        self.bd_scroll.setWidget(self.bd_container)
        layout.addWidget(self.bd_scroll, 1)

        # Overall Stats
        stats_frame = QFrame()
        stats_frame.setStyleSheet("background: #1e1e2e; border-radius: 8px;")
        stats_layout = QHBoxLayout(stats_frame)
        self.lbl_total_images = QLabel("Total Imágenes: 0")
        self.lbl_total_boxes = QLabel("Total Anotaciones: 0")
        for lbl in (self.lbl_total_images, self.lbl_total_boxes):
            lbl.setStyleSheet("color: #89b4fa; font-size: 16px; font-weight: bold;")
            stats_layout.addWidget(lbl)
        layout.addWidget(stats_frame)

    def _populate_filters(self):
        self.combo_project.blockSignals(True)
        self.combo_class.blockSignals(True)
        self.combo_status.blockSignals(True)

        self.combo_project.clear()
        projects = self.db.conn.execute("SELECT id, name FROM projects").fetchall()
        for p_id, p_name in projects:
            self.combo_project.addItem(p_name, p_id)
            if p_id == self.project_id:
                self.combo_project.setCurrentIndex(self.combo_project.count() - 1)

        self.combo_class.clear()
        self.combo_class.addItem("Todos", None)
        classes = self.db.conn.execute("SELECT id, name FROM classes WHERE project_id = ?", (self.project_id,)).fetchall()
        for c_id, c_name in classes:
            self.combo_class.addItem(c_name, c_id)

        self.combo_status.clear()
        self.combo_status.addItems(["Todos", "Pendientes", "Entrenados", "Pruebas", "Validaciones"])
        self.combo_status.setCurrentText("Todos")

        self.combo_project.blockSignals(False)
        self.combo_class.blockSignals(False)
        self.combo_status.blockSignals(False)

        self.combo_project.currentIndexChanged.connect(self._on_filter_changed)
        self.combo_class.currentIndexChanged.connect(self._on_filter_changed)
        self.combo_status.currentIndexChanged.connect(self._on_filter_changed)

    def _on_filter_changed(self):
        new_project_id = self.combo_project.currentData()
        if new_project_id and new_project_id != self.project_id:
            self.project_id = new_project_id
            self.combo_class.blockSignals(True)
            self.combo_class.clear()
            self.combo_class.addItem("Todos", None)
            classes = self.db.conn.execute("SELECT id, name FROM classes WHERE project_id = ?", (self.project_id,)).fetchall()
            for c_id, c_name in classes:
                self.combo_class.addItem(c_name, c_id)
            self.combo_class.blockSignals(False)
            
        estado = self.combo_status.currentText()
        self.btn_train_pending.setEnabled(estado == "Pendientes")
            
        self.refresh_data()

    def refresh_data(self):
        self._update_dashboard()
        self._update_stats()

    def _on_train_pending_clicked(self):
        from PyQt6.QtWidgets import QMessageBox
        reply = QMessageBox.question(
            self,
            "Confirmar entrenamiento",
            "¿Seguro que quieres entrenar los pendientes seleccionados?",
            QMessageBox.StandardButton.Yes | QMessageBox.StandardButton.No
        )
        if reply == QMessageBox.StandardButton.Yes:
            window = self.window()
            if hasattr(window, "train_page"):
                rows = self.db.conn.execute(
                    "SELECT file_path FROM images WHERE project_id = ? AND status = 'annotated'",
                    (self.project_id,)
                ).fetchall()
                paths = [r[0] for r in rows]
                if paths:
                    window._stack.setCurrentWidget(window.train_page)
                    window._nav_group.button(2).setChecked(True)
                    window.start_training_from_paths(paths)
                else:
                    QMessageBox.warning(self, "Aviso", "No hay imágenes pendientes de entrenar en el proyecto.")

    def _update_dashboard(self):
        while self.pies_layout.count():
            item = self.pies_layout.takeAt(0)
            if item.widget():
                item.widget().deleteLater()
                
        while self.bd_layout.count():
            item = self.bd_layout.takeAt(0)
            if item.widget():
                item.widget().deleteLater()

        estado = self.combo_status.currentText()
        estado_filter = ""
        if estado == "Pendientes":
            estado_filter = " AND (i.split IS NULL OR i.split NOT IN ('train', 'test', 'val')) AND i.status = 'annotated'"
        elif estado == "Entrenados":
            estado_filter = " AND i.split = 'train'"
        elif estado == "Pruebas":
            estado_filter = " AND i.split = 'test'"
        elif estado == "Validaciones":
            estado_filter = " AND i.split = 'val'"

        clase_id = self.combo_class.currentData()
        class_filters = ""
        params = [self.project_id, self.project_id]
        if clase_id is not None:
            class_filters = " AND c.id = ?"
            params.append(clase_id)

        query = f"""
            SELECT 
                c.id, c.name,
                COUNT(DISTINCT i.id) as total_images,
                COUNT(a.id) as total_instances,
                SUM(CASE WHEN i.split = 'train' THEN 1 ELSE 0 END) as inst_entrenados,
                SUM(CASE WHEN i.split = 'test' THEN 1 ELSE 0 END) as inst_pruebas,
                SUM(CASE WHEN i.split = 'val' THEN 1 ELSE 0 END) as inst_validaciones,
                SUM(CASE WHEN (i.split IS NULL OR i.split NOT IN ('train', 'test', 'val')) AND i.status = 'annotated' THEN 1 ELSE 0 END) as inst_pendientes,
                COUNT(DISTINCT CASE WHEN i.split = 'train' THEN i.id END) as img_entrenados,
                COUNT(DISTINCT CASE WHEN i.split = 'test' THEN i.id END) as img_pruebas,
                COUNT(DISTINCT CASE WHEN i.split = 'val' THEN i.id END) as img_validaciones,
                COUNT(DISTINCT CASE WHEN (i.split IS NULL OR i.split NOT IN ('train', 'test', 'val')) AND i.status = 'annotated' THEN i.id END) as img_pendientes
            FROM classes c
            LEFT JOIN annotations a ON c.id = a.class_id
            LEFT JOIN images i ON a.image_id = i.id AND i.project_id = ? {estado_filter}
            WHERE c.project_id = ? {class_filters}
            GROUP BY c.id, c.name
            ORDER BY c.name
        """
        
        rows = self.db.conn.execute(query, params).fetchall()
        
        colors = ["#f38ba8", "#f9e2af", "#a6e3a1", "#89b4fa"] # Red, Yellow, Green, Blue
        
        for r in rows:
            c_id, c_name = r[0], r[1]
            t_img, t_inst = r[2], r[3]
            inst_data = [r[7] or 0, r[4] or 0, r[5] or 0, r[6] or 0] # Pendientes, Entrenados, Pruebas, Validaciones
            img_data = [r[11] or 0, r[8] or 0, r[9] or 0, r[10] or 0]
            
            # --- PIE CHART WIDGET ---
            pie_widget = QWidget()
            pie_layout = QVBoxLayout(pie_widget)
            pie_layout.setAlignment(Qt.AlignmentFlag.AlignTop)
            
            lbl_name = QLabel(c_name)
            lbl_name.setAlignment(Qt.AlignmentFlag.AlignCenter)
            lbl_name.setStyleSheet("color: #cdd6f4; font-weight: bold; font-size: 14px;")
            pie_layout.addWidget(lbl_name)
            
            plot = pg.PlotWidget()
            plot.setFixedSize(130, 130)
            plot.setBackground('transparent')
            plot.setMouseEnabled(x=False, y=False)
            plot.hideAxis('left')
            plot.hideAxis('bottom')
            pie_item = PieChartItem(inst_data, colors)
            pie_item.rect = QRectF(-55, -55, 110, 110)
            plot.addItem(pie_item)
            pie_layout.addWidget(plot, alignment=Qt.AlignmentFlag.AlignCenter)
            
            legend_layout = QVBoxLayout()
            legend_layout.setSpacing(2)
            labels = [("Pendientes", "#f38ba8"), ("Entrenados", "#f9e2af"), ("Pruebas", "#a6e3a1"), ("Validación", "#89b4fa")]
            for text, col in labels:
                hl = QHBoxLayout()
                sq = QLabel()
                sq.setFixedSize(10, 10)
                sq.setStyleSheet(f"background-color: {col}; border-radius: 2px;")
                t = QLabel(text)
                t.setStyleSheet("color: #bac2de; font-size: 10px;")
                hl.addWidget(sq)
                hl.addWidget(t)
                hl.addStretch()
                legend_layout.addLayout(hl)
                
            pie_layout.addLayout(legend_layout)
            self.pies_layout.addWidget(pie_widget)
            
            # --- BREAKDOWN ROW ---
            row_widget = QFrame()
            row_widget.setStyleSheet("QFrame { background-color: #1e1e2e; border-bottom: 1px solid #313244; }")
            r_layout = QHBoxLayout(row_widget)
            r_layout.setContentsMargins(10, 10, 10, 10)
            
            c_widget = QWidget()
            c_layout = QHBoxLayout(c_widget)
            c_layout.setContentsMargins(0,0,0,0)
            c_name_lbl = QLabel(c_name)
            c_name_lbl.setStyleSheet("color: #cdd6f4; font-weight: bold;")
            c_layout.addWidget(c_name_lbl)
            
            s_plot = pg.PlotWidget()
            s_plot.setFixedSize(50, 50)
            s_plot.setBackground('transparent')
            s_plot.setMouseEnabled(x=False, y=False)
            s_plot.hideAxis('left')
            s_plot.hideAxis('bottom')
            s_pie = PieChartItem(inst_data, colors, show_text=False)
            s_pie.rect = QRectF(-20, -20, 40, 40)
            s_plot.addItem(s_pie)
            c_layout.addWidget(s_plot)
            r_layout.addWidget(c_widget, 2)
            
            t_img_lbl = QLabel(str(t_img))
            t_img_lbl.setAlignment(Qt.AlignmentFlag.AlignCenter)
            t_img_lbl.setStyleSheet("color: #cdd6f4;")
            r_layout.addWidget(t_img_lbl, 1)
            
            t_inst_lbl = QLabel(str(t_inst))
            t_inst_lbl.setAlignment(Qt.AlignmentFlag.AlignCenter)
            t_inst_lbl.setStyleSheet("color: #cdd6f4;")
            r_layout.addWidget(t_inst_lbl, 1)
            
            status_lbl = QLabel("✅" if t_inst > 0 else "❌")
            status_lbl.setAlignment(Qt.AlignmentFlag.AlignCenter)
            r_layout.addWidget(status_lbl, 1)
            
            health_lbl = QLabel()
            health_lbl.setAlignment(Qt.AlignmentFlag.AlignCenter)
            health_lbl.setFont(QFont("Arial", 10, QFont.Weight.Bold))
            if t_inst < 50:
                health_lbl.setText("CRÍTICO")
                health_lbl.setStyleSheet("color: #f38ba8;")
            elif t_inst <= 200:
                health_lbl.setText("ACEPTABLE")
                health_lbl.setStyleSheet("color: #f9e2af;")
            else:
                health_lbl.setText("ÓPTIMO")
                health_lbl.setStyleSheet("color: #a6e3a1;")
            r_layout.addWidget(health_lbl, 1)
            
            inst_bar = StackedBarWidget(inst_data, colors)
            r_layout.addWidget(inst_bar, 3)
            
            img_bar = StackedBarWidget(img_data, colors)
            r_layout.addWidget(img_bar, 3)
            
            self.bd_layout.addWidget(row_widget)
            
        self.pies_layout.addStretch()

    def _update_stats(self):
        filters_img = ""
        params_img = [self.project_id]
        filters_boxes = ""
        params_boxes = [self.project_id]

        estado = self.combo_status.currentText()
        if estado == "Pendientes":
            filters_img += " AND (i.split IS NULL OR i.split NOT IN ('train', 'test', 'val')) AND i.status = 'annotated'"
            filters_boxes += " AND (i.split IS NULL OR i.split NOT IN ('train', 'test', 'val')) AND i.status = 'annotated'"
        elif estado == "Entrenados":
            filters_img += " AND i.split = 'train'"
            filters_boxes += " AND i.split = 'train'"
        elif estado == "Pruebas":
            filters_img += " AND i.split = 'test'"
            filters_boxes += " AND i.split = 'test'"
        elif estado == "Validaciones":
            filters_img += " AND i.split = 'val'"
            filters_boxes += " AND i.split = 'val'"
            
        clase_id = self.combo_class.currentData()
        if clase_id is not None:
            filters_boxes += " AND a.class_id = ?"
            params_boxes.append(clase_id)
            
        row_boxes = self.db.conn.execute(
            f"""
            SELECT COUNT(a.id) FROM annotations a 
            JOIN images i ON a.image_id = i.id 
            WHERE i.project_id = ? {filters_boxes}
            """,
            params_boxes
        ).fetchone()

        if clase_id is not None:
            query_img = f"""
                SELECT COUNT(DISTINCT i.id) 
                FROM images i
                JOIN annotations a ON a.image_id = i.id
                WHERE i.project_id = ? {filters_boxes}
            """
            row_img = self.db.conn.execute(query_img, params_boxes).fetchone()
        else:
            query_img = f"""
                SELECT COUNT(id) FROM images i
                WHERE project_id = ? {filters_img}
            """
            row_img = self.db.conn.execute(query_img, params_img).fetchone()
            
        self.lbl_total_images.setText(f"Total Imágenes: {row_img[0] if row_img else 0}")
        self.lbl_total_boxes.setText(f"Total Anotaciones: {row_boxes[0] if row_boxes else 0}")

