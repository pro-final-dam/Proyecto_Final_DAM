import pyqtgraph as pg
from PyQt6.QtWidgets import (
    QWidget, QVBoxLayout, QHBoxLayout, QPushButton, QLabel, QFrame, QGridLayout, QComboBox
)
from PyQt6.QtCore import Qt
from PyQt6.QtGui import QColor

from app.database import DatabaseManager

class AnalyticsWidget(QWidget):
    """Módulo de Inteligencia y Analítica: Visualización de métricas."""

    def __init__(self, db: DatabaseManager, project_id: str, parent=None):
        super().__init__(parent)
        self.db = db
        self.project_id = project_id
        self.bar_item = None
        self._build_ui()
        self._populate_filters()
        self.refresh_data()

    def _build_ui(self):
        layout = QVBoxLayout(self)
        layout.setContentsMargins(20, 20, 20, 20)
        layout.setSpacing(20)

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

        # Grid for charts
        grid = QGridLayout()
        grid.setSpacing(20)
        
        # 1. Top Classes (Bar Chart)
        self.plot_top_classes = pg.PlotWidget(title="Distribución de clases")
        self.plot_top_classes.setBackground("#1e1e2e")
        self.plot_top_classes.setTitle("Distribución de clases", color="#cdd6f4")
        self.plot_top_classes.setYRange(0, 300)
        self._draw_health_regions()
        
        grid.addWidget(self.plot_top_classes, 0, 0)

        # 2. Evolution (Line Chart)
        self.plot_evolution = pg.PlotWidget(title="Evolución por fechas")
        self.plot_evolution.setBackground("#1e1e2e")
        self.plot_evolution.setTitle("Evolución por fechas", color="#cdd6f4")
        grid.addWidget(self.plot_evolution, 0, 1)
        
        # 3. Overall Stats
        stats_frame = QFrame()
        stats_frame.setStyleSheet("background: #1e1e2e; border-radius: 8px;")
        stats_layout = QVBoxLayout(stats_frame)
        self.lbl_total_images = QLabel("Total Imágenes: 0")
        self.lbl_total_boxes = QLabel("Total Anotaciones: 0")
        for lbl in (self.lbl_total_images, self.lbl_total_boxes):
            lbl.setStyleSheet("color: #89b4fa; font-size: 18px; font-weight: bold;")
            stats_layout.addWidget(lbl)
        grid.addWidget(stats_frame, 1, 0, 1, 2)

        layout.addLayout(grid)
        layout.setStretchFactor(grid, 1)

    def _populate_filters(self):
        # Desactivar señales temporalmente
        self.combo_project.blockSignals(True)
        self.combo_class.blockSignals(True)
        self.combo_status.blockSignals(True)

        # Proyectos
        self.combo_project.clear()
        projects = self.db.conn.execute("SELECT id, name FROM projects").fetchall()
        for p_id, p_name in projects:
            self.combo_project.addItem(p_name, p_id)
            if p_id == self.project_id:
                self.combo_project.setCurrentIndex(self.combo_project.count() - 1)

        # Clases
        self.combo_class.clear()
        self.combo_class.addItem("Todos", None)
        classes = self.db.conn.execute("SELECT id, name FROM classes WHERE project_id = ?", (self.project_id,)).fetchall()
        for c_id, c_name in classes:
            self.combo_class.addItem(c_name, c_id)

        # Estado
        self.combo_status.clear()
        self.combo_status.addItems(["Todos", "Pendientes", "Entrenados", "Pruebas", "Validaciones"])
        self.combo_status.setCurrentText("Pendientes")

        # Reactivar señales y conectar a la función central
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
        self._update_top_classes()
        self._update_evolution()
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

    def _update_stats(self):
        filters_img = ""
        params_img = [self.project_id]
        filters_boxes = ""
        params_boxes = [self.project_id]

        estado = self.combo_status.currentText()
        if estado == "Pendientes":
            filters_img += " AND status = 'annotated'"
            filters_boxes += " AND i.status = 'annotated'"
        elif estado == "Entrenados":
            filters_img += " AND split = 'train'"
            filters_boxes += " AND i.split = 'train'"
        elif estado == "Pruebas":
            filters_img += " AND split = 'test'"
            filters_boxes += " AND i.split = 'test'"
        elif estado == "Validaciones":
            filters_img += " AND split = 'val'"
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
                SELECT COUNT(id) FROM images 
                WHERE project_id = ? {filters_img}
            """
            row_img = self.db.conn.execute(query_img, params_img).fetchone()
            
        self.lbl_total_images.setText(f"Total Imágenes: {row_img[0] if row_img else 0}")
        self.lbl_total_boxes.setText(f"Total Anotaciones: {row_boxes[0] if row_boxes else 0}")

    def _draw_health_regions(self):
        self.plot_top_classes.setYRange(0, 300)
        def add_region(y_min, y_max, color):
            region = pg.LinearRegionItem([y_min, y_max], orientation='horizontal', movable=False)
            region.setBrush(pg.mkBrush(color))
            region.setHoverBrush(pg.mkBrush(color))
            for line in region.lines:
                line.setPen(pg.mkPen(None))
                line.setHoverPen(pg.mkPen(None))
            region.setZValue(-10)
            self.plot_top_classes.addItem(region)
            
        add_region(0, 50, (255, 0, 0, 30))      # Roja
        add_region(50, 100, (255, 255, 0, 30))  # Amarilla
        add_region(100, 200, (0, 255, 0, 30))   # Verde
        add_region(200, 300, (0, 0, 255, 30))   # Azul
        
        line50 = pg.InfiniteLine(pos=50, angle=0, pen=pg.mkPen('y', style=Qt.PenStyle.DashLine))
        self.plot_top_classes.addItem(line50)
        line200 = pg.InfiniteLine(pos=200, angle=0, pen=pg.mkPen('g', style=Qt.PenStyle.DashLine))
        self.plot_top_classes.addItem(line200)

    def _update_top_classes(self):
        self.plot_top_classes.clear()
        self._draw_health_regions()
        if hasattr(self, 'bar_item') and self.bar_item is not None:
            self.bar_item = None
            
        estado = self.combo_status.currentText()
        image_filters = ""
        
        if estado == "Pendientes":
            image_filters += " AND i.status = 'annotated'"
        elif estado == "Entrenados":
            image_filters += " AND i.split = 'train'"
        elif estado == "Pruebas":
            image_filters += " AND i.split = 'test'"
        elif estado == "Validaciones":
            image_filters += " AND i.split = 'val'"

        class_filters = ""
        clase_id = self.combo_class.currentData()
        if clase_id is not None:
            class_filters += " AND c.id = ?"

        opacity_map = {
            "Pendientes": 255,
            "Entrenados": 200,
            "Pruebas": 140,
            "Validaciones": 80
        }

        names = []
        counts = []
        colors = []

        if estado == "Todos":
            query = f"""
                SELECT c.name,
                       COALESCE(
                           CASE 
                               WHEN i.split = 'train' THEN 'Entrenados'
                               WHEN i.split = 'test' THEN 'Pruebas'
                               WHEN i.split = 'val' THEN 'Validaciones'
                               WHEN i.status = 'annotated' THEN 'Pendientes'
                           END, 'Otros'
                       ) as estado_label,
                       COUNT(a.id) as count, 
                       c.color
                FROM classes c
                LEFT JOIN annotations a ON c.id = a.class_id
                LEFT JOIN images i ON a.image_id = i.id AND i.project_id = ?
                WHERE c.project_id = ? {class_filters}
                GROUP BY c.name, estado_label, c.color
            """
            params = [self.project_id, self.project_id]
            if clase_id is not None:
                params.append(clase_id)
                
            rows = self.db.conn.execute(query, params).fetchall()
            if not rows:
                ax = self.plot_top_classes.getAxis('bottom')
                ax.setTicks([])
                return
                
            classes_data = {}
            for r in rows:
                c_name, st_label, count, c_color = r[0], r[1], r[2], r[3]
                if c_name not in classes_data:
                    classes_data[c_name] = {'color': c_color, 'counts': {}}
                if st_label != 'Otros':
                    classes_data[c_name]['counts'][st_label] = classes_data[c_name]['counts'].get(st_label, 0) + count
                    
            ordered_states = ["Pendientes", "Entrenados", "Pruebas", "Validaciones"]
            
            for c_name in sorted(classes_data.keys()):
                c_color = classes_data[c_name]['color']
                for st in ordered_states:
                    names.append(f"{c_name} - {st}")
                    counts.append(classes_data[c_name]['counts'].get(st, 0))
                    
                    c = QColor(c_color)
                    if st in opacity_map:
                        c.setAlpha(opacity_map[st])
                    colors.append(pg.mkBrush(c))
        else:
            query = f"""
                SELECT c.name, COUNT(a.id) as count, c.color, '{estado}' as estado_label
                FROM classes c
                LEFT JOIN (
                    SELECT a.id, a.class_id
                    FROM annotations a
                    JOIN images i ON a.image_id = i.id
                    WHERE i.project_id = ? {image_filters}
                ) a ON c.id = a.class_id
                WHERE c.project_id = ? {class_filters}
                GROUP BY c.name, c.color
                ORDER BY count DESC
                LIMIT 10
            """
            params = [self.project_id, self.project_id]
            if clase_id is not None:
                params.append(clase_id)
        
            rows = self.db.conn.execute(query, params).fetchall()

            if not rows:
                ax = self.plot_top_classes.getAxis('bottom')
                ax.setTicks([])
                return

            for r in rows:
                c_name = r[0]
                count = r[1]
                color_hex = r[2]
                names.append(c_name)
                counts.append(count)
                colors.append(pg.mkBrush(QColor(color_hex)))
        
        x = range(len(names))
        
        self.bar_item = pg.BarGraphItem(x=list(x), height=counts, width=0.6, brushes=colors)
        self.plot_top_classes.addItem(self.bar_item)
        
        ax = self.plot_top_classes.getAxis('bottom')
        ax.setTicks([list(zip(x, names))])
        
        from PyQt6.QtGui import QFont
        font = QFont("Arial", 8)
        if len(names) > 5:
            font.setPointSize(7)
        ax.setTickFont(font)

    def _update_evolution(self):
        self.plot_evolution.clear()
        
        # 1. Obtener TODAS las fechas disponibles en el proyecto
        dates_query = """
            SELECT DISTINCT CAST(a.created_at AS DATE) as date
            FROM annotations a
            JOIN images i ON a.image_id = i.id
            WHERE i.project_id = ?
            ORDER BY date ASC
        """
        all_dates_rows = self.db.conn.execute(dates_query, [self.project_id]).fetchall()
        
        if not all_dates_rows:
            ax = self.plot_evolution.getAxis('bottom')
            ax.setTicks([])
            return
            
        all_dates = [str(r[0]) for r in all_dates_rows]
        
        # 2. Obtener los conteos aplicando los filtros seleccionados
        filters = ""
        params = [self.project_id]

        estado = self.combo_status.currentText()
        if estado == "Pendientes":
            filters += " AND i.status = 'annotated'"
        elif estado == "Entrenados":
            filters += " AND i.split = 'train'"
        elif estado == "Pruebas":
            filters += " AND i.split = 'test'"
        elif estado == "Validaciones":
            filters += " AND i.split = 'val'"

        clase_id = self.combo_class.currentData()
        if clase_id is not None:
            filters += " AND a.class_id = ?"
            params.append(clase_id)
            
        query = f"""
            SELECT CAST(a.created_at AS DATE) as date, COUNT(a.id) as count
            FROM annotations a
            JOIN images i ON a.image_id = i.id
            WHERE i.project_id = ? {filters}
            GROUP BY date
        """
        
        rows = self.db.conn.execute(query, params).fetchall()
        counts_dict = {str(r[0]): r[1] for r in rows}
        
        # 3. Rellenar con ceros las fechas sin registros
        counts = [counts_dict.get(d, 0) for d in all_dates]
        x = range(len(all_dates))
        
        pen = pg.mkPen(color='#a6e3a1', width=3)
        self.plot_evolution.plot(list(x), counts, pen=pen, symbol='o', symbolBrush='#89b4fa')
        
        ax = self.plot_evolution.getAxis('bottom')
        ax.setTicks([list(zip(x, all_dates))])
        
        from PyQt6.QtGui import QFont
        font = QFont("Arial", 8)
        if len(all_dates) > 5:
            font.setPointSize(7)
        ax.setTickFont(font)
