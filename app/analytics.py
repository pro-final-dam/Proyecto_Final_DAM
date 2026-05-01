import pyqtgraph as pg
from PyQt6.QtWidgets import (
    QWidget, QVBoxLayout, QHBoxLayout, QPushButton, QLabel, QFrame, QGridLayout
)
from PyQt6.QtCore import Qt, pyqtSignal
from PyQt6.QtGui import QColor

from app.database import DatabaseManager

class AnalyticsWidget(QWidget):
    """Módulo de Inteligencia y Analítica: Visualización de métricas."""

    def __init__(self, db: DatabaseManager, project_id: str, parent=None):
        super().__init__(parent)
        self.db = db
        self.project_id = project_id
        self._build_ui()
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
        
        self.btn_refresh = QPushButton("↻ Refrescar")
        self.btn_refresh.setObjectName("btn_primary")
        self.btn_refresh.clicked.connect(self.refresh_data)
        header.addWidget(self.btn_refresh, alignment=Qt.AlignmentFlag.AlignRight)
        
        layout.addLayout(header)

        # Grid for charts
        grid = QGridLayout()
        grid.setSpacing(20)
        
        # 1. Top Classes (Bar Chart)
        self.plot_top_classes = pg.PlotWidget(title="Top 10 Clases")
        self.plot_top_classes.setBackground("#1e1e2e")
        self.plot_top_classes.setTitle("Distribución de Clases (Top 10)", color="#cdd6f4")
        grid.addWidget(self.plot_top_classes, 0, 0)

        # 2. Evolution (Line Chart)
        self.plot_evolution = pg.PlotWidget(title="Evolución Temporal de Etiquetado")
        self.plot_evolution.setBackground("#1e1e2e")
        self.plot_evolution.setTitle("Anotaciones creadas (por día)", color="#cdd6f4")
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

    def refresh_data(self):
        self._update_top_classes()
        self._update_evolution()
        self._update_stats()

    def _update_stats(self):
        # Total images
        row_img = self.db.conn.execute(
            "SELECT COUNT(*) FROM images WHERE project_id = ?",
            (self.project_id,)
        ).fetchone()
        
        # Total boxes
        row_boxes = self.db.conn.execute(
            """
            SELECT COUNT(*) FROM annotations a 
            JOIN images i ON a.image_id = i.id 
            WHERE i.project_id = ?
            """,
            (self.project_id,)
        ).fetchone()
        
        self.lbl_total_images.setText(f"Total Imágenes: {row_img[0] if row_img else 0}")
        self.lbl_total_boxes.setText(f"Total Anotaciones: {row_boxes[0] if row_boxes else 0}")

    def _update_top_classes(self):
        self.plot_top_classes.clear()
        
        rows = self.db.conn.execute(
            """
            SELECT c.name, COUNT(a.id) as count, c.color
            FROM classes c
            LEFT JOIN annotations a ON c.id = a.class_id
            WHERE c.project_id = ?
            GROUP BY c.name, c.color
            ORDER BY count DESC
            LIMIT 10
            """,
            (self.project_id,)
        ).fetchall()

        if not rows:
            return

        names = [r[0] for r in rows]
        counts = [r[1] for r in rows]
        colors = [QColor(r[2]) for r in rows]
        
        x = range(len(names))
        
        # Create BarGraphItem
        bars = pg.BarGraphItem(x=list(x), height=counts, width=0.6, brushes=colors)
        self.plot_top_classes.addItem(bars)
        
        # Set x-axis ticks
        ax = self.plot_top_classes.getAxis('bottom')
        ax.setTicks([list(zip(x, names))])

    def _update_evolution(self):
        self.plot_evolution.clear()
        
        # DuckDB query to group by date
        rows = self.db.conn.execute(
            """
            SELECT CAST(a.created_at AS DATE) as date, COUNT(a.id) as count
            FROM annotations a
            JOIN images i ON a.image_id = i.id
            WHERE i.project_id = ?
            GROUP BY date
            ORDER BY date ASC
            """,
            (self.project_id,)
        ).fetchall()

        if not rows:
            return

        dates_str = [str(r[0]) for r in rows]
        counts = [r[1] for r in rows]
        x = range(len(dates_str))
        
        pen = pg.mkPen(color='#a6e3a1', width=3)
        self.plot_evolution.plot(list(x), counts, pen=pen, symbol='o', symbolBrush='#89b4fa')
        
        ax = self.plot_evolution.getAxis('bottom')
        ax.setTicks([list(zip(x, dates_str))])
