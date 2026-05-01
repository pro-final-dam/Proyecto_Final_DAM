DARK_THEME = """
/* === Base === */
QWidget {
    background-color: #1e1e2e;
    color: #cdd6f4;
    font-family: "Segoe UI", Arial, sans-serif;
    font-size: 10pt;
}

QMainWindow {
    background-color: #181825;
}

/* === Toolbar top === */
QToolBar {
    background-color: #181825;
    border-bottom: 1px solid #313244;
    padding: 4px 8px;
    spacing: 6px;
}

/* === Buttons === */
QPushButton {
    background-color: #313244;
    color: #cdd6f4;
    border: 1px solid #45475a;
    border-radius: 6px;
    padding: 6px 14px;
    min-width: 90px;
}

QPushButton:hover {
    background-color: #45475a;
    border-color: #89b4fa;
}

QPushButton:pressed {
    background-color: #89b4fa;
    color: #1e1e2e;
}

QPushButton:disabled {
    background-color: #1e1e2e;
    color: #585b70;
    border-color: #313244;
}

QPushButton#btn_primary {
    background-color: #89b4fa;
    color: #1e1e2e;
    font-weight: bold;
    border: none;
}

QPushButton#btn_primary:hover {
    background-color: #b4befe;
}

QPushButton#btn_danger {
    background-color: #f38ba8;
    color: #1e1e2e;
    font-weight: bold;
    border: none;
}

QPushButton#btn_danger:hover {
    background-color: #fab387;
}

/* === Panels laterales === */
QFrame#left_panel, QFrame#right_panel {
    background-color: #181825;
    border: none;
}

QFrame#left_panel {
    border-right: 1px solid #313244;
}

QFrame#right_panel {
    border-left: 1px solid #313244;
}

/* === Labels de sección === */
QLabel#section_title {
    color: #89b4fa;
    font-size: 11px;
    font-weight: bold;
    letter-spacing: 1px;
    padding: 8px 0 4px 0;
}

/* === Canvas central === */
QGraphicsView {
    background-color: #11111b;
    border: none;
}

/* === Lista de clases === */
QListWidget {
    background-color: #1e1e2e;
    border: 1px solid #313244;
    border-radius: 6px;
    padding: 4px;
    outline: none;
}

QListWidget::item {
    padding: 6px 8px;
    border-radius: 4px;
    margin: 1px 0;
}

QListWidget::item:selected {
    background-color: #313244;
    color: #89b4fa;
}

QListWidget::item:hover {
    background-color: #2a2a3c;
}

/* === LineEdit === */
QLineEdit {
    background-color: #313244;
    border: 1px solid #45475a;
    border-radius: 6px;
    padding: 6px 10px;
    color: #cdd6f4;
}

QLineEdit:focus {
    border-color: #89b4fa;
}

/* === Scrollbars === */
QScrollBar:vertical {
    background: #1e1e2e;
    width: 8px;
    border-radius: 4px;
}

QScrollBar::handle:vertical {
    background: #45475a;
    border-radius: 4px;
    min-height: 20px;
}

QScrollBar::handle:vertical:hover {
    background: #89b4fa;
}

QScrollBar:horizontal {
    background: #1e1e2e;
    height: 8px;
    border-radius: 4px;
}

QScrollBar::handle:horizontal {
    background: #45475a;
    border-radius: 4px;
    min-width: 20px;
}

QScrollBar::add-line, QScrollBar::sub-line {
    width: 0;
    height: 0;
}

/* === StatusBar === */
QStatusBar {
    background-color: #181825;
    border-top: 1px solid #313244;
    color: #6c7086;
    font-size: 11px;
}

/* === Separadores === */
QFrame[frameShape="4"], QFrame[frameShape="5"] {
    color: #313244;
}

/* === Tooltips === */
QToolTip {
    background-color: #313244;
    color: #cdd6f4;
    border: 1px solid #45475a;
    border-radius: 4px;
    padding: 4px 8px;
}

/* === ColorBadge en lista === */
QLabel#color_badge {
    border-radius: 4px;
    min-width: 14px;
    max-width: 14px;
    min-height: 14px;
    max-height: 14px;
}
"""

# Paleta de colores para las clases (ciclica)
CLASS_COLORS = [
    "#89b4fa",  # blue
    "#a6e3a1",  # green
    "#f38ba8",  # red
    "#fab387",  # peach
    "#f9e2af",  # yellow
    "#94e2d5",  # teal
    "#cba6f7",  # mauve
    "#89dceb",  # sky
    "#eba0ac",  # maroon
    "#b4befe",  # lavender
]
