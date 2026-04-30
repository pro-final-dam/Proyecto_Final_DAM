import sys
from PyQt6.QtWidgets import QApplication
from PyQt6.QtGui import QIcon
from app.main_window import MainWindow
from app.styles import DARK_THEME


def main():
    app = QApplication(sys.argv)
    app.setApplicationName("VisionHub Desktop")
    app.setOrganizationName("VisionHub")
    app.setStyleSheet(DARK_THEME)

    window = MainWindow()
    window.show()

    sys.exit(app.exec())


if __name__ == "__main__":
    main()
