import sys
import logging
from pathlib import Path
from PyQt6.QtWidgets import QApplication
from PyQt6.QtGui import QIcon, QFont
from app.main_window import MainWindow
from app.project_selector import ProjectSelectorWindow
from app.styles import DARK_THEME

_LOG_PATH = Path(__file__).parent / "data" / "visionhub.log"
_LOG_PATH.parent.mkdir(exist_ok=True)

_handler = logging.FileHandler(str(_LOG_PATH), encoding="utf-8")
_handler.setFormatter(logging.Formatter(
    "%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
))
logging.getLogger().setLevel(logging.WARNING)
logging.getLogger().addHandler(_handler)


def main():
    app = QApplication(sys.argv)
    app.setApplicationName("VisionHub Desktop")
    app.setOrganizationName("VisionHub")
    app.setFont(QFont("Segoe UI", 10))
    app.setStyleSheet(DARK_THEME)

    selector = ProjectSelectorWindow()

    def open_project(project_id: str, base_path: str, name: str):
        selector.hide()
        window = MainWindow(
            db=selector._db,
            project_id=project_id,
            base_path=base_path,
            project_name=name,
        )

        def back():
            window.back_to_projects.disconnect(back)
            window.close()
            selector._load_projects()
            selector.show()
            selector.raise_()

        window.back_to_projects.connect(back)
        window.show()
        app._window = window   # evita garbage collection

    selector.project_selected.connect(open_project)
    selector.show()

    sys.exit(app.exec())


if __name__ == "__main__":
    main()
