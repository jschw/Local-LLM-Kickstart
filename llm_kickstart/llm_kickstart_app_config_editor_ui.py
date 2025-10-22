from PyQt5.QtWidgets import QWidget, QFormLayout, QLineEdit, QPushButton, QFileDialog, QCheckBox, QHBoxLayout
from PyQt5.QtCore import pyqtSignal
import json
import os

class AppConfigEditor(QWidget):
    config_saved = pyqtSignal()

    def __init__(self, parent=None, app_config_path="app_config.json"):
        super().__init__(parent)
        self.setWindowTitle("Edit App Config")
        self.setGeometry(200, 200, 500, 180)
        self.app_config_path = app_config_path
        self.load_config()
        self.init_ui()

    def load_config(self):
        try:
            with open(self.app_config_path, "r") as f:
                self.config = json.load(f)
        except Exception as e:
            self.config = {
                "llama-server-path": "",
                "use-llama-server-python": "False"
            }
            print(f"Failed to load {self.app_config_path}: {e}")

    def init_ui(self):
        layout = QFormLayout(self)

        # File picker for llama-server-path
        self.server_path_input = QLineEdit(self)
        self.server_path_input.setText(str(self.config.get("llama-server-path", "")))
        self.server_path_input.setToolTip("Path to the llama server executable.")

        self.browse_button = QPushButton("Browse", self)
        self.browse_button.setToolTip("Browse for server executable")
        self.browse_button.clicked.connect(self.browse_server_file)

        server_row_layout = QHBoxLayout()
        server_row_layout.addWidget(self.server_path_input)
        server_row_layout.addWidget(self.browse_button)
        layout.addRow("llama-server-path", server_row_layout)

        # Checkbox for use-llama-server-python
        self.use_python_checkbox = QCheckBox("Activate llama-server-python", self)
        use_python = str(self.config.get("use-llama-server-python", "False")).lower() == "true"
        self.use_python_checkbox.setChecked(use_python)
        self.use_python_checkbox.setToolTip("Check to activate llama-server-python.")
        layout.addRow(self.use_python_checkbox)

        # Buttons
        button_layout = QHBoxLayout()
        self.save_button = QPushButton("Save", self)
        self.save_button.clicked.connect(self.save_config)
        button_layout.addWidget(self.save_button)

        self.close_button = QPushButton("Close", self)
        self.close_button.clicked.connect(self.close)
        button_layout.addWidget(self.close_button)

        layout.addRow(button_layout)

    def browse_server_file(self):
        file_path, _ = QFileDialog.getOpenFileName(self, "Select Server Executable", "", "All Files (*)")
        if file_path:
            self.server_path_input.setText(file_path)

    def save_config(self):
        self.config["llama-server-path"] = self.server_path_input.text()
        self.config["use-llama-server-python"] = "True" if self.use_python_checkbox.isChecked() else "False"
        try:
            with open(self.app_config_path, "w") as f:
                json.dump(self.config, f, indent=4)
            self.config_saved.emit()
            self.close()
        except Exception as e:
            print(f"Failed to save app_config.json: {e}")