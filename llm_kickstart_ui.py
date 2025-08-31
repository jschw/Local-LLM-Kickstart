import sys
from PyQt5.QtWidgets import QApplication, QMainWindow, QVBoxLayout, QPushButton, QWidget, QListWidget, QComboBox, QMessageBox
from PyQt5.QtWidgets import QComboBox, QPushButton

from llm_kickstart import LLMKickstart
from llm_kickstart_editor_ui import LLMConfigEditor
import json

class LLMKickstartUi(QMainWindow):
    def __init__(self):
        super().__init__()
        self.manager = LLMKickstart()
        self.init_ui()

    def init_ui(self):
        self.setWindowTitle("LLM Kickstart UI")
        self.setGeometry(100, 100, 400, 350)

        # Main layout
        central_widget = QWidget()
        self.setCentralWidget(central_widget)
        layout = QVBoxLayout(central_widget)

        # Dropdown for LLM configurations
        self.llm_dropdown = QComboBox(self)
        self.llm_dropdown.setToolTip("Select LLM Configuration")
        layout.addWidget(self.llm_dropdown)

        # Buttons
        self.create_button = QPushButton("Start Endpoint", self)
        self.create_button.clicked.connect(self.create_process)
        layout.addWidget(self.create_button)

        self.stop_button = QPushButton("Stop Endpoint", self)
        self.stop_button.clicked.connect(self.stop_process)
        layout.addWidget(self.stop_button)

        self.stop_all_button = QPushButton("Stop All Endpoints", self)
        self.stop_all_button.clicked.connect(self.stop_all_processes)
        layout.addWidget(self.stop_all_button)

        self.restart_button = QPushButton("Restart Endpoint", self)
        self.restart_button.clicked.connect(self.restart_process)
        layout.addWidget(self.restart_button)

        edit_button = QPushButton("Edit LLM Config", self)
        edit_button.clicked.connect(self.open_llm_config_editor)
        self.centralWidget().layout().addWidget(edit_button)

        # Process list
        self.process_list = QListWidget(self)
        layout.addWidget(self.process_list)

        # Refresh button
        self.refresh_button = QPushButton("Refresh Endpoint List", self)
        self.refresh_button.clicked.connect(self.refresh_process_list)
        layout.addWidget(self.refresh_button)

        # Custom Close button
        self.close_button = QPushButton("Close", self)
        self.close_button.clicked.connect(self.handle_close_button)
        layout.addWidget(self.close_button)

    def create_process(self):
        name = self.llm_dropdown.currentText()
        if name:
            self.manager.create_endpoint(name)
            self.refresh_process_list()
        else:
            self.show_message("Please select a valid LLM endpoint configuration.")

    def load_llm_configurations(self):
        try:
            with open("llm_config.json", "r") as f:
                self.llm_configs = json.load(f)
            self.llm_dropdown.clear()
            for config in self.llm_configs:
                self.llm_dropdown.addItem(config.get("name", "Unnamed LLM"))
        except Exception as e:
            self.show_message(f"Failed to load LLM endpoint configurations: {e}")


    # Add method to LLMKickstartUi to open the editor
    def open_llm_config_editor(self):
        index = self.llm_dropdown.currentIndex()
        if index < 0:
            self.show_message("Please select a valid LLM configuration to edit.")
            return
        self.editor = LLMConfigEditor(None, config_index=index)
        self.editor.show()
        self.editor.destroyed.connect(self.load_llm_configurations)
        self.editor.config_saved.connect(self.load_llm_configurations)
    
    def stop_process(self):
        name = self.llm_dropdown.currentText()
        if name:
            self.manager.stop_process(name)
            self.refresh_process_list()
        else:
            self.show_message("Please select an endpoint to stop.")

    def stop_all_processes(self):
        self.manager.stop_all_processes()
        self.refresh_process_list()
    
    def restart_process(self):
        name = self.llm_dropdown.currentText()
        self.manager.restart_process(name)
        self.refresh_process_list()

    def refresh_process_list(self):
        self.process_list.clear()
        for name, process in self.manager.processes.items():
            status = "running" if process.is_alive() else "stopped"
            self.process_list.addItem(f"{name}: PID {process.pid}, Status: {status}")

    def show_message(self, message):
        self.process_list.addItem(f"Error: {message}")


    def closeEvent(self, event):
        # Override OS window bar close button: ignore and do nothing
        # Only allow closing via the custom close button
        event.ignore()
        # Optionally, you could show a message here if desired

    def handle_close_button(self):
        # Show confirmation dialog
        running_endpoints = any(
            process.is_alive() for process in getattr(self.manager, "processes", {}).values()
        )
        msg = "Do you really want to close the application?"
        if running_endpoints:
            msg += "\n\nThis will stop any running inference endpoint."
        reply = QMessageBox.question(
            self,
            "Confirm Close",
            msg,
            QMessageBox.Yes | QMessageBox.No,
            QMessageBox.No
        )
        if reply == QMessageBox.Yes:
            QApplication.instance().quit()
        # else: do nothing

if __name__ == "__main__":
    app = QApplication(sys.argv)
    window = LLMKickstartUi()
    window.load_llm_configurations()
    window.show()
    sys.exit(app.exec_())