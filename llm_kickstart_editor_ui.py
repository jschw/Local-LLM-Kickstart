from PyQt5.QtWidgets import QPushButton, QLineEdit, QWidget, QComboBox
from PyQt5.QtWidgets import QFormLayout, QComboBox, QLineEdit, QPushButton, QHBoxLayout
from PyQt5.QtCore import pyqtSignal
import json

class LLMConfigEditor(QWidget):
    config_saved = pyqtSignal()
    def __init__(self, parent=None, config_index=0):
        super().__init__(parent)
        self.setWindowTitle("Edit LLM Configuration")
        self.setGeometry(150, 150, 400, 500)
        self.config_index = config_index
        self.load_config()
        self.init_ui()

    def load_config(self):
        try:
            with open("llm_config.json", "r") as f:
                self.llm_configs = json.load(f)
            if self.config_index < 0 or self.config_index >= len(self.llm_configs):
                self.config_index = 0
            self.config = self.llm_configs[self.config_index]
        except Exception as e:
            self.llm_configs = []
            self.config = {}
            print(f"Failed to load llm_config.json: {e}")

    def init_ui(self):
        
        layout = QFormLayout(self)

        self.inputs = {}

        # Define keys for text inputs
        text_input_keys = [
            "name", "ip", "port", "model", "ctx-size",
            "cache-type-k", "cache-type-v", "n-gpu-layers",
            "lora", "api-key"
        ]

        # Define keys for dropdown inputs with True, False, Default
        dropdown_keys = [
            "flash-attn", "no-kv-offload", "no-mmap", "no-context-shift"
        ]

        # Create inputs for text_input_keys
        for key in text_input_keys:
            value = self.config.get(key, "")
            line_edit = QLineEdit(self)
            line_edit.setText(str(value))
            # Set tooltips for each input
            if key == "name":
                line_edit.setToolTip("Name for this configuration.")
            elif key == "ip":
                line_edit.setToolTip("IP address to listen (default: 127.0.0.1)")
            elif key == "port":
                line_edit.setToolTip("Port to listen (default: 8080)")
            elif key == "model":
                line_edit.setToolTip("Model path (default: models/$filename or models/7B/ggml-model-f16.gguf)")
            elif key == "ctx-size":
                line_edit.setToolTip("Size of the prompt context (default: 4096, 0 = loaded from model)")
            elif key == "cache-type-k":
                line_edit.setToolTip("KV cache data type for K (default: f16; allowed: f32, f16, bf16, q8_0, q4_0, q4_1, iq4_nl, q5_0, q5_1)")
            elif key == "cache-type-v":
                line_edit.setToolTip("KV cache data type for V (default: f16; allowed: f32, f16, bf16, q8_0, q4_0, q4_1, iq4_nl, q5_0, q5_1)")
            elif key == "n-gpu-layers":
                line_edit.setToolTip("Number of layers to store in VRAM")
            elif key == "lora":
                line_edit.setToolTip("Path to LoRA adapter (can be repeated to use multiple adapters)")
            elif key == "api-key":
                line_edit.setToolTip("API key to use for authentication (default: none)")
            layout.addRow(key, line_edit)
            self.inputs[key] = line_edit

        # Create inputs for dropdown_keys
        for key in dropdown_keys:
            value = self.config.get(key, "")
            combo = QComboBox(self)
            combo.addItem("Default")
            combo.addItem("True")
            combo.addItem("False")
            val_lower = str(value).lower()
            if val_lower == "true":
                combo.setCurrentText("True")
            elif val_lower == "false":
                combo.setCurrentText("False")
            else:
                combo.setCurrentText("Default")
            # Set tooltips for each dropdown
            if key == "flash-attn":
                combo.setToolTip("Enable Flash Attention (default: disabled)")
            elif key == "no-kv-offload":
                combo.setToolTip("Disable KV offload")
            elif key == "no-mmap":
                combo.setToolTip("Do not memory-map model (slower load but may reduce pageouts if not using mlock)")
            elif key == "no-context-shift":
                combo.setToolTip("Disables context shift on infinite text generation (default: disabled)")
            layout.addRow(key, combo)
            self.inputs[key] = combo

        # Buttons
        button_layout = QHBoxLayout()
        self.save_button = QPushButton("Save", self)
        self.save_button.clicked.connect(self.save_config)
        button_layout.addWidget(self.save_button)

        self.close_button = QPushButton("Close", self)
        self.close_button.clicked.connect(self.close)
        button_layout.addWidget(self.close_button)

        layout.addRow(button_layout)

    def save_config(self):
        # Update config from inputs
        for key, widget in self.inputs.items():
            if isinstance(widget, QComboBox):
                val = widget.currentText()
                if val == "Default":
                    self.config[key] = ""
                else:
                    self.config[key] = val.lower()
            elif isinstance(widget, QLineEdit):
                self.config[key] = widget.text()

        # Save back to file
        try:
            self.llm_configs[self.config_index] = self.config
            with open("llm_config.json", "w") as f:
                json.dump(self.llm_configs, f, indent=2)
            self.config_saved.emit()
            self.close()
        except Exception as e:
            print(f"Failed to save llm_config.json: {e}")