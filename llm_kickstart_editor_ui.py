from PyQt5.QtWidgets import QPushButton, QLineEdit, QWidget, QComboBox, QFileDialog
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

        # Individual text inputs
        self.name_input = QLineEdit(self)
        self.name_input.setText(str(self.config.get("name", "")))
        self.name_input.setToolTip("Name for this configuration.")
        layout.addRow("name", self.name_input)

        self.ip_input = QLineEdit(self)
        self.ip_input.setText(str(self.config.get("ip", "")))
        self.ip_input.setToolTip("IP address to listen (default: 127.0.0.1)")
        layout.addRow("ip", self.ip_input)

        self.port_input = QLineEdit(self)
        self.port_input.setText(str(self.config.get("port", "")))
        self.port_input.setToolTip("Port to listen (default: 8080)")
        layout.addRow("port", self.port_input)

        from PyQt5.QtWidgets import QHBoxLayout

        self.model_input = QLineEdit(self)
        self.model_input.setText(str(self.config.get("model", "")))
        self.model_input.setToolTip("Model path (default: models/$filename or models/7B/ggml-model-f16.gguf)")

        self.model_browse_button = QPushButton("Browse", self)
        self.model_browse_button.setToolTip("Browse for model file")
        self.model_browse_button.clicked.connect(self.browse_model_file)

        model_row_layout = QHBoxLayout()
        model_row_layout.addWidget(self.model_input)
        model_row_layout.addWidget(self.model_browse_button)
        layout.addRow("model", model_row_layout)

        self.ctx_size_input = QLineEdit(self)
        self.ctx_size_input.setText(str(self.config.get("ctx-size", "")))
        self.ctx_size_input.setToolTip("Size of the prompt context (default: 4096, 0 = loaded from model)")
        layout.addRow("ctx-size", self.ctx_size_input)

        self.cache_type_k_input = QLineEdit(self)
        self.cache_type_k_input.setText(str(self.config.get("cache-type-k", "")))
        self.cache_type_k_input.setToolTip("KV cache data type for K (default: f16; allowed: f32, f16, bf16, q8_0, q4_0, q4_1, iq4_nl, q5_0, q5_1)")
        layout.addRow("cache-type-k", self.cache_type_k_input)

        self.cache_type_v_input = QLineEdit(self)
        self.cache_type_v_input.setText(str(self.config.get("cache-type-v", "")))
        self.cache_type_v_input.setToolTip("KV cache data type for V (default: f16; allowed: f32, f16, bf16, q8_0, q4_0, q4_1, iq4_nl, q5_0, q5_1)")
        layout.addRow("cache-type-v", self.cache_type_v_input)

        self.n_gpu_layers_input = QLineEdit(self)
        self.n_gpu_layers_input.setText(str(self.config.get("n-gpu-layers", "")))
        self.n_gpu_layers_input.setToolTip("Number of layers to store in VRAM")
        layout.addRow("n-gpu-layers", self.n_gpu_layers_input)

        self.lora_input = QLineEdit(self)
        self.lora_input.setText(str(self.config.get("lora", "")))
        self.lora_input.setToolTip("Path to LoRA adapter (can be repeated to use multiple adapters)")
        layout.addRow("lora", self.lora_input)

        self.api_key_input = QLineEdit(self)
        self.api_key_input.setText(str(self.config.get("api-key", "")))
        self.api_key_input.setToolTip("API key to use for authentication (default: none)")
        layout.addRow("api-key", self.api_key_input)

        # Individual dropdowns
        self.flash_attn_combo = QComboBox(self)
        self.flash_attn_combo.addItem("Default")
        self.flash_attn_combo.addItem("True")
        self.flash_attn_combo.addItem("False")
        val = str(self.config.get("flash-attn", "")).lower()
        if val == "true":
            self.flash_attn_combo.setCurrentText("True")
        elif val == "false":
            self.flash_attn_combo.setCurrentText("False")
        else:
            self.flash_attn_combo.setCurrentText("Default")
        self.flash_attn_combo.setToolTip("Enable Flash Attention (default: disabled)")
        layout.addRow("flash-attn", self.flash_attn_combo)

        self.no_kv_offload_combo = QComboBox(self)
        self.no_kv_offload_combo.addItem("Default")
        self.no_kv_offload_combo.addItem("True")
        self.no_kv_offload_combo.addItem("False")
        val = str(self.config.get("no-kv-offload", "")).lower()
        if val == "true":
            self.no_kv_offload_combo.setCurrentText("True")
        elif val == "false":
            self.no_kv_offload_combo.setCurrentText("False")
        else:
            self.no_kv_offload_combo.setCurrentText("Default")
        self.no_kv_offload_combo.setToolTip("Disable KV offload")
        layout.addRow("no-kv-offload", self.no_kv_offload_combo)

        self.no_mmap_combo = QComboBox(self)
        self.no_mmap_combo.addItem("Default")
        self.no_mmap_combo.addItem("True")
        self.no_mmap_combo.addItem("False")
        val = str(self.config.get("no-mmap", "")).lower()
        if val == "true":
            self.no_mmap_combo.setCurrentText("True")
        elif val == "false":
            self.no_mmap_combo.setCurrentText("False")
        else:
            self.no_mmap_combo.setCurrentText("Default")
        self.no_mmap_combo.setToolTip("Do not memory-map model (slower load but may reduce pageouts if not using mlock)")
        layout.addRow("no-mmap", self.no_mmap_combo)

        self.no_context_shift_combo = QComboBox(self)
        self.no_context_shift_combo.addItem("Default")
        self.no_context_shift_combo.addItem("True")
        self.no_context_shift_combo.addItem("False")
        val = str(self.config.get("no-context-shift", "")).lower()
        if val == "true":
            self.no_context_shift_combo.setCurrentText("True")
        elif val == "false":
            self.no_context_shift_combo.setCurrentText("False")
        else:
            self.no_context_shift_combo.setCurrentText("Default")
        self.no_context_shift_combo.setToolTip("Disables context shift on infinite text generation (default: disabled)")
        layout.addRow("no-context-shift", self.no_context_shift_combo)

        # Buttons
        button_layout = QHBoxLayout()
        self.save_button = QPushButton("Save", self)
        self.save_button.clicked.connect(self.save_config)
        button_layout.addWidget(self.save_button)

        self.close_button = QPushButton("Close", self)
        self.close_button.clicked.connect(self.close)
        button_layout.addWidget(self.close_button)

        layout.addRow(button_layout)

    def browse_model_file(self):
        file_path, _ = QFileDialog.getOpenFileName(self, "Select Model File", "", "All Files (*)")
        if file_path:
            self.model_input.setText(file_path)

    def save_config(self):
        # Update config from inputs
        # Save each input explicitly
        self.config["name"] = self.name_input.text()
        self.config["ip"] = self.ip_input.text()
        self.config["port"] = self.port_input.text()
        self.config["model"] = self.model_input.text()
        self.config["ctx-size"] = self.ctx_size_input.text()
        self.config["cache-type-k"] = self.cache_type_k_input.text()
        self.config["cache-type-v"] = self.cache_type_v_input.text()
        self.config["n-gpu-layers"] = self.n_gpu_layers_input.text()
        self.config["lora"] = self.lora_input.text()
        self.config["api-key"] = self.api_key_input.text()

        # Save dropdowns
        val = self.flash_attn_combo.currentText()
        self.config["flash-attn"] = "" if val == "Default" else val.lower()
        val = self.no_kv_offload_combo.currentText()
        self.config["no-kv-offload"] = "" if val == "Default" else val.lower()
        val = self.no_mmap_combo.currentText()
        self.config["no-mmap"] = "" if val == "Default" else val.lower()
        val = self.no_context_shift_combo.currentText()
        self.config["no-context-shift"] = "" if val == "Default" else val.lower()

        # Save back to file
        try:
            self.llm_configs[self.config_index] = self.config
            with open("llm_config.json", "w") as f:
                json.dump(self.llm_configs, f, indent=2)
            self.config_saved.emit()
            self.close()
        except Exception as e:
            print(f"Failed to save llm_config.json: {e}")