import os
import json
import torch
from safetensors.torch import load_file
from gguf import GGUFWriter


def create_lora_gguf(model_dir, output_path):

    config_path = os.path.join(model_dir, "adapter_config.json")
    if not os.path.exists(config_path):
        print(f"❌ Error: {config_path} not found!")
        return

    with open(config_path, "r") as f:
        config = json.load(f)

    weights_path = os.path.join(model_dir, "adapter_model.safetensors")
    if not os.path.exists(weights_path):
        print(f"❌ Error: {weights_path} not found!")
        return

    print(f"📂 Loading weights from {weights_path}...")
    tensors = load_file(weights_path)

    writer = GGUFWriter(output_path, "qwen2")


    writer.add_string("general.architecture", "qwen2")
    writer.add_string("general.type", "adapter")
    writer.add_uint32("adapter.type", 1)  # LoRA type
    writer.add_float32("adapter.lora.alpha", config.get("lora_alpha", 16.0))


    print(f"⚙️  Converting {len(tensors)} tensors...")
    for name, data in tensors.items():
        # GGUF expects Float32 or Float16 for adapters.
        data_np = data.to(torch.float32).numpy()
        writer.add_tensor(name, data_np)


    writer.write_header_to_file()
    writer.write_kv_data_to_file()
    writer.write_tensors_to_file()
    writer.close()

    print(f"✅ SUCCESS! GGUF created at: {output_path}")


if __name__ == "__main__":
    BASE_DIR = os.path.dirname(os.path.abspath(__file__))

    INPUT_FOLDER = os.path.join(BASE_DIR, "LoRa", "lora_weights")
    OUTPUT_FILE = os.path.join(BASE_DIR, "LoRa", "qwen-tun-adapter.gguf")

    create_lora_gguf(INPUT_FOLDER, OUTPUT_FILE)