import os
from llama_cpp.llama_cpp import convert_model_to_gguf


def convert_folder_to_gguf(model_dir, output_filename, quant_type="q4_k_m"):
    """
    Standalone script to convert a folder of HF weights into a single GGUF file.
    """
    if not os.path.exists(model_dir):
        print(f"Error: Directory {model_dir} not found.")
        return

    print(f"🚀 Starting conversion for: {model_dir}")

    convert_model_to_gguf(
        model_path=model_dir,
        out_path=output_filename,
        quantization=quant_type
    )

    print(f"✅ Success! GGUF saved to: {output_filename}")


if __name__ == "__main__":
    INPUT_DIR = "src/optimization/LoRa/fine_tunning_model"
    OUTPUT_FILE = "src/optimization/LoRa/qwen-tun.gguf"

    convert_folder_to_gguf(INPUT_DIR, OUTPUT_FILE)