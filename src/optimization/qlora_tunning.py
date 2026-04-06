import os
import torch
import logging
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    BitsAndBytesConfig,
    TrainingArguments
)
from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training, PeftModel
from datasets import load_dataset
from trl import SFTTrainer

# Windows DLL Fixes
os.environ["BNB_CUDA_VERSION"] = "121"
os.environ["CUDA_VISIBLE_DEVICES"] = "0"

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def train_and_save():
    model_id = "Qwen/Qwen2.5-1.5B-Instruct"
    output_dir = "src/optimization/LoRa"

    # 1. Link CUDA DLLs
    torch_lib_path = os.path.join(os.path.dirname(torch.__file__), "lib")
    if os.path.exists(torch_lib_path):
        os.add_dll_directory(torch_lib_path)

    # 2. Quantization (4-bit is mandatory for 4GB VRAM)
    bnb_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_compute_dtype=torch.float16,
        bnb_4bit_use_double_quant=True,
    )

    tokenizer = AutoTokenizer.from_pretrained(model_id)
    tokenizer.pad_token = tokenizer.eos_token

    model = AutoModelForCausalLM.from_pretrained(
        model_id,
        quantization_config=bnb_config,
        device_map={"": 0},
        trust_remote_code=True
    )

    model.gradient_checkpointing_enable()
    model = prepare_model_for_kbit_training(model)

    # 3. LoRA Setup
    lora_config = LoraConfig(
        r=16,  # Higher rank for better learning
        lora_alpha=32,
        target_modules=["q_proj", "v_proj", "k_proj", "o_proj"],
        lora_dropout=0.1,
        bias="none",
        task_type="CAUSAL_LM"
    )
    model = get_peft_model(model, lora_config)

    # 4. Dataset
    raw_dataset = load_dataset("json", data_files="data_processing/lora_train_data.jsonl", split="train")

    # 2. Split it into 90% train and 10% test (validation)
    dataset_dict = raw_dataset.train_test_split(test_size=0.1,shuffle=True)
    train_dataset = dataset_dict["train"]
    eval_dataset = dataset_dict["test"]

    def formatting_prompts_func(example):
        return {"text": f"User: {example['instruction']}\nAssistant: {example['response']}{tokenizer.eos_token}"}

    train_dataset = train_dataset.map(formatting_prompts_func)
    eval_dataset = eval_dataset.map(formatting_prompts_func)

    # 5. Batched Training Arguments
    training_args = TrainingArguments(
        output_dir=output_dir,
        per_device_train_batch_size=2,  # Smaller physical batch to save VRAM
        gradient_accumulation_steps=4,  # Total batch 8 (better for small data)
        learning_rate=1e-4,  # Increase slightly to learn faster
        weight_decay=0.01,
        lr_scheduler_type="cosine",  # Don't let the rate drop for small datasets
        num_train_epochs=12,  # Train for MORE epochs (at least 10-20)
        warmup_ratio=0.1,
        fp16=True,
        optim="paged_adamw_8bit",
        logging_steps=1,
        evaluation_strategy="epoch",
        eval_steps=1,
        save_strategy="epoch",
        load_best_model_at_end=True,  # Keep the version with lowest eval_loss
        report_to="none"
    )

    trainer = SFTTrainer(
        model=model,
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
        dataset_text_field="text",
        max_seq_length=128,
        args=training_args,
    )
    logger.info("🚀 Training with Batch Size...")
    trainer.train()

    # --- SAVING SECTION ---

    adapter_path = os.path.join(output_dir, "lora_weights")
    model.save_pretrained(adapter_path)
    tokenizer.save_pretrained(adapter_path)
    logger.info(f"✅ LoRA Weights saved to: {adapter_path}")

    logger.info("Merging weights... This requires ~8GB-12GB System RAM.")

    # Free up GPU VRAM before merging to avoid crashes
    del model
    torch.cuda.empty_cache()

    # Load base model in FP16 (NOT 4-bit) on CPU
    base_model_cpu = AutoModelForCausalLM.from_pretrained(
        model_id,
        torch_dtype=torch.float16,  # Standard precision for merging
        low_cpu_mem_usage=True,
        device_map="cpu",
        trust_remote_code=True
    )

    # Load adapters and merge
    merged_model = PeftModel.from_pretrained(base_model_cpu, adapter_path)
    merged_model = merged_model.merge_and_unload()

    # 3. Save as ONE set of weights (SafeTensors)
    final_model_path = os.path.join(output_dir, "fine_tunning_model")

    merged_model.save_pretrained(
        final_model_path,
        safe_serialization=True,
        max_shard_size="10GB"
    )
    tokenizer.save_pretrained(final_model_path)

    logger.info(f"✅ Merged Model saved correctly to: {final_model_path}")
if __name__ == "__main__":
    train_and_save()