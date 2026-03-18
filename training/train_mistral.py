import torch
import sys
sys.path.append('..')
from transformers import AutoModelForCausalLM, AutoTokenizer, TrainingArguments, BitsAndBytesConfig
from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training
from trl import SFTTrainer
from data.dataset_preparation import load_and_prepare

MODEL_NAME = "mistralai/Mistral-7B-Instruct-v0.2"

def train():
    print("Loading Mistral-7B with 4-bit quantization...")
    bnb_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_compute_dtype=torch.float16,
        bnb_4bit_use_double_quant=True
    )

    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
    tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "right"

    model = AutoModelForCausalLM.from_pretrained(
        MODEL_NAME,
        quantization_config=bnb_config,
        device_map="auto",
        trust_remote_code=True
    )

    # Enable gradient checkpointing BEFORE prepare_model_for_kbit_training
    model.gradient_checkpointing_enable()

    # Prepare model for k-bit training
    model = prepare_model_for_kbit_training(model)

    lora_config = LoraConfig(
        r=8,
        lora_alpha=16,
        lora_dropout=0.05,
        target_modules=["q_proj", "v_proj", "k_proj", "o_proj"],
        task_type="CAUSAL_LM",
        bias="none"
    )

    model = get_peft_model(model, lora_config)
    model.print_trainable_parameters()

    # Load dataset in Mistral format
    dataset = load_and_prepare(model_type="mistral")

    # Training arguments
    training_args = TrainingArguments(
        output_dir="../medical_llm_mistral",
        num_train_epochs=2,
        per_device_train_batch_size=2,
        gradient_accumulation_steps=4,
        learning_rate=2e-4,
        fp16=True,
        logging_steps=50,
        save_steps=200,
        save_total_limit=2,
        warmup_ratio=0.03,
        lr_scheduler_type="cosine",
        report_to="none",
        optim="paged_adamw_8bit",
    )

    trainer = SFTTrainer(
        model=model,
        train_dataset=dataset,
        args=training_args,
        peft_config=lora_config,
        dataset_text_field="text",
        max_seq_length=512,
    )

    print("Starting Mistral-7B training...")
    trainer.train()

    print("Saving trained model...")
    trainer.save_model("../medical_llm_mistral")
    tokenizer.save_pretrained("../medical_llm_mistral")
    print("Training complete! Model saved to ../medical_llm_mistral")

if __name__ == "__main__":
    train()