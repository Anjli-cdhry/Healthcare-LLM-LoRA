import torch
import sys
sys.path.append('..')
from transformers import AutoModelForCausalLM, AutoTokenizer, TrainingArguments
from peft import LoraConfig, get_peft_model
from trl import SFTTrainer
from data.dataset_preparation import load_and_prepare

MODEL_NAME = "TinyLlama/TinyLlama-1.1B-Chat-v1.0"

def train():
    print("Loading TinyLlama model...")
    
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
    tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "right"

    model = AutoModelForCausalLM.from_pretrained(
        MODEL_NAME,
        torch_dtype=torch.float16,
        device_map="auto",
    )

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

    # Load dataset
    dataset = load_and_prepare(model_type="tinyllama")

    # Training arguments
    training_args = TrainingArguments(
        output_dir="../medical_llm_tinyllama",
        num_train_epochs=2,
        per_device_train_batch_size=2,
        gradient_accumulation_steps=4,
        learning_rate=2e-4,
        fp16=torch.cuda.is_available(),
        logging_steps=50,
        save_steps=200,
        save_total_limit=2,
        warmup_ratio=0.03,
        lr_scheduler_type="cosine",
        report_to="none",
        optim="adamw_torch",
    )

    trainer = SFTTrainer(
        model=model,
        train_dataset=dataset,
        args=training_args,
        peft_config=lora_config,
        dataset_text_field="text",
        max_seq_length=512,
    )

    print("Starting TinyLlama training...")
    trainer.train()

    print("Saving trained model...")
    trainer.save_model("../medical_llm_tinyllama")
    tokenizer.save_pretrained("../medical_llm_tinyllama")
    print("Training complete! Model saved to ../medical_llm_tinyllama")

if __name__ == "__main__":
    train()