from datasets import load_dataset, concatenate_datasets

def load_and_prepare(model_type="tinyllama"):
    print("Loading multiple medical datasets...")

    # Dataset 1 - MedQA
    print("Loading MedQA dataset...")
    ds1 = load_dataset("medalpaca/medical_meadow_medqa")

    # Dataset 2 - MedAlpaca
    print("Loading MedAlpaca dataset...")
    ds2 = load_dataset("medalpaca/medical_meadow_wikidoc_patient_information")

    # Dataset 3 - HealthCareMagic
    print("Loading HealthCareMagic dataset...")
    ds3 = load_dataset("lavita/ChatDoctor-HealthCareMagic-100k")

    def format_tinyllama(example):
        if 'input' in example and 'output' in example:
            q = example['input']
            a = example['output']
        elif 'question' in example and 'answer' in example:
            q = example['question']
            a = example['answer']
        elif 'Patient' in example and 'Doctor' in example:
            q = example['Patient']
            a = example['Doctor']
        else:
            q = str(list(example.values())[0])
            a = str(list(example.values())[1])
        return {
            "text": f"<|system|>\nYou are an expert medical assistant. Provide accurate, detailed medical information.</s>\n<|user|>\n{q}</s>\n<|assistant|>\n{a}</s>"
        }

    def format_mistral(example):
        if 'input' in example and 'output' in example:
            q = example['input']
            a = example['output']
        elif 'question' in example and 'answer' in example:
            q = example['question']
            a = example['answer']
        elif 'Patient' in example and 'Doctor' in example:
            q = example['Patient']
            a = example['Doctor']
        else:
            q = str(list(example.values())[0])
            a = str(list(example.values())[1])
        return {
            "text": f"[INST] You are an expert medical assistant. {q} [/INST] {a}"
        }

    fmt = format_tinyllama if model_type == "tinyllama" else format_mistral

    # Format all datasets
    f1 = ds1['train'].select(range(2000)).map(fmt, remove_columns=ds1['train'].column_names)
    f2 = ds2['train'].select(range(1500)).map(fmt, remove_columns=ds2['train'].column_names)
    f3 = ds3['train'].select(range(1500)).map(fmt, remove_columns=ds3['train'].column_names)

    # Combine all
    combined = concatenate_datasets([f1, f2, f3])
    combined = combined.shuffle(seed=42)

    print(f"Total training samples: {len(combined)}")
    print(f"Sample:\n{combined[0]['text'][:300]}")

    return combined


if __name__ == "__main__":
    data = load_and_prepare("tinyllama")
    print("Dataset preparation complete!")