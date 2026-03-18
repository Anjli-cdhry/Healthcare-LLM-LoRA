import torch
import sys
sys.path.append('..')
from transformers import AutoModelForCausalLM, AutoTokenizer, pipeline
from rouge_score import rouge_scorer
import json

MODEL_PATH = "../medical_llm_tinyllama"

# Test questions with reference answers
test_data = [
    {
        "question": "What are the symptoms of diabetes?",
        "reference": "Common symptoms of diabetes include frequent urination, excessive thirst, unexplained weight loss, fatigue, blurred vision, slow healing wounds, and frequent infections."
    },
    {
        "question": "What causes high blood pressure?",
        "reference": "High blood pressure can be caused by obesity, lack of physical activity, too much salt, stress, smoking, alcohol consumption, age, family history, and certain medications."
    },
    {
        "question": "What are symptoms of pneumonia?",
        "reference": "Pneumonia symptoms include cough with phlegm, fever, chills, shortness of breath, chest pain, fatigue, nausea and vomiting."
    },
    {
        "question": "How is anemia treated?",
        "reference": "Anemia treatment depends on the cause. Iron deficiency anemia is treated with iron supplements and diet changes. Other types may require vitamin B12, folic acid, or blood transfusions."
    },
    {
        "question": "What are the symptoms of PCOD?",
        "reference": "PCOD symptoms include irregular periods, excess androgen, polycystic ovaries, weight gain, acne, hair loss, and infertility."
    }
]

def load_model():
    print("Loading model...")
    tokenizer = AutoTokenizer.from_pretrained(MODEL_PATH)
    model = AutoModelForCausalLM.from_pretrained(
        MODEL_PATH,
        torch_dtype=torch.float32,
        device_map="cpu",
    )
    pipe = pipeline("text-generation", model=model, tokenizer=tokenizer)
    print("Model loaded!")
    return pipe

def get_answer(question, pipe):
    prompt = f"""<|system|>
You are a helpful medical assistant. Answer medical questions accurately.</s>
<|user|>
{question}</s>
<|assistant|>"""
    output = pipe(prompt, max_new_tokens=200, temperature=0.7, do_sample=True, repetition_penalty=1.1)
    response = output[0]['generated_text']
    return response.split("<|assistant|>")[-1].strip()

def evaluate_model(pipe):
    print("\n" + "="*60)
    print("EVALUATING TINYLLAMA MEDICAL MODEL")
    print("="*60)

    scorer = rouge_scorer.RougeScorer(['rouge1', 'rouge2', 'rougeL'], use_stemmer=True)

    results = []
    total_rouge1 = 0
    total_rouge2 = 0
    total_rougeL = 0

    for i, item in enumerate(test_data):
        print(f"\nTest {i+1}/{len(test_data)}: {item['question']}")
        generated = get_answer(item['question'], pipe)
        scores = scorer.score(item['reference'], generated)

        rouge1 = scores['rouge1'].fmeasure
        rouge2 = scores['rouge2'].fmeasure
        rougeL = scores['rougeL'].fmeasure

        total_rouge1 += rouge1
        total_rouge2 += rouge2
        total_rougeL += rougeL

        print(f"Generated: {generated[:150]}...")
        print(f"ROUGE-1: {rouge1:.4f} | ROUGE-2: {rouge2:.4f} | ROUGE-L: {rougeL:.4f}")

        results.append({
            "question": item['question'],
            "generated": generated,
            "rouge1": rouge1,
            "rouge2": rouge2,
            "rougeL": rougeL
        })

    n = len(test_data)
    print("\n" + "="*60)
    print("FINAL RESULTS")
    print("="*60)
    print(f"Average ROUGE-1: {total_rouge1/n:.4f}")
    print(f"Average ROUGE-2: {total_rouge2/n:.4f}")
    print(f"Average ROUGE-L: {total_rougeL/n:.4f}")
    print("="*60)

    with open("evaluation_results.json", "w") as f:
        json.dump(results, f, indent=2)
    print("Results saved to evaluation_results.json")

if __name__ == "__main__":
    pipe = load_model()
    evaluate_model(pipe)