import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, pipeline

MODEL_PATH = "../medical_llm_tinyllama"

def load_model():
    print("Loading model...")
    tokenizer = AutoTokenizer.from_pretrained(MODEL_PATH)
    model = AutoModelForCausalLM.from_pretrained(
        MODEL_PATH,
        torch_dtype=torch.float32,
        device_map="cpu",
    )
    return model, tokenizer

def get_answer(question, pipe):
    prompt = f"""<|system|>
You are a helpful medical assistant. Answer medical questions accurately.</s>
<|user|>
{question}</s>
<|assistant|>"""

    output = pipe(
        prompt,
        max_new_tokens=200,
        temperature=0.7,
        do_sample=True,
        repetition_penalty=1.1
    )
    response = output[0]['generated_text']
    answer = response.split("<|assistant|>")[-1].strip()
    return answer

if __name__ == "__main__":
    model, tokenizer = load_model()
    pipe = pipeline(
        "text-generation",
        model=model,
        tokenizer=tokenizer
    )

    print("\n=== Medical AI Assistant ===")
    print("Type 'quit' to exit\n")

    test_questions = [
        "What are symptoms of diabetes?",
        "How is pneumonia diagnosed?",
        "What causes high blood pressure?"
    ]

    print("--- Automatic Test ---")
    for q in test_questions:
        print(f"\nQ: {q}")
        print(f"A: {get_answer(q, pipe)}")
        print("-" * 50)

    print("\n--- Chat Mode ---")
    while True:
        question = input("\nEnter your medical question: ")
        if question.lower() == 'quit':
            break
        print(f"Answer: {get_answer(question, pipe)}")