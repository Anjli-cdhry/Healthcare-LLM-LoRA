import torch
import sys
sys.path.append('..')
from transformers import AutoModelForCausalLM, AutoTokenizer, pipeline

MODEL_PATH = "../../medical_llm_tinyllama"

class MedicalAgent:
    def __init__(self, pipe):
        self.pipe = pipe
        self.conversation_history = []
        print("Medical Agent initialized with conversation memory!")

    def get_response(self, user_input):
        # Build history context
        history = ""
        for msg in self.conversation_history[-4:]:
            history += f"Patient: {msg['patient']}\nDoctor: {msg['doctor']}\n"

        prompt = f"""<|system|>
You are an expert medical assistant. You remember the conversation history.
{history}</s>
<|user|>
{user_input}</s>
<|assistant|>"""

        output = self.pipe(
            prompt,
            max_new_tokens=200,
            temperature=0.7,
            do_sample=True,
            repetition_penalty=1.1
        )
        response = output[0]['generated_text']
        answer = response.split("<|assistant|>")[-1].strip()

        # Save to memory
        self.conversation_history.append({
            "patient": user_input,
            "doctor": answer
        })
        return answer

    def clear_memory(self):
        self.conversation_history = []
        print("Memory cleared!")

def load_model():
    print("Loading TinyLlama model...")
    tokenizer = AutoTokenizer.from_pretrained(MODEL_PATH)
    model = AutoModelForCausalLM.from_pretrained(
        MODEL_PATH,
        torch_dtype=torch.float32,
        device_map="cpu",
    )
    pipe = pipeline("text-generation", model=model, tokenizer=tokenizer)
    print("Model loaded!")
    return pipe

def run_agent():
    pipe = load_model()
    agent = MedicalAgent(pipe)

    print("\n" + "="*60)
    print("MEDICAL AI AGENT - With Conversation Memory")
    print("="*60)
    print("Commands: 'quit' to exit, 'clear' to clear memory\n")

    # Demo showing memory works
    test_conversation = [
        "I have been having headaches for 3 days.",
        "The pain is mostly on the right side.",
        "What medicines can help with what I described?",
    ]

    print("--- Demo Conversation (showing memory) ---")
    for question in test_conversation:
        print(f"\nPatient: {question}")
        response = agent.get_response(question)
        print(f"Doctor: {response[:200]}...")
        print("-" * 40)

    print("\n--- Chat Mode ---")
    while True:
        user_input = input("\nPatient: ")
        if user_input.lower() == 'quit':
            break
        if user_input.lower() == 'clear':
            agent.clear_memory()
            continue
        response = agent.get_response(user_input)
        print(f"Doctor: {response}")

if __name__ == "__main__":
    run_agent()