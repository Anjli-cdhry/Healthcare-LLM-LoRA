import streamlit as st
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, pipeline
from peft import PeftModel

# Page config
st.set_page_config(
    page_title="Healthcare AI Assistant",
    page_icon="🏥",
    layout="wide"
)

# Custom CSS
st.markdown("""
<style>
    .main-header {
        font-size: 2.5rem;
        color: #2E86AB;
        text-align: center;
        padding: 1rem;
    }
    .sub-header {
        font-size: 1rem;
        color: #666;
        text-align: center;
        margin-bottom: 2rem;
    }
    .stChatMessage {
        border-radius: 10px;
        padding: 10px;
    }
</style>
""", unsafe_allow_html=True)

st.markdown('<h1 class="main-header">🏥 Healthcare AI Assistant</h1>', unsafe_allow_html=True)
st.markdown('<p class="sub-header">Powered by Fine-tuned TinyLlama with LoRA | Trained on 5000+ Medical QA samples</p>', unsafe_allow_html=True)

# Sidebar
with st.sidebar:
    st.header("⚙️ Settings")
    st.info("**Model:** TinyLlama-1.1B\n\n**Training:** LoRA Fine-tuned\n\n**Dataset:** 5000+ Medical QA")
    st.warning("⚠️ For educational purposes only. Always consult a doctor.")
    max_tokens = st.slider("Max Response Length", 100, 500, 200)
    temperature = st.slider("Temperature", 0.1, 1.0, 0.7)
    st.divider()
    if st.button("Clear Chat History"):
        st.session_state.messages = []
        st.rerun()

# Load model
@st.cache_resource
def load_model():
    with st.spinner("Loading AI model... Please wait..."):
        MODEL_PATH = "../medical_llm_tinyllama"
        tokenizer = AutoTokenizer.from_pretrained(MODEL_PATH)
        model = AutoModelForCausalLM.from_pretrained(
            MODEL_PATH,
            torch_dtype=torch.float32,
            device_map="cpu",
        )
        pipe = pipeline(
            "text-generation",
            model=model,
            tokenizer=tokenizer
        )
    return pipe

pipe = load_model()
st.success("✅ Model loaded successfully!")

# Chat history
if "messages" not in st.session_state:
    st.session_state.messages = []

# Display chat history
for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

# Chat input
if prompt := st.chat_input("Ask a medical question..."):
    st.session_state.messages.append({"role": "user", "content": prompt})
    with st.chat_message("user"):
        st.markdown(prompt)

    with st.chat_message("assistant"):
        with st.spinner("Thinking..."):
            formatted_prompt = f"""<|system|>
You are a helpful medical assistant. Answer medical questions accurately.</s>
<|user|>
{prompt}</s>
<|assistant|>"""
            output = pipe(
                formatted_prompt,
                max_new_tokens=max_tokens,
                temperature=temperature,
                do_sample=True,
                repetition_penalty=1.1
            )
            response = output[0]['generated_text']
            answer = response.split("<|assistant|>")[-1].strip()
            st.markdown(answer)

    st.session_state.messages.append({"role": "assistant", "content": answer})