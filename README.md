# 🏥 Healthcare Instruction-Tuned LLM using LoRA

Fine-tuned **TinyLlama-1.1B**  on 5000+ medical QA samples using **LoRA (Low-Rank Adaptation)** and **4-bit quantization**. Built end-to-end pipeline with Streamlit deployment and Medical Agent with conversation memory.

## 🚀 Key Features
- Fine-tuned LLMs on 3 medical datasets (5000+ samples)
- LoRA fine-tuning — only 0.02% parameters trained
- 4-bit quantization for memory efficiency
- Streamlit chatbot UI with adjustable parameters
- Medical AI Agent with conversation memory
- ROUGE evaluation metrics

## 🛠️ Tech Stack
- **Models:** TinyLlama-1.1B, 
- **Fine-tuning:** LoRA, QLoRA, PEFT
- **Framework:** PyTorch, HuggingFace Transformers
- **Deployment:** Streamlit
- **Agent:** Custom Memory-based Conversational Agent
- **Evaluation:** ROUGE-1, ROUGE-2, ROUGE-L

## 📊 Dataset
| Dataset | Samples |
|---------|---------|
| MedAlpaca MedQA | 2000 |
| MedAlpaca WikiDoc | 1500 |
| ChatDoctor HealthCareMagic | 1500 |
| **Total** | **5000** |

## 📈 Evaluation Results
| Metric | Score |
|--------|-------|
| ROUGE-1 | 0.2199 |
| ROUGE-2 | 0.0674 |
| ROUGE-L | 0.1447 |

## 📁 Project Structure
```
Healthcare-LLM-LoRA/
├── data/dataset_preparation.py
├── training/
│   ├── train_lora.py
│   └── train_mistral.py
├── inference/chat.py
├── app/medical_chatbot.py
├── agent/medical_agent.py
├── evaluate/evaluate.py
└── README.md
```

## ⚡ Quick Start
```bash
git clone https://github.com/YOUR_USERNAME/healthcare-llm-lora
cd healthcare-llm-lora
pip install -r requirements.txt
cd app
streamlit run medical_chatbot.py
```

## ⚠️ Disclaimer
For educational purposes only. Always consult a qualified medical professional.
