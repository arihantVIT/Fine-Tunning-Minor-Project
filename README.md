# 🧠 Reasoning Model — Fine-Tuning Llama 3.2 with QLoRA

A minor project that fine-tunes **Meta's Llama 3.2 (3B)** into a **reasoning model** using **QLoRA** via [Unsloth](https://github.com/unslothai/unsloth), trained on the **ServiceNow R1-Distill-SFT** dataset. The resulting model mimics human-like, stream-of-consciousness reasoning — exploring, self-doubting, and iteratively refining its answers before arriving at a final response.

**If the notebook preview does not render on GitHub, please download and open the .ipynb file locally or in Google Colab.**
---

## 📌 Project Overview

| Detail | Value |
|--------|-------|
| Base Model | `unsloth/Llama-3.2-3B-Instruct` |
| Fine-Tuning Method | QLoRA (Quantized Low-Rank Adaptation) |
| Dataset | `ServiceNow-AI/R1-Distill-SFT` (v0) — 171,647 examples |
| Training Steps | 60 |
| Hardware | NVIDIA Tesla T4 (Google Colab) |
| Output Format | LoRA adapters + GGUF |

---

## 🚀 What This Project Does

1. **Loads** a 4-bit quantized Llama 3.2 (3B) model using Unsloth's `FastLanguageModel`
2. **Attaches QLoRA adapters** (`r=16`) on all key projection layers
3. **Formats the dataset** using a custom chain-of-thought prompt template that wraps problems with `<think>` reasoning blocks
4. **Fine-tunes** the model using `SFTTrainer` with AdamW 8-bit optimizer
5. **Runs inference** to test the model's reasoning ability
6. **Saves** the fine-tuned model in both LoRA adapter format and GGUF format

---

## 🗂️ Repository Structure

```
├── Fine_Tunning_Minor_Project.ipynb   # Main training notebook
├── requirements.txt                   # Python dependencies
├── README.md                          # Project documentation
└── arihant-reasoning/                 # Saved model outputs (generated after training)
    ├── tokenizer_config.json
    ├── chat_template.jinja
    └── tokenizer.json
```

---

## ⚙️ Setup & Installation

This project is designed to run on **Google Colab** with a GPU (T4 or better).

### 1. Clone the Repository

```bash
git clone https://github.com/<your-username>/<your-repo-name>.git
cd <your-repo-name>
```

### 2. Open in Google Colab

Upload `Fine_Tunning_Minor_Project.ipynb` to Google Colab, or open it directly from GitHub via the Colab interface. Make sure to select a **GPU runtime** (T4 recommended).

### 3. Install Dependencies

The notebook handles installation automatically via:

```python
!pip install -q unsloth
!pip install -q --force-reinstall --no-cache-dir --no-deps git+https://github.com/unslothai/unsloth.git
```

Or install manually from `requirements.txt`:

```bash
pip install -r requirements.txt
```

---

## 🧩 Key Components

### Model Loading
The model is loaded in 4-bit quantization to reduce memory footprint while preserving performance:

```python
model, tokenizer = FastLanguageModel.from_pretrained(
    model_name='unsloth/Llama-3.2-3B-Instruct',
    max_seq_length=2048,
    load_in_4bit=True
)
```

### QLoRA Configuration
LoRA adapters are applied to all attention and MLP projection layers:

```python
model = FastLanguageModel.get_peft_model(
    model,
    r=16,
    target_modules=["q_proj", "k_proj", "v_proj", "o_proj",
                    "gate_proj", "up_proj", "down_proj"],
    lora_alpha=16,
    lora_dropout=0,
    use_gradient_checkpointing="unsloth"
)
```

### Prompt Template
The model is trained with a structured reasoning prompt that encourages step-by-step thinking:

```
You are a reflective assistant engaging in thorough, iterative reasoning,
mimicking human stream-of-consciousness thinking...

<problem>
{problem}
</problem>

{chain_of_thought}
{solution}
```

### Dataset
- **Source:** `ServiceNow-AI/R1-Distill-SFT` — a distilled reasoning dataset inspired by DeepSeek-R1
- **Size:** 171,647 examples covering math, logic, and word problems
- **Fields used:** `problem`, `reannotated_assistant_content` (think traces), `solution`

### Training Configuration

| Hyperparameter | Value |
|---------------|-------|
| Batch size | 2 per device |
| Gradient accumulation | 4 steps |
| Effective batch size | 8 |
| Learning rate | 2e-4 |
| LR scheduler | Linear |
| Optimizer | AdamW 8-bit |
| Max steps | 60 |
| Warmup steps | 5 |
| Weight decay | 0.01 |
| Trainable parameters | 24.3M / 3.2B (0.75%) |

---

## 📊 Training Results

The model trained for 60 steps, with loss decreasing from ~1.0 to ~0.5:

| Step | Loss |
|------|------|
| 1 | 1.007 |
| 10 | 0.737 |
| 30 | 0.471 |
| 60 | 0.503 |

---

## 🔍 Sample Inference

After training, the model is tested with a reasoning question:

**Prompt:** *How many 'r's are present in 'strawberry'?*

The fine-tuned model responds with a deliberate, exploratory reasoning trace — thinking through the problem letter-by-letter before reaching a conclusion, closely mimicking the style seen in DeepSeek-R1 and similar reasoning models.

---

## 💾 Saving the Model

The model is saved in two formats:

```python
# LoRA adapter format
model.save_pretrained("arihant-reasoning")
tokenizer.save_pretrained("arihant-reasoning")

# GGUF format (for llama.cpp / Ollama compatibility)
model.save_pretrained_gguf("arihant-reasoning", tokenizer)
```

---

## 🛠️ Technologies Used

- [Unsloth](https://github.com/unslothai/unsloth) — 2x faster fine-tuning with memory optimization
- [Hugging Face Transformers](https://github.com/huggingface/transformers) — model architecture & tokenization
- [PEFT](https://github.com/huggingface/peft) — parameter-efficient fine-tuning (LoRA)
- [TRL](https://github.com/huggingface/trl) — `SFTTrainer` for supervised fine-tuning
- [Datasets](https://github.com/huggingface/datasets) — dataset loading and preprocessing
- [PyTorch](https://pytorch.org/) — training backend
- Google Colab — training environment (Tesla T4 GPU)

---

## 📚 References

- [Unsloth Documentation](https://docs.unsloth.ai/)
- [ServiceNow-AI/R1-Distill-SFT Dataset](https://huggingface.co/datasets/ServiceNow-AI/R1-Distill-SFT)
- [Meta Llama 3.2](https://huggingface.co/meta-llama/Llama-3.2-3B-Instruct)
- [DeepSeek-R1 Paper](https://arxiv.org/abs/2501.12948) — inspiration for reasoning distillation

---

## 👤 Author

**Arihant**  
Minor Project — Fine-Tuning a Reasoning Model  
