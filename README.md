# 🦙 Fine-Tuning Llama-3.2-3B into a Reasoning Assistant

[![Jupyter Notebook](https://img.shields.io/badge/Jupyter-Notebook-orange?logo=jupyter)](./)
[![Unsloth](https://img.shields.io/badge/Unsloth-4bit%20LoRA-blue)](https://github.com/unslothai/unsloth)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)

An end-to-end implementation for fine-tuning **Llama-3.2-3B-Instruct** using **Unsloth 4-bit QLoRA** and **TRL's `SFTTrainer`**. This project fine-tunes the base model on synthetic stream-of-consciousness datasets to output structured reasoning steps (`<thinking>`) prior to producing final answers. The fine-tuned model is then exported to **GGUF** format and served locally using **Ollama**.

---

## 📌 Project Features

- **Efficient Fine-Tuning:** Uses 4-bit Quantized Low-Rank Adaptation (QLoRA) with Unsloth kernels to keep memory usage under 8GB VRAM (runs on free Kaggle T4/P100 or Colab GPUs).
- **Supervised Fine-Tuning Pipeline:** Powered by TRL's `SFTTrainer` to automate prompt formatting, dataset packing, and batch tokenization.
- **Reasoning Structure:** Trains the model to wrap reasoning processes inside `<thinking>` and `<solution>` XML blocks.
- **Export Formats:** Saves native PyTorch safetensors as well as quantized GGUF binaries (`Q4_K_M` / `Q8_0`).
- **Local Deployment:** Complete setup instructions for running the GGUF model via Ollama and querying it using Python REST endpoints.

---

## 🛠️ Tech Stack & Requirements

- **Base Model:** `unsloth/Llama-3.2-3B-Instruct`
- **Dataset:** `ServiceNow-AI/R1-Distill-SFT`
- **Frameworks:** PyTorch, Unsloth, Hugging Face `transformers`, `trl`, `datasets`
- **Serving Engine:** Ollama

---

## 🚀 Quickstart & Notebook Execution

### 1. View or Run the Notebook
The main code and outputs are stored directly in the executable notebook file in this repository:
- Click on the `.ipynb` file above to view the rendered outputs, training steps, and generation logs directly inside GitHub.

### 2. Environment Setup
To run the notebook on a local GPU, Kaggle, or Google Colab, install the required packages:

```bash
  # System dependencies for Ollama extraction
  sudo apt-get update && sudo apt-get install -y zstd
  
  # Python libraries
  pip install unsloth trl datasets transformers torch requests
```
---
## 💻 Pipeline Overview

```text
                                            ┌────────────────────────────────┐
                                            │  unsloth/Llama-3.2-3B-Instruct │
                                            └───────────────┬────────────────┘
                                                            │
                                                            ▼
                                            ┌────────────────────────────────┐
                                            │   4-bit QLoRA Adapter (r=16)   │
                                            └───────────────┬────────────────┘
                                                            │
                                                            ▼
                                            ┌────────────────────────────────┐
                                            │ ServiceNow-AI/R1-Distill-SFT   │ (Fine-Tuning via SFTTrainer)
                                            └───────────────┬────────────────┘
                                                            │
                                                            ▼
                                            ┌────────────────────────────────┐
                                            │    GGUF Model Export           │
                                            └───────────────┬────────────────┘
                                                            │
                                                            ▼
                                            ┌────────────────────────────────┐
                                            │   Local Serving via Ollama     │
                                            └────────────────────────────────┘
```
---

## 🦙 Running the GGUF Model via Ollama

After executing the quantization cell in the notebook, register and run your custom model locally:

1. **Start the Ollama background service:**
   ```bash
   ollama serve &
   ```

2. **Register the model using the generated Modelfile:**
   ```bash
    ollama create unsloth_model -f ./subhrajyoti-001-3B-GGUF_gguf/Modelfile
   ```

3. **Query via Python:**
   ```python
   import requests
   response = requests.post(
       "http://localhost:11434/api/generate",
       json={
           "model": "unsloth_model",
           "prompt": "How many 'r's are present in 'strawberry'?",
           "stream": False
       }
   )
   print(response.json()["response"])
   ```
---
## 📄 License

Distributed under the MIT License. See `LICENSE` for details.

---

### Checklist of Files You Should Have in Your Repo Now

| File Name | Description | Status |
| :--- | :--- | :--- |
| `*.ipynb` | Your executed Jupyter notebook containing code & training logs | Uploaded |
| `README.md` | The markdown file documenting your project | Need to create/edit |
| `.gitignore` | Prevents accidentally committing huge `.gguf` / cache files | Optional (if uploading manually) |
| `LICENSE` | MIT License file for open-source sharing | Recommended |








