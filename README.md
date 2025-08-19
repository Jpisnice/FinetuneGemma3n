# FinetuneGemma3n

A small cookbook for fine-tuning Gemma-3N models using the Unsloth FastModel wrapper and TRL (SFTTrainer).

This repository contains a runnable Colab/VS Code notebook that demonstrates how to:
- load a Gemma-3N base model (4-bit quantized),
- prepare conversational datasets for instruction/response fine-tuning,
- apply LoRA-style PEFT adapters,
- train using TRL's SFTTrainer configured to train only on assistant responses,
- save the resulting adapters and export to formats such as merged fp16 or GGUF for deployment.

Quick links
- Notebook: `finetuneGemma3n.ipynb` (Colab badge available inside the notebook)

Requirements
- Python 3.8+ (recommended 3.10+)
- GPU with CUDA for training (optional for small tests)
- The notebook installs the required libraries when run in Colab. Locally, install the packages below:

```
pip install -r requirements.txt
```

(If there is no `requirements.txt`, the notebook installs the necessary packages automatically.)

Getting started (Colab)
1. Open `finetuneGemma3n.ipynb` in Colab using the badge at the top of the notebook.
2. Run the installation cells to install dependencies.
3. Edit the model/dataset cells as needed (e.g., change model name or dataset split).
4. Run the training cells.

Key notebook sections
- Installation: installs Unsloth and supporting libraries (bitsandbytes, accelerate, trl, peft, etc.)
- Load the Model: demonstrates FastModel.from_pretrained and 4-bit loading
- Dataset Preparation: loads a dataset, standardizes chat format and masks instruction tokens
- Training: configures TRL's SFTTrainer and trains only on assistant responses
- Saving: shows how to save LoRA adapters, merged fp16 model, and export GGUF

Saving and Export
- `model.save_pretrained("gemma-3n")` and `tokenizer.save_pretrained("gemma-3n")` saves LoRA adapters locally.
- The notebook contains examples to merge and save to fp16 for VLLM or export to GGUF for llama.cpp.

Troubleshooting
- Notebook rendering on GitHub: If you run into widget metadata rendering errors (missing `metadata.widgets.state`), open the notebook locally and remove `metadata.widgets` or run a small nbformat script to clean metadata. The repository's notebook has been cleaned for GitHub rendering.
- Out-of-memory: reduce batch size, use gradient accumulation, or use 4-bit loading as shown.
