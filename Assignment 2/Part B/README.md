# Legal Domain Fine-tuning with Qwen2.5-7B and LoRA

This repository implements Part B of the assignment: **Quantized LoRA fine-tuning of Qwen2.5-7B-Instruct** on the `DISC-Law-SFT` dataset using `LLaMA-Factory`. The model demonstrates competency in Chinese legal Q&A tasks while operating efficiently on a T4 GPU.

---

## 🗂️ Code Structure  
```
.
├── data/
│ └── dataset_info.json
│ └── disc_law_sft/
|   |── train/ # data arrow
|   |── test/ # data arrow
├── lora_law/ # all training results and checkpoints
├── src/ # utils of LLaMA Factory
├── DL_Assignment2_partB.ipynb # main finetuing
├── setup.py # install for LLaMA Factory
└── train_config.json # config file for training param.s
```

---
## ⚙️ Environment Setup  
**Requirements**:  
- Python 3.10+
- PyTorch 2.1+ with CUDA 11.8
- LLaMA-Factory (`pip install -e .[torch,bitsandbytes]`)
- Hugging Face Libraries (`datasets`, `transformers`, `accelerate`)
- NVIDIA Driver ≥535 (for T4 GPU compatibility)

**Critical Dependencies**:  
```bash
!pip install numpy==1.26.4 bitsandbytes==0.43.0
!pip uninstall thinc gcsfs -y  # Resolve conflicts
```

---
##  🚀 Training Pipeline
```bash
# Clone repository
git clone https://github.com/hiyouga/LLaMA-Factory.git
cd LLaMA-Factory

# Start training with 4-bit quantization
llamafactory-cli train configs/train_llama3.json
```
**Key Configuration (`train_llama3.json`)**
```bash
{
  "model_name_or_path": "Qwen/Qwen2.5-7B-Instruct",
  "dataset": "disc_law_sft",
  "template": "chatml",
  "finetuning_type": "lora",
  "lora_target": "c_attn,q_proj,v_proj",
  "quantization_bit": 4,
  "per_device_train_batch_size": 4,
  "gradient_accumulation_steps": 4,
  "learning_rate": 3e-5,
  "num_train_epochs": 0.05
}
```

**Inference**
```bash
llamafactory-cli chat \
  --model_name_or_path Qwen/Qwen2.5-7B-Instruct \
  --adapter_path llama3_lora_law
```

---

## 📝 Key Implementation Details
**LoRA Configuration**

- Target Layers: `c_attn`, `q_proj`, `v_proj`

- Quantization: 4-bit NF4 (bitsandbytes)

- Rank: 8 (default)

- LoRA+: Enabled with `loraplus_lr_ratio=16`

**Dataset Processing**

- Format Conversion: Original JSONL → Alpaca format (`instruction`/`input`/`output`)

- Train-Test Split: 90%-10% with `seed=42`

- Context Template: `"根据中国法律知识回答以下问题"`

---

## 📊 Performance Summary
Please refer to the tables and figures in `report.pdf`

🛠️ Reproducibility Note: Set `random_state=42` in data splitting for deterministic results.
Full configuration exported in `train_llama3.json` and training logs stored in `trainer_log.jsonl`
