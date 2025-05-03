# Neural Machine Translation (Chinese-to-English)

This repository part A implements two models for Chinese-to-English translation:  
1. **RNN-based Seq2Seq with Bahdanau Attention**  
2. **MinGPT (Transformer-based)**  

---

## 🗂️ Code Structure  
```
.
├── cmn-eng/
│ └── _about.txt
│ └── cmn.txt # Raw parallel corpus (Chinese-English)
├── minGPT/
├── plots/
├── seq2seq.py # Train RNN-based model
├── minGPT.py # Train GPT model
├── bpe.py 
└── tokenization_gpt2.py
```

---
## ⚙️ Environment Setup  
**Python** 3.8+ 
and **PyTorch** 2.0+

**scikit-learn** (for dataset splitting), 
**tqdm** (progress bars) and 
**matplotlib** (visualization)

---
##  🚀 Training Commands
You can direct run the `.py` files with **ipy-kernel**, the evaluations process are also included, or customize your training use with the following command:
1. **Train RNN-based Seq2Seq Model**
```bash
python train_seq2seq.py \
    --data_path ./data/cmn-eng/cmn.txt \
    --hidden_size 256 \
    --batch_size 32 \
    --epochs 20 \
    --save_dir ./models/seq2seq
```
2. **Train MinGPT Model**
```bash
python train_mingpt.py \
    --data_path ./data/cmn-eng/cmn.txt \
    --batch_size 128 \
    --max_iters 5000 \
    --save_dir ./models/mingpt
```

Or you can write your own `evaluation.py` and customize your use your saved `.pth`
```
python evaluate.py \
    --model_type seq2seq \
    --encoder_checkpoint ./models/seq2seq/encoder.pth \
    --decoder_checkpoint ./models/seq2seq/decoder.pth

python evaluate.py \
    --model_type mingpt \
    --checkpoint ./models/mingpt/gpt_translator.pth
```

---

## 📝 Key Implementation Details
- **Tokenization**:

    - Chinese: Character-level tokenization (`Lang` class)

    - English: Word-level tokenization with `<UNK>` for OOV words.

- **Seq2Seq Architecture**:

    - GRU encoder-decoder with Bahdanau attention.

    - Max sequence length: `35` tokens.

- **MinGPT Architecture**:

    - 6-layer transformer with 8 attention heads (`93.28M` parameters).

    - Temperature sampling (`0.7`) for autoregressive decoding.

---

## 📊 Results
Please refer to the tables and figures in `report.pdf`

🛠️ Reproducibility Note: Set `random_state=42` in data splitting for deterministic results.