# Pico LLM

Pico LLM is a lightweight, modular, and highly extensible framework for building small-scale language models using custom neural architectures. The project supports CNN-based, MLP-based, and hybrid token encoders, enabling experimentation with computationally efficient NLP models.

---

## 🚀 Features

**• Modular Architecture** — Switch easily between K-gram CNN, K-gram MLP, or other encoders.

**• Custom Token Processors** — Implement windowing, stacking, positional embeddings, or byte-level processing.

**• Flexible Training Pipeline** — Custom training loops for debugging and rapid prototyping.

**• Long-Range Context Support** — Optional dilated CNN layers and layer normalization.

**• Compact and Deployable** — Designed to run on limited hardware.

---

## 📁 Project Structure (High-Level Architecture)

```
pico-llm/
│
├── pico-llm.py               # Main training script
├── KgramCNN.py               # CNN-based k-gram encoder
├── kgramMLP.py               # MLP-based k-gram encoder
│
├── data/
│   ├── dataset_loader.py     # Loads and tokenizes text
│   └── sample_text.txt       # Sample dataset
│
├── models/
│   ├── __init__.py
│   ├── transformer_block.py  # Optional transformer experiments
│   └── utils.py              # Shared functions (loss, initialization, etc.)
│
├── training/
│   ├── trainer.py            # Epoch loop, batching, evaluation
│   └── metrics.py            # Perplexity, accuracy, loss curves
│
├── experiments/
│   ├── config.json           # Model hyperparameters
│   └── results/              # Logs and checkpoints
│
└── README.md                 # Project documentation
```

---

## 🔧 Installation

```
git clone <repo-url>
cd pico-llm
pip install -r requirements.txt
```

---

## 🧠 Model Architectures

### 1. **K-gram CNN Encoder (KgramCNN.py)**

* Multi-channel 1D CNN
* Layer Normalization (optional)
* Dilated convolutions for long-range context
* Max-pooling or attention-like aggregation

### 2. **K-gram MLP Encoder (kgramMLP.py)**

* Fully-connected layers over k-gram windows
* Fast for small vocabularies and tiny models
* Dropout/LN supported

### 3. **Main LLM Wrapper (pico-llm.py)**

Handles:

* Tokenization
* Dataset slicing
* Model loading
* Training loop
* Validation logging

---

## 📊 Training

```
python pico-llm.py --model kgram_cnn \
                   --epochs 20 \
                   --lr 3e-4 \
                   --context 128
```

You can switch between encoder types:

```
--model kgram_mlp
--model kgram_cnn
```

---

## 📦 Output

The framework generates:

* `.pt` model checkpoints
* training logs
* perplexity curves
* generated text samples

---

## 🧩 Example Architecture Diagram

```
                +---------------------------+
                |       Dataset Loader      |
                +--------------+------------+
                               |
                               v
                    +----------+----------+
                    |   Token Processor   |
                    +----------+----------+
                               |
                    +----------v----------+
                    |   K-gram Builder   |
                    +----------+----------+
                               |
            +------------------v------------------+
            |       Encoder Module (Choose)       |
            |  - KgramCNN                         |
            |  - KgramMLP                         |
            +------------------+------------------+
                               |
                      +--------v--------+
                      |   LLM Head      |
                      +--------+--------+
                               |
                        +------v------+
                        |   Trainer   |
                        +-------------+
```

---

## 🛠️ Future Enhancements

* Add Rotary Positional Embeddings
* Add lightweight attention block
* Add Byte-level tokenizer
* Add benchmarking suite

---

## 🤝 Contributing

Pull requests are welcome. Please open an issue for major changes.

---

## 📜 License

MIT License.
