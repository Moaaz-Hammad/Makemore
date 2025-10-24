# Makemore — Character-level Language Modeling (Two Architectures)

A small educational repository that implements and compares two approaches for character-level sequence modeling:

- A classic feed‑forward Neural Probabilistic Language Model (NPLM) inspired by Bengio et al. (2003).  
- A modern Dilated Causal Convolutional model (WaveNet-like) that demonstrates efficient receptive field growth via dilations.

This repo is intended for learning, experimentation, and demonstration rather than production use. The code is straightforward and annotated so you can trace how inputs (names or short sequences) are tokenized, embedded, and used to predict next characters.

---

Table of contents
- About
- Quick start
- Project structure
- What’s inside (architectures)
  - 1) Bengio-style NPLM (with image)
  - 2) Dilated Causal Convolutions (with image)
- How to train / sample
- Dependencies
- Citations
- Contributing / Contact
- License

---

About
-----
Makemore shows two different ways to solve the same problem — predicting the next character in a sequence — to illustrate the tradeoffs between older feed-forward NPLMs and modern convolutional sequence models that keep causality and expand receptive field efficiently.

Quick start
-----------
1. Clone the repository:
   ```
   git clone https://github.com/Moaaz-Hammad/Makemore.git
   ```
2. Create and activate a venv (recommended):
   ```
   python -m venv .venv
   source .venv/bin/activate  # macOS / Linux
   .venv\Scripts\activate     # Windows
   ```
3. Install dependencies:
   ```
   pip install -r requirements.txt
   ```
   If a requirements file is not present, install PyTorch and the usual data libs manually:
   ```
   pip install torch numpy tqdm jupyter
   ```
4. Open the main notebook for the Bengio model:
   ```
   Makemore using BENGIO, DUCHARME, VINCENT AND JAUVIN.ipynb
   ```
   Or run the scripts for the dilated conv model (if present):
   ```
   python dilated_conv_model.py
   python train.py
   python generate.py
   ```

Project structure
-----------------
- `Makemore using BENGIO, DUCHARME, VINCENT AND JAUVIN.ipynb` — A Jupyter notebook implementing the NPLM (data loading, vocab, PyTorch model, training loop, sampling).
- `BENGIO, DUCHARME, VINCENT AND JAUVIN.png` — Diagram referencing the original NPLM architecture (included below).
- `names.txt` — Example dataset (one name per line) used for training and demonstration.
- `dilated_conv_model.py` — (Example) implementation of a dilated causal convolutional network.
- `train.py`, `generate.py` — Training and sampling scripts for the dilated conv model (if present).
- `dilated_causal_convolution.png` — Diagram illustrating dilated causal convolution receptive fields (included below).

If a script is missing, the notebook contains fully working code for the Bengio NPLM and can be adapted to a `.py` script.

What’s inside — short descriptions
----------------------------------

1) Bengio-style Neural Probabilistic Language Model (NPLM)
- Goal: Predict the next character from a fixed-size context (N-gram window).
- Core components:
  - Embedding layer: map discrete characters to dense vectors.
  - Concatenate embeddings for the context window.
  - Feed-forward hidden layer(s) -> softmax over vocabulary for next character.
- Strengths: Simple, interpretable, good for teaching the fundamentals of embedding + feed-forward context modeling.
- Limitations: Fixed context window (can't easily capture very long-range dependencies).

Figure: Bengio-style NPLM
-------------------------
Below is the NPLM diagram illustrating the embedding lookups, hidden layer (tanh) and the final softmax prediction over the vocabulary.

![Bengio et al. NPLM](./BENGIO,%20DUCHARME,%20VINCENT%20AND%20JAUVIN.png)
*Figure: Bengio, Ducharme, Vincent and Jauvin — Neural Probabilistic Language Model (NPLM).*

2) Dilated Causal Convolutions (WaveNet-like)
- Goal: Model sequences with causal convolutions and exponentially growing receptive fields via dilation.
- Core ideas:
  - Causality: outputs depend only on past timesteps (no peeking ahead).
  - Dilation: expand effective context exponentially with depth while keeping computation per layer small.
  - Stacked layers with increasing dilation factors (1, 2, 4, 8, ...).
- Strengths: Capture long-range dependencies with fewer layers; efficient and parallelizable relative to RNNs.
- Notes: Often combined with residual connections and gated activations in production WaveNet implementations.

Figure: Dilated causal convolution (WaveNet-like)
-------------------------------------------------
Below is a diagram showing multiple dilated causal convolution layers and how the receptive field grows with dilation.

![Dilated Causal Convolution (WaveNet-like)](./dilated_causal_convolution.png)
*Figure: Dilated causal convolutions with increasing dilation factors (1,2,4,8) — causal receptive field illustrated.*

How to train & sample (example workflow)
----------------------------------------
1. Prepare data:
   - Ensure `names.txt` (or other sequence data) is cleaned. Each example on its own line.
   - Build vocabulary from data (typically all characters present plus a special token for padding if needed).
2. Train:
   - For the Bengio notebook, run the training cells (the notebook includes a minimal training loop).
   - For a script-based conv model:
     ```
     python train.py --data data/names.txt --epochs 50 --batch-size 128 --lr 1e-3
     ```
3. Sample:
   - Use the `generate.py` script (or sampling cells in the notebook) to sample new sequences from the trained network:
     ```
     python generate.py --model checkpoints/latest.pt --seed "a" --length 20
     ```
4. Tips:
   - Start small (fewer hidden units / layers) to ensure the pipeline works.
   - Use learning rate schedules or Adam optimizer.
   - For notebook experiments, shorten epochs for faster iteration.

Dependencies
------------
- Python 3.8+
- PyTorch (or TensorFlow if you adapt code)
- numpy, tqdm, jupyter (for running the notebook)

If a `requirements.txt` is missing, create one similar to:
```
torch
numpy
tqdm
jupyter
```

Citations
---------
If you use this work or derive code from it, please cite:
- Bengio, Y., Ducharme, R., Vincent, P., & Jauvin, C. (2003). A Neural Probabilistic Language Model.
- Van den Oord, A., Dieleman, S., Zen, H., et al. (2016). WaveNet: A Generative Model for Raw Audio.

Contributing / Contact
----------------------
Contributions, bug reports, and enhancements are welcome. Open an issue or submit a pull request. For questions you can reach out to the repository owner: Moaaz-Hammad (GitHub).

License
-------
Specify a license for the repository (e.g., MIT). If you want the project to be open source under MIT, add an `LICENSE` file with the MIT text.

Acknowledgements
----------------
This repository is an educational exploration inspired by the original NPLM paper and WaveNet model. Visual diagrams are included for reference.
```
