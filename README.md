## Makemore Project Architectures: A Comparative Overview 💡

This text is a description of a project, likely a GitHub repository, structured into two main sections, each detailing a different implementation of a character-level language model for the "Makemore" task.

---

## 1. 🧠 Makemore with Bengio's Neural Probabilistic Language Model (NPLM) 

This section details an implementation of the **character-level language model** using the foundational architecture from the influential paper "**A Neural Probabilistic Language Model**" by Bengio, Ducharme, Vincent, and Jauvin (2003).

### Goal and Context
The project serves as an **educational exercise** to explore the roots of deep learning for sequence modeling and understand how modern techniques (like **Batch Normalization**) can be applied to classic architectures.

### Model Architecture (NPLM)
The model follows the core structure of the 2003 paper, predicting the **next character** based on a fixed-size context (an **N-gram window**).

* **Key Components:** An **embedding layer** and a **feed-forward hidden layer**.
* **Visual Aid:** 

### 🚀 Project Structure
* `Makemore using BENGIO, DUCHARME, VINCENT AND JAUVIN.ipynb`: The **main Jupyter Notebook** containing all code: data loading, vocabulary construction, the PyTorch NPLM implementation, the training loop, and the final name generation/sampling logic.
* `BENGIO, DUCHARME, VINCENT AND JAUVIN.png`: A diagram or reference image related to the **NPLM architecture**.
* `names.txt`: The assumed **dataset of names** used for training the model.

---

## 2. 🌊 Makemore with Dilated Causal Convolutions (WaveNet-like Architecture) 

This repository explores an implementation of a **Dilated Causal Convolutional Network**, an advanced architecture famously used in models like Google DeepMind's **WaveNet** for high-fidelity audio generation.

### Goal and Context
The project aims to illustrate how **dilated convolutions** enable a **wide receptive field** with fewer layers, maintaining **causality** and **efficiency** for sequence prediction.

### 🏛️ Model Architecture: Dilated Causal Convolutions
The network is designed for efficient sequence modeling with several key characteristics:

| Characteristic | Description |
| :--- | :--- |
| **Causality** | Predictions for a given timestep depend **only on past timesteps**, crucial for generation tasks (e.g., generating the next word or audio sample). |
| **Dilation** | Convolutional filters skip input values with a certain step, allowing the **receptive field** to grow **exponentially with depth** without increased computational cost or loss of resolution. |
| **Stacked Layers** | Multiple layers with increasing dilation rates ensure the top layer can capture **long-range dependencies** in the sequence. |

* **Visual Aid:** 

### 🚀 Project Structure (Hypothetical)
* `dilated_conv_model.py`: The core file with the PyTorch/TensorFlow implementation of the **Dilated Causal Convolutional layers** and network architecture.
* `train.py`: Script for **training** the model.
* `generate.py`: Script for **sampling** new sequences.
* `data/`: Directory for storing input **sequence data**.
* `dilated_causal_convolution.png`: The diagram illustrating the network architecture.
