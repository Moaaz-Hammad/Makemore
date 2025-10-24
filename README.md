🧠 Makemore with Bengio's Neural Probabilistic Language Model (NPLM)
This repository implements a character-level language model to generate names (the "Makemore" task), specifically using the foundational architecture described in the influential paper "A Neural Probabilistic Language Model" by Bengio, Ducharme, Vincent, and Jauvin (2003).
The project serves as an educational exercise to explore the roots of deep learning for sequence modeling and understand how modern techniques (like Batch Normalization) can be applied to classic architectures.
🚀 Project Structure
•	Makemore using BENGIO, DUCHARME, VINCENT AND JAUVIN.ipynb: The main Jupyter Notebook. This contains all the code, including data loading, vocabulary construction, the PyTorch implementation of the NPLM (including its embedding layer and feed-forward hidden layer), the training loop, and the final name generation/sampling logic.
•	BENGIO, DUCHARME, VINCENT AND JAUVIN.png: A diagram or reference image related to the NPLM architecture from the original paper.
•	names.txt: (Assumed) The dataset of names used for training the model.
🏛️ Model Architecture (NPLM)
The model follows the core structure of the 2003 paper, predicting the next character based on a fixed-size context (an N-gram window).
Diagram of the NPLM Architecture: 
 
🌊 Makemore with Dilated Causal Convolutions (WaveNet-like Architecture)
This repository provides an implementation and exploration of a Dilated Causal Convolutional Network, an architecture famously utilized in models like Google DeepMind's WaveNet for high-fidelity audio generation, and also relevant for advanced sequence modeling tasks in natural language processing.
The project aims to illustrate how dilated convolutions enable a wide receptive field with fewer layers, maintaining causality and efficiency for sequence prediction.
🚀 Project Structure (Hypothetical)
•	dilated_conv_model.py: (Assumed) The core Python file containing the PyTorch (or TensorFlow) implementation of the Dilated Causal Convolutional layers and the overall network architecture.
•	train.py: (Assumed) Script for training the model on a given sequence dataset.
•	generate.py: (Assumed) Script for sampling new sequences from the trained model.
•	data/: (Assumed) Directory for storing input sequence data (e.g., character sequences, audio samples, time series).
•	dilated_causal_convolution.png: The diagram illustrating the network architecture.
🏛️ Model Architecture: Dilated Causal Convolutions
The core of this project is the Dilated Causal Convolutional Network. This architecture is designed for efficient sequence modeling with several key characteristics:
•	Causality: Predictions for a given timestep depend only on past timesteps, not future ones. This is crucial for generation tasks (e.g., generating the next word or audio sample).
•	Dilation: Convolutional filters are applied over an area larger than their length by skipping input values with a certain step. This allows the receptive field of the network to grow exponentially with depth without a proportional increase in computational cost or loss of resolution, unlike pooling layers.
•	Stacked Layers: Multiple layers with increasing dilation rates allow the top layer to have a very wide receptive field, capturing long-range dependencies in the sequence.
•	Visual Representation of Dilated Causal Convolutions:


