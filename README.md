# Character-Level-Language-Model-for-Name-Generation


## Indian Name Generator using Neural Networks (PyTorch)
A character-level language model trained to generate novel Indian-style names using a multi-layer perceptron (MLP) with learned character embeddings. Built from scratch in PyTorch without any pretrained models.

### Model Architecture

27-dimensional vocabulary (a-z + padding token)
4-dimensional character embeddings
Context window of 3 previous characters
64-neuron hidden layer with tanh activation
Output: softmax over 27 characters

### Dataset

300+ curated Indian names
Expanded to ~3,000 sequence samples via sliding window
90/10 train-validation split

### Key Results

Validation Loss: ~2.1 (Cross-Entropy)
Name Validity Rate: ~90%
Vowel/consonant clusters clearly separated in PCA embedding visualization

### Features

Autoregressive name generation with temperature sampling
PCA-based embedding visualization
Training & validation loss curve plotting
Interactive CLI name generator
Model checkpointing (saves/loads name_model.pt)

### How to Run
pip install torch matplotlib
python name_generator.py



Tech Stack: Python, PyTorch, Matplotlib, NumPy
