# Pythia: A NumPy-Only GPT

![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg) ![Python](https://img.shields.io/badge/python-3.8+-blue.svg) ![Dependencies](https://img.shields.io/badge/dependencies-numpy-green.svg)

> **Note:** This project is an educational implementation of a Transformer from scratch. It is unrelated to [EleutherAI's Pythia](https://github.com/EleutherAI/pythia) LLM suite or the [PYTHIA](https://pythia.org/) physics event generator.

**Pythia** is a minimal, clean, and hackable implementation of a Generative Pre-trained Transformer (GPT) written entirely in **Python** and **NumPy**. 

It is designed for educational purposes: to demystify how Large Language Models work by stripping away the complexity of deep learning frameworks like PyTorch or TensorFlow. 

## Features

*   **Zero Heavy Dependencies:** Runs on pure `numpy`. No `torch`, `tensorflow`, or `jax`.
*   **Readable Codebase:** The core logic is optimized for clarity over speed.
*   **From Scratch:** Implements standard Attention, Feed-Forward, and LayerNorm blocks manually.
*   **Pre-trained Compatible:** (Optional) Can load converted weights from GPT-2.

## Installation

1.  Clone the repository:
    ```bash
    git clone https://github.com/yourusername/pythia.git
    cd pythia
    ```

2.  Install NumPy:
    ```bash
    pip install numpy
    ```

## File Structure

Ensure your project directory matches the imports used in `main.py`:

```text
Pythia/
├── engine.py          # Contains DeltaOptimizer
├── model.py           # Contains Pythia model class
├── tokenizer.py       # Contains SimpleTokenizer
├── ui.py              # Contains ConsoleUI
├── main.py            # Entry point (Training + Chat)
├── input.txt          # REQUIRED: Training source text
└── model.txt          # REQUIRED: Secondary training text
```

## Usage

Pythia runs in a unified **Train-then-Chat** mode. When you run the script, it will first train on your data and then immediately launch an interactive session.

### 1. Prepare Your Data
Create two text files in the root directory:
*   `input.txt`: The main body of text you want Pythia to learn from.
*   `model.txt`: Additional context or stylistic text.

### 2. Run Pythia
Execute the main script:

```bash
python main.py
```

**What to expect:**
1.  **Boot Sequence:** The `ConsoleUI` will initialize.
2.  **Training Loop:** The model will train for 500 steps. You will see real-time loss metrics and an Estimated Time of Arrival (ETA).
3.  **Chat Mode:** Once training is complete, the interface switches to interactive mode.

### 3. Chat Controls
Inside the chat session, you can use the following commands:
*   `exit` or `quit`: Terminate the program.
*   `/clear`: Clear the console screen.

## Configuration

You can tune the model's size and training speed by editing the variables at the top of `main.py`:

```python
# Speed Config (Lightweight)
BATCH_SIZE = 8    # Lower this if you run out of memory
BLOCK_SIZE = 32   # Context length (how far back it looks)
D_MODEL = 48      # Embedding dimension (network width)
N_LAYER = 3       # Number of Transformer blocks (network depth)
STEPS = 500       # Total training iterations
```
## Architecture

Pythia implements the standard Transformer decoder architecture:

*   **Multi-Head Self Attention:** Implemented with `np.matmul` and manual softmax.
*   **GELU Activation:** Approximate implementation using `np.tanh`.
*   **Layer Normalization:** Manual calculation of mean and variance.

## Contributing

contributions are welcome! If you find a way to make the matrix multiplications faster or the code more readable, please open a PR.

1.  Fork the project
2.  Create your feature branch (`git checkout -b feature/AmazingFeature`)
3.  Commit your changes (`git commit -m 'Add some AmazingFeature'`)
4.  Push to the branch (`git push origin feature/AmazingFeature`)
5.  Open a Pull Request

## License

Distributed under the MIT License. See `LICENSE` for more information.
