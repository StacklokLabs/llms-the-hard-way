# Building a Simple LLM with PyTorch

This repository contains code and tutorials for building a simple Large Language Model (LLM) using PyTorch. The project is structured as an educational series that walks through the process of creating, training, and fine-tuning a basic LLM using the Wikitext-2 dataset.

## Project Structure

```
llm-pytorch/
│
├── notebooks/              # Jupyter notebooks for step-by-step tutorials
│   └── 01_Setup_and_Data_Preprocessing.ipynb
│
├── src/                   # Source code for the LLM implementation
│   ├── __init__.py
│   ├── data.py           # Data loading and preprocessing
│   ├── model.py          # LLM model architecture
│   ├── trainer.py        # Training loop and utilities
│   └── inference_pipeline.py  # Text generation pipeline
│
├── README.md
└── requirements.txt
```

## Dataset

This project uses the Wikitext-2 dataset, a collection of good-quality Wikipedia articles. We chose this dataset because:
- It's a manageable size (~2M words)
- Contains well-written, coherent text suitable for language modeling
- Includes a diverse range of topics
- Has built-in train/validation/test splits

### Dataset Details
- Source: [WikiText-2](https://blog.einstein.ai/the-wikitext-long-term-dependency-language-modeling-dataset/)
- Size: ~2 million words
- Format: Raw text files
- Splits: Train (2,088,628 words), Valid (217,646 words), Test (245,569 words)

### Getting the Dataset

1. **Direct Download**
```bash
# Create data directory
mkdir -p data/wikitext-2

# Download dataset files
wget https://s3.amazonaws.com/research.metamind.io/wikitext/wikitext-2-v1.zip
unzip wikitext-2-v1.zip
mv wikitext-2/wiki.* data/wikitext-2/
rm -r wikitext-2
rm wikitext-2-v1.zip
```

2. **Using HuggingFace Datasets**
```python
from datasets import load_dataset

# Load the dataset
dataset = load_dataset("wikitext", "wikitext-2-raw-v1")

# Dataset structure:
# - train: 36,718 examples
# - validation: 3,760 examples
# - test: 4,358 examples
```

### Data Preprocessing Steps

1. **Text Cleaning**
   - Remove HTML tags and special characters
   - Normalize whitespace
   - Handle special tokens (e.g., [START], [END])

2. **Tokenization**
   - Use BytePair Encoding (BPE) tokenization
   - Build vocabulary from training data
   - Convert text to token IDs

3. **Dataset Creation**
   - Create sliding windows of fixed size (e.g., 512 tokens)
   - Generate input-target pairs for language modeling
   - Apply padding and create attention masks

Detailed implementation of these steps is available in `notebooks/01_Setup_and_Data_Preprocessing.ipynb`.

## Getting Started

### Prerequisites

- Python 3.8+
- PyTorch 2.0+
- CUDA-capable GPU (recommended)

### Installation

1. Clone the repository:
```bash
git clone https://github.com/yourusername/llm-pytorch.git
cd llm-pytorch
```

2. Create a virtual environment (recommended):
```bash
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
```

3. Install dependencies:
```bash
pip install -r requirements.txt
```

### Initial Setup

1. Install required packages:
```bash
pip install torch torchvision torchaudio
pip install transformers datasets numpy pandas jupyter
```

2. Download and prepare the dataset:
```bash
# Run the data preparation script
python src/data.py --download --prepare
```

3. Start Jupyter notebook:
```bash
jupyter notebook notebooks/01_Setup_and_Data_Preprocessing.ipynb
```

## Implementation Details

### Data Preprocessing (`data.py`)
- Text cleaning and normalization
- BPE tokenization using HuggingFace's tokenizers
- Dataset creation with sliding windows
- Efficient data loading with PyTorch DataLoader

### Model Architecture (`model.py`)
- Transformer-based decoder-only architecture
- Custom embedding layer with learned positional embeddings
- Multi-head self-attention mechanism
- Position-wise feed-forward networks
- Layer normalization and residual connections

### Training (`trainer.py`)
- Training loop with gradient accumulation
- Learning rate scheduling
- Mixed-precision training
- Checkpointing and model saving
- Training metrics logging

### Inference Pipeline (`inference_pipeline.py`)
- Text generation with temperature control
- Top-k and nucleus (top-p) sampling
- Batch generation support
- Generation with sliding context window

## Training Process

The training process is divided into several phases:

1. **Pretraining**
   - Train on the full WikiText-2 dataset
   - Use masked language modeling objective
   - Monitor perplexity and loss metrics

2. **Fine-tuning**
   - Adapt the model for specific tasks
   - Implement gradient checkpointing for memory efficiency
   - Use smaller learning rates

3. **Evaluation**
   - Calculate perplexity on test set
   - Generate sample texts
   - Compare with baseline models

## Contributing

Contributions are welcome! Please feel free to submit a Pull Request.

## License

This project is licensed under the MIT License - see the LICENSE file for details.

## Acknowledgments

- WikiText-2 dataset creators
- PyTorch team
- HuggingFace team for their excellent transformers library