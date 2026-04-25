# Banking Intent Classification with Unsloth

Fine-tuning a Large Language Model for intent detection on the BANKING77 dataset using [Unsloth](https://unsloth.ai/).

## Project Structure

```
banking-intent-unsloth/
├── scripts/
│   ├── preprocess_data.py    # Data preparation and preprocessing
│   ├── train.py              # Model fine-tuning with Unsloth
│   └── inference.py          # IntentClassification class for inference
├── configs/
│   ├── train.yaml            # Training hyperparameters configuration
│   └── inference.yaml        # Inference configuration
├── sample_data/
│   ├── train.csv             # Preprocessed training data
│   └── test.csv              # Preprocessed test data
├── train.sh                  # Script to run training pipeline
├── inference.sh              # Script to run inference
├── requirements.txt          # Python dependencies
└── README.md                 # This file
```

## Requirements

- Python 3.10+
- NVIDIA GPU with at least 8GB VRAM (recommended: T4 or higher)
- CUDA 11.8+

## Setup

### 1. Clone the repository

```bash
git clone https://github.com/<your-username>/banking-intent-unsloth.git
cd banking-intent-unsloth
```

### 2. Install dependencies

```bash
pip install -r requirements.txt
```

### 3. Preprocess data

```bash
python scripts/preprocess_data.py --config configs/train.yaml
```

### 4. Train model

```bash
bash train.sh
```

Or run directly:

```bash
python scripts/train.py --config configs/train.yaml
```

### 5. Run inference

```bash
bash inference.sh
```

Or use in Python:

```python
from scripts.inference import IntentClassification

classifier = IntentClassification("configs/inference.yaml")
result = classifier("I want to top up my account")
print(f"Predicted intent: {result}")
```

## Model Details

- **Base model**: `unsloth/Llama-3.2-1B-Instruct-bnb-4bit`
- **Fine-tuning method**: LoRA (Low-Rank Adaptation) with 4-bit quantization
- **Dataset**: [BANKING77](https://huggingface.co/datasets/PolyAI/banking77)

## Hyperparameters

| Parameter | Value |
|---|---|
| LoRA rank (r) | 16 |
| LoRA alpha | 16 |
| Learning rate | 2e-4 |
| Batch size | 4 |
| Gradient accumulation | 4 |
| Epochs | 3 |
| Max sequence length | 256 |
| Optimizer | AdamW 8-bit |

## Results

| Metric | Score |
|---|---|
| Test Accuracy | TBD |

## Video Demo

🎥 [Watch the demo video](https://drive.google.com/...) *(TODO: update link)*

## Author

- **Name**: TODO
- **Student ID**: TODO
- **Course**: Applications of Natural Language Processing in Industry
- **Lecturer**: Dr. Nguyen Hong Buu Long
