# Assignment 2: Neural Language Model Training (PyTorch)

A PyTorch implementation of LSTM and GRU neural language models trained on Jane Austen's "Pride and Prejudice" to predict the next word in a sequence.

## 🚀 Key Results

| Model | Architecture | Best Validation Perplexity |
|-------|--------------|---------------------------|
| `lstm-v2` | 2-Layer LSTM (256 hidden) | **120.55** |
| `gru-v1` | 2-Layer GRU (256 hidden) | **123.81** |
| `lstm-v1` | 1-Layer LSTM (64 hidden) | **143.17** |

## 📂 Project Structure

```
Assignment2/
├── dataset/
│   └── Pride_and_Prejudice-Jane_Austen.txt
├── notebooks/
│   └── Assignment_2.ipynb
├── src/
│   ├── dataset.py
│   ├── model.py
│   └── train.py
├── requirements.txt
└── README.md
```

## Installation & Setup

### Option 1: Local Setup

```bash
# Clone repository
git clone https://github.com/vamsi-krishna-2005/Assignment-2.git
cd Assignment2

# Install dependencies
pip install -r requirements.txt

# Verify installation
python -c "import torch; print(f'PyTorch: {torch.__version__}'); print(f'CUDA: {torch.cuda.is_available()}')"
```

### Option 2: Google Colab (Recommended for Free GPU)

```python
# Cell 1: Clone and setup
!git clone https://github.com/vamsi-krishna-2005/Assignment-2.git
%cd Assignment-2
!pip install -r requirements.txt

# Cell 2: Verify GPU
import torch
print(f"GPU: {torch.cuda.is_available()}")
print(f"Device: {torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'CPU'}")
```

## Training Commands

### Underfitting Scenario (5 epochs, small model)
```bash
python train.py --run_name "lstm-v1" --model_type lstm --epochs 5 --hidden_dim 64 --num_layers 1
```

### Overfitting & Best Fit Scenario (20 epochs, full capacity)
```bash
python train.py --run_name "lstm-v2" --model_type lstm --epochs 20
```

### Alternative Model: GRU 
```bash
python train.py --run_name "gru-v1" --model_type gru --epochs 20
```

### Custom Training
```bash
python train.py --run_name "custom-model" \
  --model_type lstm \
  --epochs 10 \
  --hidden_dim 512 \
  --embed_dim 256 \
  --batch_size 32 \
  --lr 0.0005
```

## 📊 Training Scenarios

The project demonstrates three key concepts in neural networks:

### Scenario 1: Underfitting (`lstm-v1`)

- Model: 1-layer LSTM with 64 hidden units
- Training: Only 5 epochs
- Result: Both training and validation losses remain high
- Shows: Model lacks capacity to learn complex patterns

### Scenario 2: Overfitting (`lstm-v2`)

- Model: 2-layer LSTM with 256 hidden units
- Training: 20 epochs with full capacity
- Result: Training loss decreases, validation loss increases after epoch 4
- Shows: Model memorizes training data instead of generalizing
- **Best Fit**: Automatically saved at epoch with lowest validation loss (perplexity: 120.55)

### Scenario 3: GRU Comparison (`gru-v1`)

- Model: 2-layer GRU with 256 hidden units
- Training: 20 epochs
- Result: Similar performance to LSTM (perplexity: 123.81) with fewer parameters
- Shows: GRU is an efficient alternative to LSTM

| Argument | Type | Default | Description |
|----------|------|---------|-------------|
| `--run_name` | str | `run_default` | Name for the training run (creates output folder) |
| `--model_type` | str | `lstm` | Model architecture: `lstm` or `gru` |
| `--vocab_size` | int | 7000 | Maximum vocabulary size |
| `--embed_dim` | int | 128 | Embedding dimension |
| `--hidden_dim` | int | 256 | LSTM/GRU hidden dimension |
| `--num_layers` | int | 2 | Number of LSTM/GRU layers |
| `--dropout` | float | 0.3 | Dropout probability (prevents overfitting) |
| `--epochs` | int | 20 | Number of training epochs |
| `--batch_size` | int | 64 | Batch size for training |
| `--seq_len` | int | 30 | Sequence length (context window) |
| `--lr` | float | 0.001 | Learning rate for Adam optimizer |
| `--data_file` | str | `dataset/Pride_and_Prejudice-Jane_Austen.txt` | Path to training data |

### Quick Reference: Common Configurations

#### Example 1: Quick LSTM Training (5 epochs) - Underfitting
```bash
python train.py --run_name "lstm-v1" --model_type lstm --epochs 5 --hidden_dim 64 --num_layers 1
```

#### Example 2: Full LSTM Training (20 epochs) - Overfitting & Best Fit
```bash
python train.py --run_name "lstm-v2" --model_type lstm --epochs 20
```

#### Example 3: GRU Model Training (20 epochs) - Extra Credit
```bash
python train.py --run_name "gru-v1" --model_type gru --epochs 20
```

#### Example 4: Custom Hyperparameters
```bash
python train.py --run_name "custom-model" --model_type lstm --epochs 10 --hidden_dim 512 --embed_dim 256 --batch_size 32 --lr 0.0005
```

#### Example 5: Minimal Resources (CPU-friendly)
```bash
python train.py --run_name "small-model" --model_type gru --epochs 10 --hidden_dim 128 --num_layers 1 --batch_size 32
```

## Running from Jupyter Notebook

You can also run the training pipeline from the Jupyter notebook (`notebooks/Assignment_2.ipynb`):

1. Open the notebook in Jupyter or JupyterLab:
```bash
jupyter notebook notebooks/Assignment_2.ipynb
```

2. Execute cells in order:
   - **Cell 1**: Clone the repository (if needed)
   - **Cell 2**: Navigate to project directory
   - **Cell 3**: Install dependencies
   - **Cell 4-6**: Run different training configurations (lstm-v1, lstm-v2, gru-v1)

### Using in Google Colab

If using Colab, simply copy each training command into separate cells and run them sequentially. Colab's GPU will automatically accelerate the training.

## Output & Results

- **best_model.pth** - Best model checkpoint (lowest validation loss)
- **<run_name>_loss_plot.png** - Training and validation loss plot

### Example Output Structure
```
runs/
├── lstm-v1/
│   ├── best_model.pth
│   └── lstm-v1_loss_plot.png
├── lstm-v2/
│   ├── best_model.pth
│   └── lstm-v2_loss_plot.png
└── gru-v1/
    ├── best_model.pth
    └── gru-v1_loss_plot.png
```

### Understanding the Output

**Loss Plot Analysis:**
- **Training Loss** (blue line): Loss on data the model has seen
- **Validation Loss** (orange line): Loss on held-out data
- **Divergence**: If validation loss > training loss, model is overfitting
- **Best Epoch**: The epoch with the lowest validation loss (saved automatically)

## Model Architecture Details

### LSTM (Long Short-Term Memory)
- **Structure**: Memory cells with input, forget, and output gates
- **Advantages**: Captures long-term dependencies, handles vanishing gradients
- **Use Case**: Text generation, language modeling
- **Parameters**: Embedding → LSTM layers → Fully Connected layer

### GRU (Gated Recurrent Unit)
- **Structure**: Simpler than LSTM, combines forget and input gates
- **Advantages**: Faster training, fewer parameters, similar performance
- **Use Case**: Same as LSTM, but more efficient
- **Parameters**: Embedding → GRU layers → Fully Connected layer

### Model Components

```
Input Sequence (batch_size, seq_len)
    ↓
Embedding Layer (vocab_size → embed_dim)
    ↓
LSTM/GRU Layer(s) (embed_dim → hidden_dim)
    ↓
Dropout Layer (regularization)
    ↓
Fully Connected (hidden_dim → vocab_size)
    ↓
Output Logits (batch_size, vocab_size)
```

## Training Process

1. **Data Loading**: Loads Pride and Prejudice text (~122K words)
2. **Preprocessing**: Tokenizes, lowercases, removes special characters
3. **Vocabulary Building**: Creates word-to-index mappings (7000 most common words)
4. **Dataset Creation**: Creates (30-word sequence, next word) pairs (~75K pairs)
5. **Train/Validation Split**: 80% training, 20% validation
6. **Training Loop** (per epoch):
   - Forward pass: sequence → embeddings → LSTM/GRU → output
   - Compute loss: Cross-entropy between predicted and actual next word
   - Backward pass: Compute gradients
   - Gradient clipping: Prevent exploding gradients (max norm = 5)
   - Optimization: Adam optimizer updates weights
7. **Validation**: Evaluate on held-out data without updating weights
8. **Checkpointing**: Save best model (lowest validation loss)
9. **Visualization**: Generate loss plot

## Evaluation Metrics

### Perplexity (PPL)
- **Formula**: PPL = exp(loss)
- **Interpretation**: On average, model is "surprised" by 1 out of PPL words
- **Example**: PPL of 120 means model is as confused as if choosing among 120 equally likely words
- **Lower is better**: Indicates better generalization

### Loss (Cross-Entropy)
- Measures difference between predicted probability distribution and actual next word
- Training loss should decrease (model learning)
- Validation loss should decrease initially, then plateau or increase (overfitting)

## Tips for Training

1. **Start Small**: Begin with fewer epochs to test setup (5-10 epochs)
2. **Monitor Loss**: 
   - Healthy: Training loss decreases, validation loss follows
   - Underfitting: Both losses are high and decreasing slowly
   - Overfitting: Training loss decreases, validation loss increases
3. **Adjust Hyperparameters**:
   - **Learning Rate**: If loss doesn't improve, try 0.0005 or 0.0001
   - **Batch Size**: Larger (128) = faster but noisier; Smaller (32) = slower but smoother
   - **Hidden Dimension**: Larger = more capacity but more computation
   - **Dropout**: Increase to 0.5 to reduce overfitting
4. **GPU Usage**: 
   - Check: `torch.cuda.is_available()` and `torch.cuda.get_device_name(0)`
   - CUDA 10x faster than CPU for RNNs
5. **Save Best Model**: Script automatically saves checkpoint with lowest validation loss

## Troubleshooting

### Installation Issues

**"ModuleNotFoundError: No module named 'torch'"**
```bash
pip install torch
```

**"ModuleNotFoundError: No module named 'numpy'"**
```bash
pip install numpy matplotlib tqdm
```

### Runtime Issues

**"FileNotFoundError: dataset/Pride_and_Prejudice-Jane_Austen.txt"**
- Ensure you're running the script from the project root directory
- Check that the dataset folder exists and contains the file

**"CUDA out of memory"**
- Reduce batch size: `--batch_size 32`
- Reduce hidden dimension: `--hidden_dim 128`
- Reduce sequence length: `--seq_len 20`
- Switch to CPU (slower but will work)

**"CUDA out of memory" even with small batch**
- Fall back to CPU training
- Use `torch.cuda.empty_cache()` (clears GPU memory)
- Restart Colab kernel

### Performance Issues

**"Training is very slow"**
- Check GPU usage: `torch.cuda.is_available()`
- In Colab, enable GPU: Runtime → Change runtime type → GPU
- Reduce model size: `--hidden_dim 128 --num_layers 1`
- Increase batch size (uses more GPU memory but faster training)

**"Loss is NaN or Inf"**
- Learning rate too high: Try `--lr 0.0001`
- Gradient clipping is in place, but try reducing learning rate first
- Ensure data file path is correct

### Colab-Specific

**"No GPU available in Colab"**
1. Click Runtime in menu bar
2. Select "Change runtime type"
3. Select GPU as hardware accelerator
4. Click Save

**"Can't find runs folder after training in Colab"**
```python
# Download all results
import shutil
shutil.make_archive('results', 'zip', '.')
from google.colab import files
files.download('results.zip')
```

## Project Workflow

### Recommended Development Workflow

1. **Local Development**:
   - Edit `.py` files in VSCode
   - Test with small configs locally (or CPU)
   - Commit to GitHub

2. **Cloud Training**:
   - Clone repo in Google Colab
   - Run full training on GPU
   - Download results

3. **Analysis & Results**:
   - Load plots and metrics from `runs/` folder
   - Create analysis notebook with saved results
   - Document findings

### File Organization Best Practice

```
# Keep modular Python code in src/
src/
├── dataset.py        # Data utilities
├── model.py          # Model definitions
└── train.py          # Training script

# Output automatically organized by run
runs/
├── lstm-v1/          # Each run is isolated
├── lstm-v2/
└── gru-v1/

# Notebooks for analysis and documentation
notebooks/
└── Assignment_2.ipynb
```

This separation keeps code clean and results reproducible.



