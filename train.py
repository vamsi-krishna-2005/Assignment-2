import torch
import torch.nn as nn
from torch.utils.data import DataLoader, random_split
import time
import math
import os
import argparse
import matplotlib.pyplot as plt

# Import our custom modules
from src.dataset import load_data, preprocess_text, build_vocab, TextDataset
from src.model import LSTMModel, GRUModel

def train_epoch(model, dataloader, optimizer, criterion, device):
    """Performs one training epoch."""
    model.train()
    total_loss = 0
    
    # Initialize hidden state
    hidden = model.init_hidden(dataloader.batch_size, device)

    for inputs, targets in dataloader:
        if inputs.shape[0] != dataloader.batch_size:
            continue
            
        inputs, targets = inputs.to(device), targets.to(device)
        
        # Detach hidden state
        if isinstance(hidden, tuple): # For LSTM
            hidden = tuple([h.detach() for h in hidden])
        else: # For GRU
            hidden = hidden.detach()
        
        model.zero_grad()
        output, hidden = model(inputs, hidden)
        
        loss = criterion(output, targets)
        loss.backward()
        nn.utils.clip_grad_norm_(model.parameters(), 5) # Gradient clipping
        optimizer.step()
        
        total_loss += loss.item()
        
    return total_loss / len(dataloader)

def evaluate(model, dataloader, criterion, device):
    """Performs one validation epoch."""
    model.eval()
    total_loss = 0
    
    hidden = model.init_hidden(dataloader.batch_size, device)
    
    with torch.no_grad():
        for inputs, targets in dataloader:
            if inputs.shape[0] != dataloader.batch_size:
                continue
                
            inputs, targets = inputs.to(device), targets.to(device)
            if isinstance(hidden, tuple):
                hidden = tuple([h.detach() for h in hidden])
            else:
                hidden = hidden.detach()
            
            output, hidden = model(inputs, hidden)
            loss = criterion(output, targets)
            total_loss += loss.item()
            
    return total_loss / len(dataloader)

def main(args):
    """Main training and evaluation function."""
    
    # --- Setup ---
    print(f"Starting run: {args.run_name}")
    print(f"Using device: {args.device}")
    
    # Create output directory
    output_dir = os.path.join("runs", args.run_name)
    os.makedirs(output_dir, exist_ok=True)
    checkpoint_path = os.path.join(output_dir, "best_model.pth")
    
    # --- Data Loading ---
    print("Loading and preprocessing data...")
    raw_text = load_data(args.data_file)
    all_tokens = preprocess_text(raw_text, "CHAPTER I.", "***END OF THE PROJECT GUTENBERG EBOOK PRIDE AND PREJUDICE***")
    word_to_idx, idx_to_word = build_vocab(all_tokens, args.vocab_size)
    
    dataset = TextDataset(all_tokens, word_to_idx, sequence_length=args.seq_len)
    
    # Split
    train_size = int(0.8 * len(dataset))
    val_size = len(dataset) - train_size
    train_dataset, val_dataset = random_split(dataset, [train_size, val_size])

    print(f"Total sequences: {len(dataset)}")
    print(f"Training sequences: {len(train_dataset)}")
    print(f"Validation sequences: {len(val_dataset)}")

    train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True, drop_last=True)
    val_loader = DataLoader(val_dataset, batch_size=args.batch_size, shuffle=False, drop_last=True)

    # --- Model Initialization ---
    if args.model_type.lower() == 'lstm':
        model = LSTMModel(
            vocab_size=len(word_to_idx),
            embedding_dim=args.embed_dim,
            hidden_dim=args.hidden_dim,
            num_layers=args.num_layers,
            dropout_prob=args.dropout
        ).to(args.device)
    elif args.model_type.lower() == 'gru':
        model = GRUModel(
            vocab_size=len(word_to_idx),
            embedding_dim=args.embed_dim,
            hidden_dim=args.hidden_dim,
            num_layers=args.num_layers,
            dropout_prob=args.dropout
        ).to(args.device)
    else:
        raise ValueError("Invalid model type specified. Choose 'lstm' or 'gru'.")
        
    print(model)
    print(f"Total parameters: {sum(p.numel() for p in model.parameters())}")

    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)
    
    # --- Training Loop ---
    print("\n--- Starting Training ---")
    best_val_loss = float('inf')
    train_losses = []
    val_losses = []

    for epoch in range(1, args.epochs + 1):
        start_epoch_time = time.time()
        
        train_loss = train_epoch(model, train_loader, optimizer, criterion, args.device)
        val_loss = evaluate(model, val_loader, criterion, args.device)
        
        train_losses.append(train_loss)
        val_losses.append(val_loss)
        
        train_ppl = math.exp(train_loss)
        val_ppl = math.exp(val_loss)
        
        epoch_time = time.time() - start_epoch_time
        epoch_mins, epoch_secs = divmod(epoch_time, 60)
        
        print(f'Epoch: {epoch:02}/{args.epochs} | Time: {epoch_mins:.0f}m {epoch_secs:.0f}s')
        print(f'\tTrain Loss: {train_loss:.3f} | Train PPL: {train_ppl:7.3f}')
        print(f'\t Val. Loss: {val_loss:.3f} |  Val. PPL: {val_ppl:7.3f}')
        
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            torch.save(model.state_dict(), checkpoint_path)
            print(f'\tNew best validation loss. Model saved to {checkpoint_path}')

    print(f"\n--- Training Complete ---")
    print(f"Best validation perplexity: {math.exp(best_val_loss):.3f}")

    # --- Save Plot ---
    plt.figure(figsize=(10, 6))
    plt.plot(train_losses, label='Training Loss')
    plt.plot(val_losses, label='Validation Loss')
    plt.title(f'Training vs. Validation Loss ({args.run_name})')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.legend()
    plot_path = os.path.join(output_dir, f"{args.run_name}_loss_plot.png")
    plt.savefig(plot_path)
    print(f"Loss plot saved to {plot_path}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train a Neural Language Model")
    
    # --- General ---
    parser.add_argument('--run_name', type=str, default='run_default', help='Name for this training run (creates a folder)')
    parser.add_argument('--data_file', type=str, default='data/Pride_and_Prejudice-Jane_Austen.txt')
    
    # --- Model Hyperparameters ---
    parser.add_argument('--model_type', type=str, default='lstm', help='Model architecture (lstm or gru)')
    parser.add_argument('--vocab_size', type=int, default=7000, help='Max number of words in vocab')
    parser.add_argument('--embed_dim', type=int, default=128, help='Embedding dimension')
    parser.add_argument('--hidden_dim', type=int, default=256, help='LSTM/GRU hidden dimension')
    parser.add_argument('--num_layers', type=int, default=2, help='Number of LSTM/GRU layers')
    parser.add_argument('--dropout', type=float, default=0.3, help='Dropout probability')
    
    # --- Training Hyperparameters ---
    parser.add_argument('--epochs', type=int, default=20, help='Number of epochs to train')
    parser.add_argument('--batch_size', type=int, default=64, help='Batch size')
    parser.add_argument('--seq_len', type=int, default=30, help='Sequence length')
    parser.add_argument('--lr', type=float, default=0.001, help='Learning rate')

    args = parser.parse_args()
    
    # Set device
    args.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    main(args)