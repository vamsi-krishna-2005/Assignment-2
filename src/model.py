import torch
import torch.nn as nn

class LSTMModel(nn.Module):
    """Our from-scratch LSTM Language Model."""
    def __init__(self, vocab_size, embedding_dim, hidden_dim, num_layers, dropout_prob=0.3):
        super(LSTMModel, self).__init__()
        self.num_layers = num_layers
        self.hidden_dim = hidden_dim
        
        self.embedding = nn.Embedding(vocab_size, embedding_dim)
        
        self.lstm = nn.LSTM(
            input_size=embedding_dim,
            hidden_size=hidden_dim,
            num_layers=num_layers,
            batch_first=True,
            dropout=dropout_prob if num_layers > 1 else 0
        )
        
        self.dropout = nn.Dropout(dropout_prob)
        self.fc = nn.Linear(hidden_dim, vocab_size)

    def forward(self, x, hidden):
        embeds = self.embedding(x)
        lstm_out, hidden = self.lstm(embeds, hidden)
        out = self.dropout(lstm_out)
        # We only care about the last time step
        out = self.fc(out[:, -1, :])
        return out, hidden

    def init_hidden(self, batch_size, device):
        """Initializes hidden state to zeros."""
        return (torch.zeros(self.num_layers, batch_size, self.hidden_dim).to(device),
                torch.zeros(self.num_layers, batch_size, self.hidden_dim).to(device))

class GRUModel(nn.Module):
    """Our from-scratch GRU Language Model (for extra credit)."""
    def __init__(self, vocab_size, embedding_dim, hidden_dim, num_layers, dropout_prob=0.3):
        super(GRUModel, self).__init__()
        self.num_layers = num_layers
        self.hidden_dim = hidden_dim
        
        self.embedding = nn.Embedding(vocab_size, embedding_dim)
        
        self.gru = nn.GRU(
            input_size=embedding_dim,
            hidden_size=hidden_dim,
            num_layers=num_layers,
            batch_first=True,
            dropout=dropout_prob if num_layers > 1 else 0
        )
        
        self.dropout = nn.Dropout(dropout_prob)
        self.fc = nn.Linear(hidden_dim, vocab_size)

    def forward(self, x, hidden):
        embeds = self.embedding(x)
        gru_out, hidden = self.gru(embeds, hidden)
        out = self.dropout(gru_out)
        out = self.fc(out[:, -1, :])
        return out, hidden

    def init_hidden(self, batch_size, device):
        """Initializes hidden state to zeros."""
        # GRU only has one hidden state, not two like LSTM
        return torch.zeros(self.num_layers, batch_size, self.hidden_dim).to(device)