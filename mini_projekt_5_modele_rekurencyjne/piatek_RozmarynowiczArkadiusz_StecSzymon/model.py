import torch
import torch.nn as nn

class LSTMComposerClassifier(nn.Module):
    def __init__(self, vocab_size, embedding_dim, hidden_size, num_layers, out_size):
        super().__init__()
        self.num_layers = num_layers
        self.hidden_size = hidden_size
        
        self.embedding = nn.Embedding(vocab_size, embedding_dim, padding_idx=2)
        self.lstm = nn.LSTM(input_size=embedding_dim, hidden_size=hidden_size, num_layers=num_layers)
        self.fc = nn.Linear(hidden_size, out_size)
        
    def init_hidden(self, batch_size, device):
        hidden = torch.zeros(self.num_layers, batch_size, self.hidden_size).to(device)
        state = torch.zeros(self.num_layers, batch_size, self.hidden_size).to(device)
        return hidden, state
    
    def forward(self, x, hidden_state):
        x = self.embedding(x)
        x = torch.transpose(x, 0, 1)
        all_outputs, hidden_state = self.lstm(x, hidden_state)
        out = all_outputs[-1] 
        x = self.fc(out)
        return x, hidden_state

