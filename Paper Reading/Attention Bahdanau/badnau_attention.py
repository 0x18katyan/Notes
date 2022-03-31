import torch
import torch.nn as nn
import torch.nn.functional as F


class BadhanauAttention(nn.Module):
    
    def __init__(self, hidden_dim):
        self.alignment_nn = nn.Linear(in_features = hidden_dim * 2, out_features = hidden_dim)
        self.v_t = nn.Linear(in_features = hidden_dim, out_features = 1)
    
    def forward(self, encoder_hidden_states, decoder_hidden_state):
        
        ##score is the e_ij
        
        score = F.tanh(self.alignment_nn(torch.concat(encoder_hidden_states, decoder_hidden_state, dim = -1)))
        score = self.v_t(score).squeeze(-1)
        
        context_vector = F.softmax(score, dim = -1)
        return score, context_vector
    

