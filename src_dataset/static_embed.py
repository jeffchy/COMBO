import torch
from torch import nn
import sys
sys.path.append('../')
from src_dataset.span_embed import span_reps_static

class StaticEmbedder(nn.Module):
    
    def __init__(self, emb, args):
        super(StaticEmbedder, self).__init__()
        self.V, self.D = emb.shape
        self.args = args
        if self.args.embedding == 'random':
            self.emb = nn.Embedding(self.V, self.D)
        else:
            self.emb = nn.Embedding.from_pretrained(torch.from_numpy(emb).float())

    def forward(self, idx):
        emb = self.emb(idx).view(-1, self.D)
        emb = span_reps_static(emb, self.args.rep_strategy)
        return emb