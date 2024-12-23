import torch
import torch.nn as nn
import random



class Similarity(nn.Module):
    """
    Dot product or cosine similarity
    """

    def __init__(self, temp):
        super().__init__()
        self.temp = temp
        self.cos = nn.CosineSimilarity(dim=-1)

    def forward(self, x, y):
        return self.cos(x, y) / self.temp
    
class ContrastiveLoss(nn.Module):
    def __init__(self, temp=0.5):
        super().__init__()
        self.loss = nn.CrossEntropyLoss()
        self.sim_func = Similarity(temp=temp)

    def forward(self, emb1, emb2):
        batch_sim = self.sim_func(emb1.unsqueeze(1), emb2.unsqueeze(0))
        labels = torch.arange(batch_sim.size(0)).long().to('cuda')
        return self.loss(batch_sim, labels)

