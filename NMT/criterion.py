import torch 
import torch.nn as nn

class LabelSmoothing(nn.Module):
    "Implement label smoothing."
    def __init__(self, size, padding_idx, smoothing=0.0):
        super(LabelSmoothing, self).__init__()
        self.criterion = nn.KLDivLoss(reduction='sum')
        self.padding_idx = padding_idx
        self.confidence = 1.0 - smoothing
        self.smoothing = smoothing
        self.size = size
        self.true_dist = None
        
    def forward(self, x, target):
        #print(x.size(1), self.size)
        assert x.size(1) == self.size
        true_dist = x.data.clone()
        true_dist.fill_(self.smoothing / (self.size - 2))
        true_dist.scatter_(1, target.data.unsqueeze(1), self.confidence)
        true_dist[:, self.padding_idx] = 0
        mask = torch.nonzero(target.data == self.padding_idx)
        if mask.dim() > 0:
            true_dist.index_fill_(0, mask.squeeze(), 0.0)
        self.true_dist = true_dist
        true_dist.requires_grad = False
        return self.criterion(x, true_dist)

class SimpleLossCompute(nn.Module):
    "A simple loss compute and train function."
    def __init__(self, generator, criterion, opt=None):
        super(SimpleLossCompute, self).__init__()
        self.generator = generator
        self.criterion = criterion
        self.opt = opt
        
    def forward(self, x, y, norm):
        x = self.generator(x)
        loss = self.criterion(x.contiguous().view(-1, x.size(-1)), 
                              y.contiguous().view(-1)) / norm
        #loss.backward()
        #if self.opt is not None:
        #    self.opt.step()
        #    self.opt.optimizer.zero_grad()
        
        return loss
