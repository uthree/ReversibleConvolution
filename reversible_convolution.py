import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

import revtorch as rv


# input: [batch_size, channels, height, width]
# output: [batch_size, channels, height, width]
class ReversibleConv2d(nn.Module):
    def __init__(self, channels, groups=1, num_layers=1):
        super(ReversibleConv2d, self).__init__()
        blocks = nn.ModuleList()
        for i in range(num_layers):
            blocks.append(
                rv.ReversibleBlock(
                    nn.Sequential(
                        nn.Conv2d(channels, channels, 3, padding=1, groups=groups),
                        nn.GELU(),
                    ),
                    nn.Sequential(
                        nn.Conv2d(channels, channels, 3, padding=1, groups=groups),
                        nn.GELU(),
                    ),
                    split_along_dim=1
                )
            )
        self.seq = rv.ReversibleSequence(blocks)
    
    def forward(self, x):
        x = torch.repeat_interleave(x, repeats=2, dim=1)
        x = self.seq(x)
        x1, x2 = torch.chunk(x, 2, dim=1)
        x = (x1 + x2) / 2
        return x 

# input: [batch_size, channels, height, width]
# output: [batch_size, channels, height, width]
class ReversibleConvTranspose2d(nn.Module):
    def __init__(self, channels, groups=1, num_layers=1):
        super(ReversibleConvTranspose2d, self).__init__()
        blocks = nn.ModuleList()
        for i in range(num_layers):
            blocks.append(
                rv.ReversibleBlock(
                    nn.Sequential(
                        nn.Conv2d(channels, channels, 3, padding=0, groups=groups),
                        nn.GELU(),
                        nn.ConvTranspose2d(channels, channels, 3, padding=0, groups=groups),
                        nn.GELU(),
                    ),
                    nn.Sequential(
                        nn.Conv2d(channels, channels, 3, padding=0, groups=groups),
                        nn.GELU(),
                        nn.ConvTranspose2d(channels, channels, 3, padding=0, groups=groups),
                        nn.GELU(),
                    ),
                    split_along_dim=1,
                )
            )
            self.seq = rv.ReversibleSequence(blocks)
    
    def forward(self, x):
        x = torch.repeat_interleave(x, repeats=2, dim=1)
        x = self.seq(x)
        x1, x2 = torch.chunk(x, 2, dim=1)
        x = (x1 + x2) / 2
        return x
