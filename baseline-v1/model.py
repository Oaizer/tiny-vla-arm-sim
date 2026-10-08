"""Tiny language-conditioned visuomotor policy, trained from scratch on CPU."""
import torch
from torch import nn
from simulator import VOCAB

class TinyVLA(nn.Module):
    def __init__(self):
        super().__init__()
        self.vision=nn.Sequential(nn.Conv2d(3,16,5,padding=2),nn.ReLU(),
            nn.Conv2d(16,24,3,stride=2,padding=1),nn.ReLU(),
            nn.Conv2d(24,32,3,padding=1),nn.ReLU())
        self.words=nn.Embedding(len(VOCAB),16,padding_idx=0)
        self.query=nn.Sequential(nn.Linear(16,48),nn.ReLU(),nn.Linear(48,64))
        self.action_head=nn.Linear(8,32)
        self.register_buffer('offset',torch.tensor([.5,0.,0.,0.]))
        self.register_buffer('scale',torch.tensor([.3,.3,.3,.04]))

    def forward(self,image,tokens,state):
        feat=self.vision(image)
        mask=(tokens!=0).unsqueeze(-1)
        text=(self.words(tokens)*mask).sum(1)/mask.sum(1).clamp_min(1)
        query=self.query(text).reshape(-1,2,32)
        logits=torch.einsum('bqc,bchw->bqhw',query,feat)
        b,q,h,w=logits.shape
        attention=logits.flatten(2).softmax(-1)
        yy,xx=torch.meshgrid(torch.linspace(-1,1,h,device=image.device),
                             torch.linspace(-1,1,w,device=image.device),indexing='ij')
        grid=torch.stack([xx,yy],-1).reshape(-1,2)
        coords=attention@grid
        state=(state-self.offset)/self.scale
        normalized=self.action_head(torch.cat([coords.flatten(1),state],1)).reshape(-1,8,4)
        return normalized*self.scale+self.offset,logits,coords

    def loss(self,image,tokens,state,target,pixels):
        actions,logits,coords=self(image,tokens,state)
        h,w=logits.shape[-2:]
        yy,xx=torch.meshgrid(torch.arange(h,device=image.device),torch.arange(w,device=image.device),indexing='ij')
        centers=pixels*torch.tensor([w-1,h-1],device=image.device)
        heat=torch.exp(-((xx[None,None]-centers[:,:,0,None,None])**2+
                         (yy[None,None]-centers[:,:,1,None,None])**2)/(2*.85**2))
        heat=heat.flatten(2);heat=heat/heat.sum(-1,keepdim=True)
        heat_loss=-(heat*logits.flatten(2).log_softmax(-1)).sum(-1).mean()
        action_loss=(((actions-target)/self.scale)**2).mean()
        coord_loss=((coords-(pixels*2-1))**2).mean()
        return heat_loss+8*action_loss+4*coord_loss,actions
