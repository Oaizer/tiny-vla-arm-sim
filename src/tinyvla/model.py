"""Small model that reads camera images and commands and predicts arm actions."""
import torch
from torch import nn
from .language import OBJECTS,PADS,VOCAB

class Policy(nn.Module):
    def __init__(self):
        super().__init__();n=len(OBJECTS)+len(PADS)
        self.vision=nn.Sequential(nn.Conv2d(3,16,5,padding=2),nn.ReLU(),
            nn.Conv2d(16,24,3,stride=2,padding=1),nn.ReLU(),
            nn.Conv2d(24,32,3,padding=1),nn.ReLU(),nn.Conv2d(32,n,1))
        self.words=nn.Embedding(len(VOCAB),16,padding_idx=0);self.language=nn.GRU(16,32,batch_first=True)
        self.selection=nn.Linear(32,len(OBJECTS)+len(PADS)+2)
        self.action_head=nn.Linear(8,64)
        self.register_buffer('offset',torch.tensor([.5,0,0,0.]));self.register_buffer('scale',torch.tensor([.3,.3,.3,.04]))

    def features(self,rgb,tokens,state,calibration):
        logits=self.vision(rgb);b,c,h,w=logits.shape
        attention=logits.flatten(2).softmax(-1)
        yy,xx=torch.meshgrid(torch.linspace(0,1,h,device=rgb.device),torch.linspace(0,1,w,device=rgb.device),indexing='ij')
        grid=torch.stack([xx,yy],-1).reshape(-1,2);uv=attention@grid
        ray=torch.cat([uv,torch.ones_like(uv[:,:,:1])],-1)
        world=torch.einsum('bcij,bcj->bci',calibration,ray);world=world[:,:,:2]/world[:,:,2:3]
        sequence,_=self.language(self.words(tokens));length=(tokens!=0).sum(1).clamp_min(1)-1
        text=sequence[torch.arange(b,device=rgb.device),length]
        gates=self.selection(text);source_logits=gates[:,:4];pad_logits=gates[:,4:7];task_logits=gates[:,7:]
        source_weights=source_logits.softmax(-1);pad_weights=pad_logits.softmax(-1);task_weights=task_logits.softmax(-1)
        source=(world[:,:4]*source_weights[:,:,None]).sum(1)
        destination=(world[:,4:]*pad_weights[:,:,None]).sum(1)
        features=torch.cat([(torch.cat([source,destination],1)-torch.tensor([.5,0,.5,0],device=rgb.device))/.3,
                             (state-self.offset)/self.scale],1)
        presence=logits.flatten(2).amax(-1).sigmoid()
        return features,{'heatmaps':logits,'pixels':uv,'world':world,'presence':presence,
            'source_logits':source_logits,'pad_logits':pad_logits,'task_logits':task_logits,
            'task_weights':task_weights,'source_confidence':(presence[:,:4]*source_weights).sum(1),
            'pad_confidence':(presence[:,4:]*pad_weights).sum(1)}

    def forward(self,rgb,tokens,state,calibration):
        features,info=self.features(rgb,tokens,state,calibration)
        heads=self.action_head(features).reshape(-1,2,8,4)
        actions=(heads*info['task_weights'][:,:,None,None]).sum(1)*self.scale+self.offset
        return actions,info

    def loss(self,batch):
        action,info=self(batch['images'],batch['tokens'],batch['states'],batch['calibration'])
        maps=info['heatmaps'];h,w=maps.shape[-2:]
        yy,xx=torch.meshgrid(torch.arange(h,device=maps.device),torch.arange(w,device=maps.device),indexing='ij')
        centers=batch['pixels']*torch.tensor([w-1,h-1],device=maps.device)
        heat=torch.exp(-((xx[None,None]-centers[:,:,0,None,None])**2+(yy[None,None]-centers[:,:,1,None,None])**2)/(2*.65**2))
        heat=heat.flatten(2);heat=heat/heat.sum(-1,keepdim=True).clamp_min(1e-10)
        present=batch['present'];spatial=(-(heat*maps.flatten(2).log_softmax(-1)).sum(-1)*present).sum()/present.sum().clamp_min(1)
        coord=((info['pixels']-batch['pixels']).square().sum(-1)*present).sum()/present.sum().clamp_min(1)
        presence=(nn.functional.binary_cross_entropy_with_logits(maps.flatten(2).amax(-1),present,reduction='none')*(1+3*(1-present))).mean()
        language=nn.functional.cross_entropy(info['source_logits'],batch['source'])+nn.functional.cross_entropy(info['task_logits'],batch['task'])
        place=batch['task']==0
        if place.any():language+=nn.functional.cross_entropy(info['pad_logits'][place],batch['pad'][place])
        valid=batch['valid'];errors=((action-batch['actions'])/self.scale).square().mean((1,2))
        imitation=(errors*valid).sum()/valid.sum().clamp_min(1)
        return spatial+20*coord+.5*presence+language+8*imitation,action,info
