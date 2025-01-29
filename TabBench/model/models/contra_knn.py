import torch
import torch.nn as nn
import torch.nn.functional as F
from model.lib.tabr.utils import make_module
from typing import Optional, Union
class Residual_block(nn.Module):
    def __init__(self,d_in,d,dropout):
        super().__init__()
        self.linear0=nn.Linear(d_in,d)
        self.Linear1=nn.Linear(d,d_in)
        self.bn=nn.BatchNorm1d(d_in)
        self.dropout=nn.Dropout(dropout)
        self.activation=nn.ReLU()
    def forward(self, x):
        z=self.bn(x)
        z=self.linear0(z)
        z=self.activation(z)
        z=self.dropout(z)
        z=self.Linear1(z)
        # z=x+z
        return z

class contra_knn(nn.Module):
    def __init__(
        self,
        *,
        d_in: int,
        d_num:int,
        d_out: int,
        dim:int,
        dropout:int,
        d_block:int,
        n_blocks:int,
        num_embeddings: Optional[dict],
        temperature:float=1.0,
        numK: int=5,
        ) -> None:

        super().__init__()
        self.d_in = d_in if num_embeddings is None  else d_num*num_embeddings['d_embedding']+d_in-d_num      
        self.d_out = d_out
        self.d_num=d_num
        self.dim = dim
        self.dropout = dropout
        self.d_block = d_block
        self.n_blocks = n_blocks
        self.T=temperature
        self.numK=numK
        if n_blocks >0:
            self.post_encoder=nn.Sequential()
            for i in range(n_blocks):
                name=f"ResidualBlock{i}"
                self.post_encoder.add_module(name,self.make_layer())
            self.post_encoder.add_module('bn',nn.BatchNorm1d(dim))
        self.encoder = nn.Linear(self.d_in, dim)
        # self.bn=nn.BatchNorm1d(dim)
        self.num_embeddings = (
            None
            if num_embeddings is None
            else make_module(num_embeddings, n_features=d_num)
        )

    def make_layer(self):
        block=Residual_block(self.dim,self.d_block,self.dropout)
        return block
            

    def forward(self, x):
        if self.num_embeddings is not None and self.d_num >0:
            x_num,x_cat=x[:,:self.d_num],x[:,self.d_num:]
            x_num=self.num_embeddings(x_num).flatten(1)
            x=torch.cat([x_num,x_cat],dim=-1)
        x=x.double()
        if self.n_blocks > 0:
            x = self.post_encoder(self.encoder(x))          
        else:         
            x = self.encoder(x)
        return x

    def predict(self,x,features,labels):
        with torch.no_grad():
            x=self.forward(x)
            # x=F.normalize(x,dim=-1)
            # features=F.normalize(features,dim=-1)
            dist=torch.cdist(x,features,p=2)
            _,indices=torch.topk(dist,self.numK,largest=False)
            pred=labels[indices]
            if self.d_out==1:
                logits=pred.mean(dim=1)
            else:
                pred_oh=F.one_hot(pred,self.d_out)
            #convert to logits
                logits=pred_oh.sum(dim=1)/self.numK
            return logits,x