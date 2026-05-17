import torch, flash_attn_wmma
Q=torch.randn(1,8,1024,64,device='cuda')
K=torch.randn(1,8,1024,64,device='cuda')
V=torch.randn(1,8,1024,64,device='cuda')
flash_attn_wmma.forward(Q,K,V)
