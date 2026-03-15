import torch

tensor = torch.tensor([1,2,3])
tensor1 = torch.tensor([5,6,7])
print(torch.cat((tensor,tensor1),dim=0))
#torch.__version__
