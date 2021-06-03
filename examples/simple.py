import ragdoll
import torch

ragdoll.hello()
ragdoll.init()

z = torch.Tensor([2, 2])
z = ragdoll.add_one_op(z)
print(z)

print('Rank is', ragdoll.rank_op())
