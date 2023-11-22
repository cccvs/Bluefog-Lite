import torch
import bluefoglite
import bluefoglite.torch_api as bfl

bfl.init()
bfl.set_topology(topology=bluefoglite.RingGraph(bfl.size()))
print(f"I am rank {bfl.rank()} among size {bfl.size()}.")

tensor = torch.zeros(2).cuda() + bfl.rank()
print("Before: Rank ", bfl.rank(), " has data ", tensor)

neighbor_weights = {(bfl.rank() - 1) % bfl.size(): 1.0}
send_ranks = {(bfl.rank() + 1) % bfl.size(): 10.0}
bfl.neighbor_allreduce(
    tensor,
    self_weight=0.1,
    src_weights=neighbor_weights,
    dst_weights=send_ranks,
    inplace=True,
)
print("After: Rank ", bfl.rank(), " has data ", tensor)
