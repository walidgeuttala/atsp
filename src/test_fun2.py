import torch
import torch.nn.functional as F
from torch_geometric.nn import GCNConv

class GNN(torch.nn.Module):
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.conv1 = GCNConv(in_channels, 64)

    def forward(self, x: torch.Tensor, edge_index: torch.Tensor):
        x = self.conv1(x, edge_index)
        x = F.relu(x)
        return F.log_softmax(x, dim=1)

# Example input data for tracing
example_x = torch.randn((10, 5))  # 10 nodes with 5 features each
example_edge_index = torch.tensor([[0, 1, 2, 3], [1, 2, 3, 4]], dtype=torch.long)  # Example edge index

model = GNN(5, 5)
traced_model = torch.jit.trace(model, (example_x, example_edge_index))
model = torch.jit.script(traced_model)
