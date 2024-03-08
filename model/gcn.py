import torch
import torch_geometric


class gcn(torch.nn.Module):
    def __init__(self, args):
        super().__init__()
        self.l0 = torch_geometric.nn.GCNConv(args.input_size, args.input_size, bias=True)
        self.l1 = torch.nn.SiLU()
        self.l2 = torch_geometric.nn.GCNConv(args.input_size, args.input_size, bias=True)
        self.l3 = torch.nn.SiLU()
        self.l4 = torch.nn.Linear(args.input_size, args.output_size)

    def forward(self, x, edge_index):
        x = self.l0(x, edge_index)
        x = self.l1(x)
        x = self.l2(x, edge_index)
        x = self.l3(x)
        x = self.l4(x)
        return x


if __name__ == '__main__':
    import argparse

    parser = argparse.ArgumentParser(description='')
    parser.add_argument('--input_size', default=5, type=int)
    parser.add_argument('--output_size', default=1, type=int)
    args = parser.parse_args()
    model = gcn(args)
    x = torch.randn(1000, args.input_size, dtype=torch.float32)
    edge_index = torch.randint(1, 1000, (2, 1000), dtype=torch.int64)
    pred = model(x, edge_index)
    print(model)
    print(pred.shape)
