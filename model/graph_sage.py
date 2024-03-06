import torch
import torch_geometric
from model.layer import lbsd


class encoder(torch.nn.Module):  # in_->out_
    def __init__(self, in_, out_, dropout=0.2):
        super().__init__()
        self.lbsd0 = lbsd(in_, 2 * out_, dropout)
        self.lbsd1 = lbsd(2 * out_, 2 * out_, dropout)
        self.lbsd2 = lbsd(2 * out_, out_, dropout)

    def forward(self, x):
        x = self.lbsd0(x)
        x = self.lbsd1(x)
        x = self.lbsd2(x)
        return x


class decoder(torch.nn.Module):  # in_->out_
    def __init__(self, in_, out_, dropout=0.2):
        super().__init__()
        self.lbsd0 = lbsd(in_, 2 * in_, dropout)
        self.lbsd1 = lbsd(2 * in_, 2 * in_, dropout)
        self.lbsd2 = lbsd(2 * in_, out_, dropout)

    def forward(self, x):
        x = self.lbsd0(x)
        x = self.lbsd1(x)
        x = self.lbsd2(x)
        return x


class graph_sage(torch.nn.Module):
    def __init__(self, args):
        super().__init__()
        dim_dict = {'n': 4, 's': 8, 'm': 16, 'l': 32}
        n_dict = {'n': 1, 's': 2, 'm': 3, 'l': 5}
        dim = dim_dict[args.model_type]
        self.n = n_dict[args.model_type]
        input_size = args.input_size
        output_size = args.output_size
        self.encoder = encoder(input_size, dim, 0.2)
        self.sage_in = torch_geometric.nn.SAGEConv(in_channels=dim, out_channels=2 * dim, bias=True)
        for i in range(self.n):  # 或加bn层
            exec(f'self.sage{i} = torch_geometric.nn.SAGEConv(in_channels=2 * dim, out_channels=2 * dim, bias=False)')
        self.sage_out = torch_geometric.nn.SAGEConv(in_channels=2 * dim, out_channels=dim, bias=True)
        self.decoder = decoder(dim, output_size, 0.2)

    def forward(self, x, edge_index):
        x = self.encoder(x)
        x = self.sage_in(x, edge_index)
        for i in range(self.n):  # 或加bn层
            eval(f'self.sage{i}')(x, edge_index)
        x = self.sage_out(x, edge_index)
        x = self.decoder(x)
        return x


if __name__ == '__main__':
    import argparse

    parser = argparse.ArgumentParser(description='')
    parser.add_argument('--input_size', default=5, type=int)
    parser.add_argument('--output_size', default=1, type=int)
    parser.add_argument('--model_type', default='l', type=str)
    args = parser.parse_args()
    model = graph_sage(args)
    x = torch.randn(1000, args.input_size, dtype=torch.float32)
    edge_index = torch.randint(1, 1000, (2, 1000), dtype=torch.int64)
    pred = model(x, edge_index)
    print(model)
    print(pred.shape)
