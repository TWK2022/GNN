import torch


class lbsd(torch.nn.Module):  # in_->out_
    def __init__(self, in_, out_, dropout):
        super().__init__()
        self.linear = torch.nn.Linear(in_, out_, bias=False)
        self.bn = torch.nn.BatchNorm1d(out_, track_running_stats=False)
        self.silu = torch.nn.SiLU()
        self.dropout = torch.nn.Dropout(dropout)

    def forward(self, x):
        x = self.linear(x)
        x = self.bn(x)
        x = self.silu(x)
        x = self.dropout(x)
        return x


class series_encode(torch.nn.Module):
    def __init__(self, mean_input, std_input):
        super().__init__()
        self.mean_input = mean_input
        self.std_input = std_input

    def forward(self, x):
        for i in range(len(self.mean_input)):
            x[:, i] = (x[:, i] - self.mean_input[i]) / self.std_input[i]
        return x


class series_decode(torch.nn.Module):
    def __init__(self, mean_output, std_output):
        super().__init__()
        self.mean_output = mean_output
        self.std_output = std_output

    def forward(self, x):
        for i in range(len(self.mean_output)):
            x[:, i] = x[:, i] * self.std_output[i] + self.mean_output[i]
        return x


class deploy(torch.nn.Module):
    def __init__(self, model, mean_input, mean_output, std_input, std_output):
        super().__init__()
        self.series_encode = series_encode(mean_input, std_input)
        self.model = model
        self.series_decode = series_decode(mean_output, std_output)

    def forward(self, x):
        x = self.series_encode(x)
        x = self.model(x)
        x = self.series_decode(x)
        return x
