import torch


def data_get(args):
    data_dict = data_prepare(args).load()
    return data_dict


class data_prepare(object):
    def __init__(self, args):
        self.data_path = args.data_path
        self.divide = args.divide

    def load(self):
        # 读取数据
        data = torch.load(self.data_path)  # torch_geometric.data.Data格式
        # 划分数据集
        data_len = len(data)  # 数据总数
        boundary = int(data_len * self.divide[0] / (self.divide[0] + self.divide[1]))  # 数据划分
        train = data[0:boundary]  # 训练数据
        val = data[boundary:]  # 验证数据
        data_dict = {'train': train, 'val': val}
        return data_dict
