import os
import torch


def model_get(args):
    if os.path.exists(args.weight):
        model_dict = torch.load(args.weight, map_location='cpu')
    else:
        choice_dict = {'graph_sage': 'model_prepare(args).graph_sage()',
                       }
        model = eval(choice_dict[args.model])
        model_dict = {}
        model_dict['model'] = model
        model_dict['epoch_finished'] = 0  # 已训练的轮数
        model_dict['optimizer_state_dict'] = None  # 学习率参数
        model_dict['ema_updates'] = 0  # ema参数
        model_dict['standard'] = 999  # 评价指标
    return model_dict


class model_prepare:
    def __init__(self, args):
        self.args = args

    def graph_sage(self):
        from model.graph_sage import graph_sage
        model = graph_sage(self.args)
        return model
