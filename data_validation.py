import os
import tqdm
import time
import torch
import argparse
from block.loss_get import loss_get

# -------------------------------------------------------------------------------------------------------------------- #
# 用已经训练好的模型对数据进行验证，可以查看哪些数据的损失较大，检查是否有异常
# -------------------------------------------------------------------------------------------------------------------- #
# 设置
parser = argparse.ArgumentParser(description='|数据验证|')
parser.add_argument('--model_path', default='best.pt', type=str, help='|pt模型位置|')
parser.add_argument('--data_path', default=r'data_demo.pt', type=str, help='|数据位置|')
parser.add_argument('--label_function', default='data.y', type=str, help='|标签为eval(label_function)|')
parser.add_argument('--input_size', default=34, type=int, help='|输入的长度|')
parser.add_argument('--output_size', default=4, type=int, help='|输出的长度|')
parser.add_argument('--loss', default='mse', type=str, help='|损失函数，注意分类和数值预测任务要用不同的损失函数|')
parser.add_argument('--loss_threshold', default=0.3, type=float, help='|显示损失比较大的值|')
parser.add_argument('--device', default='cuda', type=str, help='|推理设备|')
parser.add_argument('--latch', default=True, type=bool, help='|模型和数据是否为锁存，True为锁存|')
parser.add_argument('--num_worker', default=0, type=int, help='|CPU处理数据的进程数，0只有一个主进程，一般为0、2、4、8|')
args = parser.parse_args()
args.save_path = 'save_image'
# -------------------------------------------------------------------------------------------------------------------- #
# 初步检查
assert os.path.exists(args.model_path), f'! model_path不存在:{args.model_path} !'
assert os.path.exists(args.data_path), f'! data_path不存在:{args.data_path} !'
if not os.path.exists(args.save_path):
    os.makedirs(args.save_path)


# -------------------------------------------------------------------------------------------------------------------- #
# 程序
def metric(pred, true):
    mae = torch.mean(abs(pred - true))
    mse = torch.mean(torch.square(pred - true))
    return mae, mse


def data_validation():
    # 损失
    loss = loss_get(args)
    # 加载模型
    model_dict = torch.load(args.model_path, map_location='cpu')
    model = model_dict['model'].eval().to(args.device)
    model = model.half() if args.device == 'cuda' else model.float()
    epoch = model_dict['epoch_finished']
    val_loss = round(model_dict['val_loss'], 4)
    print(f'| 模型加载成功:{args.model_path} | epoch:{epoch} | val_loss:{val_loss}|')
    # 加载数据
    data = torch.load(args.data_path)  # torch_geometric.data.Data格式
    # 推理
    dataset = torch_dataset(args, data)
    dataloader = torch.utils.data.DataLoader(dataset, batch_size=1,
                                             shuffle=False, drop_last=False, pin_memory=False,
                                             num_workers=args.num_worker, collate_fn=dataset.collate_fn)
    start_time = time.time()
    pred = []
    true = []
    val_loss = 0
    tqdm_len = len(data)
    tqdm_show = tqdm.tqdm(total=tqdm_len, mininterval=0.2)
    with torch.no_grad():
        for index, (input_data_batch, edge_index, true_batch, mask, num) in enumerate(dataloader):
            input_data_batch = input_data_batch.to(args.device, non_blocking=args.latch)
            edge_index = edge_index.to(args.device, non_blocking=args.latch)
            true_batch = true_batch.to(args.device, non_blocking=args.latch)
            pred_batch = model(input_data_batch, edge_index)
            loss_batch = loss(pred_batch[mask], true_batch[mask])
            val_loss += loss_batch.item()
            pred.extend(pred_batch.cpu())
            true.extend(true_batch.cpu())
            if loss_batch.item() > args.loss_threshold:
                print(f'| 损失较大的数据:{num} | val_loss:{loss_batch.item():.4f} |')
            # tqdm
            tqdm_show.set_postfix({'val_loss': loss_batch.item()})  # 添加loss显示
            tqdm_show.update(1)  # 更新进度条
        # tqdm
        tqdm_show.close()
    end_time = time.time()
    print('| 数据:{} 批量:{} 平均耗时:{:.4f} |'
          .format(args.data_path, args.batch, (end_time - start_time) / len(data) / pred_batch.shape[0]))
    # 计算指标
    val_loss /= (index + 1)
    if args.loss != 'cross':  # 预测数值
        pred = torch.stack(pred, dim=0)
        true = torch.stack(true, dim=0)
        mae, mse = metric(pred, true)
        print(f'\n| 验证 | val_loss:{val_loss:.4f} | val_mae:{mae:.4f} | val_mse:{mse:.4f} |')
    else:  # 分类
        print(f'\n| 验证 | val_loss:{val_loss:.4f} |')


class torch_dataset(torch.utils.data.Dataset):
    def __init__(self, args, data):
        self.data = data
        self.device = args.device
        self.input_size = args.input_size
        self.output_size = args.output_size
        self.label_function = args.label_function

    def __len__(self):
        return len(self.data)

    def __getitem__(self, index):
        data = self.data[index]
        input_data = data.x.type(torch.float16) if self.device == 'cuda' else data.x
        edge_index = data.edge_index
        true = eval(self.label_function)
        true = true.unsqueeze(1) if len(true.shape) == 1 else true
        mask = data.train_mask if hasattr(data, 'train_mask') else [True for _ in range(len(data.x))]
        return input_data, edge_index, true, mask, index

    def collate_fn(self, getitem_list):  # 自定义__getitem__的合并方式
        input_data_list = [_[0] for _ in getitem_list]
        edge_index_list = [_[1] for _ in getitem_list]
        true_list = [_[2] for _ in getitem_list]
        mask_list = [_[3] for _ in getitem_list]
        num_list = [_[4] for _ in getitem_list]
        input_data_batch = torch.concat(input_data_list, dim=0)
        edge_index = edge_index_list[0]
        true_batch = torch.concat(true_list, dim=0)
        mask = mask_list[0]
        num = num_list[0]
        return input_data_batch, edge_index, true_batch, mask, num


if __name__ == '__main__':
    data_validation()
