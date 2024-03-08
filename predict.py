import os
import tqdm
import time
import torch
import argparse

# -------------------------------------------------------------------------------------------------------------------- #
# 设置
parser = argparse.ArgumentParser(description='|pt模型推理|')
parser.add_argument('--model_path', default='best.pt', type=str, help='|pt模型位置|')
parser.add_argument('--data_path', default=r'data_demo.pt', type=str, help='|数据位置|')
parser.add_argument('--label_function', default='data.y', type=str, help='|标签为eval(label_function)|')
parser.add_argument('--divide', default='19,1', type=str, help='|训练集和验证集划分比例|')
parser.add_argument('--input_size', default=34, type=int, help='|输入的长度|')
parser.add_argument('--output_size', default=4, type=int, help='|输出的长度|')
parser.add_argument('--batch', default=1, type=int, help='|实际batch=batch*每组点数|')
parser.add_argument('--device', default='cpu', type=str, help='|推理设备|')
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


def predict_pt():
    # 加载模型
    model_dict = torch.load(args.model_path, map_location='cpu')
    model = model_dict['model'].eval().to(args.device)
    model = model.half() if args.device == 'cuda' else model.float()
    epoch = model_dict['epoch_finished']
    mse = round(model_dict['val_mse'], 4)
    print(f'| 模型加载成功:{args.model_path} | epoch:{epoch} | val_mse:{mse}|')
    # 加载数据
    data = torch.load(args.data_path)[180:]  # torch_geometric.data.Data格式
    # 推理
    loss = torch.nn.MSELoss()
    dataset = torch_dataset(args, data)
    dataloader = torch.utils.data.DataLoader(dataset, batch_size=args.batch,
                                             shuffle=False, drop_last=False, pin_memory=False,
                                             num_workers=args.num_worker, collate_fn=dataset.collate_fn)
    start_time = time.time()
    pred = []
    true = []
    val_loss = 0
    tqdm_len = len(data) // args.batch
    tqdm_show = tqdm.tqdm(total=tqdm_len, mininterval=0.2)
    with torch.no_grad():
        for index, (input_data_batch, edge_index, true_batch, num) in enumerate(dataloader):
            input_data_batch = input_data_batch.to(args.device)
            edge_index = edge_index.to(args.device)
            true_batch = true_batch.to(args.device)
            pred_batch = model(input_data_batch, edge_index)
            loss_batch = loss(pred_batch, true_batch)
            val_loss += loss_batch.item()
            pred.extend(pred_batch.cpu())
            true.extend(true_batch.cpu())
            if loss_batch.item() > 0.5:
                print(f'| 可能存在异常的数据:{num} | val_loss:{loss_batch.item():.4f} |')
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
    pred = torch.stack(pred, dim=0)
    true = torch.stack(true, dim=0)
    # 计算总相对指标
    mae, mse = metric(pred, true)
    print(f'\n| 验证 | val_loss:{val_loss:.4f} | val_mae:{mae:.4f} | val_mse:{mse:.4f} |')


class torch_dataset(torch.utils.data.Dataset):
    def __init__(self, args, data):
        self.data = data
        self.input_size = args.input_size
        self.output_size = args.output_size
        self.label_index = args.label_index

    def __len__(self):
        return len(self.data)

    def __getitem__(self, index):
        data = self.data[index]
        input_data = data.x
        edge_index = data.edge_index
        true = data.y[:, self.label_index]
        return input_data, edge_index, true, index

    def collate_fn(self, getitem_list):  # 自定义__getitem__的合并方式
        input_data_list = [_[0] for _ in getitem_list]
        edge_index_list = [_[1] for _ in getitem_list]
        true_list = [_[2] for _ in getitem_list]
        input_data_batch = torch.concat(input_data_list, dim=0)
        edge_index = edge_index_list[0]
        true_batch = torch.concat(true_list, dim=0)
        index = [_[3] for _ in getitem_list][0]
        return input_data_batch, edge_index, true_batch, index


if __name__ == '__main__':
    predict_pt()
