import tqdm
import torch
from block.metric_get import metric


def val_get(args, val_dataloader, model, loss, data_dict, ema):
    with torch.no_grad():
        model = ema.ema if args.ema else model.eval()
        pred = []
        true = []
        val_loss = 0
        tqdm_len = len(data_dict['val']) // args.batch * args.device_number
        tqdm_show = tqdm.tqdm(total=tqdm_len, mininterval=0.2)
        for index, (input_data_batch, edge_index, true_batch) in enumerate(val_dataloader):
            input_data_batch = input_data_batch.to(args.device, non_blocking=args.latch)
            edge_index = edge_index.to(args.device, non_blocking=args.latch)
            true_batch = true_batch.to(args.device, non_blocking=args.latch)
            pred_batch = model(input_data_batch, edge_index)
            loss_batch = loss(pred_batch, true_batch)
            val_loss += loss_batch.item()
            pred.extend(pred_batch.cpu())
            true.extend(true_batch.cpu())
            # tqdm
            tqdm_show.set_postfix({'val_loss': loss_batch.item()})  # 添加loss显示
            tqdm_show.update(1)  # 更新进度条
        # tqdm
        tqdm_show.close()
        # 计算指标
        val_loss /= (index + 1)
        pred = torch.stack(pred, dim=0)
        true = torch.stack(true, dim=0)
        # 计算总相对指标
        mae, mse = metric(pred, true)
        print(f'\n| 验证 | val_loss:{val_loss:.4f} | val_mae:{mae:.4f} | val_mse:{mse:.4f} |')
    return val_loss, mae.item(), mse.item()
