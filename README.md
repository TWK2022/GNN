## pytorch图神经网络预测框架
>代码兼容性较强，使用的是一些基本的库、基础的函数  
>在argparse中可以选择使用wandb，能在wandb网站中生成可视化的训练过程
### 1，环境
>torch：https://pytorch.org/get-started/previous-versions/
>```
>pip install tqdm wandb torch_geometric -i https://pypi.tuna.tsinghua.edu.cn/simple
>```
### 2，数据格式
>torch_geometric.data.Data保存为的.pt文件：包含x、edge_index、y等
### 3，run.py
>模型训练时运行该文件，argparse中有对每个参数的说明
### 4，predict_pt.py
>使用训练好的pt模型预测
### 其他
>学习笔记：https://github.com/TWK2022/notebook
