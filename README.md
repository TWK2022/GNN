## GNN
>
### 1，环境
>torch：https://pytorch.org/get-started/previous-versions/
>```
>pip install transformers peft -i https://pypi.tuna.tsinghua.edu.cn/simple
>```
### 2，predict.py
>使用模型
### 3，gradio_start.py
>用gradio将程序包装成一个可视化的界面，可以在网页可视化的展示
### 4，flask_start.py
>用flask将程序包装成一个服务，并在服务器上启动
### 5，flask_request.py
>以post请求传输数据调用服务
### 6，gunicorn_config.py
>用gunicorn多进程启动flask服务：gunicorn -c gunicorn_config.py flask_start:app
### 7，run.py
>微调模型：训练peft模型
***
![image](README_IMAGE/001.jpg)
