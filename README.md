# PMP

- model.py 模型文件
- nn_data.py 数据读取文件
- nn_submit.py 提交文件
- nn_utils.py 训练器封装文件
- preparing.py 数据预处理
- utils.py 工具文件


## 使用方法

```
python nn.py --fold 0 --gpu 0 --name 0807 --lr 0.001
python nn.py --fold 0 --gpu 0 --name 0807-fine --lr 5e05 # 用来读取之前的权重进行two stage
```