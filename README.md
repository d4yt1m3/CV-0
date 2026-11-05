# CV 实践

方便对比不同模型架构的性能

## TODO

1. 拓展模型架构
2. 拓展数据集

```bash
python train.py --dataset cifar10
                --backbone mlp
                --optimizer sgd
                --criterion cross_entropy
                --lr 0.01
                --batch_size 128
                --epochs 10
                --device mps
```

dataset: mnist, cifar10, cifar100

backbone: linear, mlp, cnn

optimizer: sgd, adam

criterion: cross_entropy, mse

device: cpu, cuda, mps

## 可视化界面

```bash
uv sync
uv run streamlit run app.py
```

左侧选数据集、模型、优化器、损失函数等参数并训练；每组实验结束后会进入对比表格，
准确率 / Loss 曲线自动叠加，并可查看混淆矩阵和错分样本。适合给刚接触 CV 的新手做横向对比。
