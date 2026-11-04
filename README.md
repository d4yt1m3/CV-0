# CV 实践

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
