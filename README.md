# 基于深度强化学习求解旅行商问题与优化算法研究
本仓库代码是本人本科毕业设计的代码，该毕业设计的题目为《基于深度强化学习求解旅行商问题与优化算法研究》

## 依赖

* python = 3.6.3
* NumPy
* Scipy
* PyTorch = 1.7
* tensorboard_logger（可以不装）

## 快速开始

训练该模型可以在控制台输入：

```
python train.py --baseline rollout --graph_size 20
```

训练过程会比较久。故在trained_models文件夹中直接提供了一个训练完成后的针对TSP20和TSP50训练的模型。

测试该模型的性能可以在控制台输入：

```
python test.py --baseline rollout --graph_size 20 --load_path trained_model/TSP20/epoch_99.pt
```

若想要使用beam search进行测试，则在控制台输入：

```
python test.py --baseline rollout --graph_size 20 --decode_type beam_search --beam_width 5 --load_path trained_model/TSP20/epoch_99.pt
```



data文件夹中存放的为测试数据集，由随机种子1234生成的10k个数据样本。
