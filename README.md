# meshgraphnets_paddlepaddle

## 1. 简介

本项目旨在用PaddlePaddle框架复现[[2010.03409\] Learning Mesh-Based Simulation with Graph Networks (arxiv.org)](https://arxiv.org/abs/2010.03409) 论文，目前支持了原论文中`cylinder_flow`数据集的模拟。

## 2. 数据准备

### 2.1 下载原始数据集

Deepmind官方提供了原始数据集的下载，脚本地址是：[download_dataset.sh](https://github.com/deepmind/deepmind-research/blob/master/meshgraphnets/download_dataset.sh)

使用方法：

```
bash download_dataset.sh cylinder_flow data/cylinder_flow
```

训练、验证和测试的tfrecord，及对应的meta.json 文件都会被下载到data/cylinder_flow 文件夹中。

### 2.2 处理tfrecord为h5py

使用`parse_tfrecord.py` 可以将tfrecord打成h5方便训练，使用方法：

```bash
python parse_tfrecord.py --dataset data/cylinder_flow --output data/cylinder_flow/datapkls
```

## 3. 训练

使用`train.py`进行训练，参数格式如下：

```
python train.py \
  --gpu
  --gpu_id 0
  --model_dir checkpoint/cylinder_flow
  --noise_std 0.02
  --data_dir data/cylinder_flow/datapkls
  --lr 0.0001
  --batch_size 4
  --gradient_accumulation_step 32
  --save_interval 200
  --max_epoch 20
```

## 4. 推理

### 4.1 计算网格结果

模型训练后进行推理可以使用`rollout.py`：

```bash
python rollout.py \
  --gpu \
  --rollout_num 5 \
  --model_dir checkpoint/cylinder_flow \
  --data_dir data/cylinder_flow/datapkls
```

在result/文件夹中会输出多个pkl文件，记录测试集的外推结果

### 4.2 可视化

```bash
python visualize_cylinder_flow.py
```

这个脚本会将result下的pkl都渲染成video中的mp4视频文件以便可视化。

## 5. Paddle复现心得

本项目是基于[echowve/meshGraphNets_pytorch: PyTorch implementations of Learning Mesh-based Simulation With Graph Networks (github.com)](https://github.com/echowve/meshGraphNets_pytorch)的paddle重构。

### 5.1 网络数据的表示和成批

Pytorch对图神经网络支持力度明显好于百度飞桨。其中torch_geometric / torch_scatter等包在构图上非常方便。为了重现这种方便，我们针对这个项目自己重现了torch_geometric中的data项，具体实现在`dataset/data.py`中。

图数据最有用的一个特性是成批训练(batching)。图网络的成批训练和图像、音频等数据简单的堆叠不同，图网络数据的成批是将多个图拼合成一个大图（这个大图中有多个互不相连的子图），在送入网络中进行推理。所以我们设计了一个函数`Data.offset_by_n(n)`：

```python
class Data(object):
	def offset_by_n(self, n):
        if self.edge_index is not None:
            self.edge_index = self.edge_index + n
        if self.face is not None:
            self.face = self.face + n
        return self
```

假设我们有两个图要成批，图一有两个点，图二在拼接上去前，可以方便地把自己的点的ID都加上2，避免和图一重合。

对应的`Collator`类则在dataloader中发挥作用，将一个批次中的图拼合成大图：

```python
class Collator(object):
    def __call__(self, batch_data):
        """
        batch_data is a list of Data, the collate_fn will produce one big graph with several disconnected subgraphs
        """
        offset = [i.num_nodes for i in batch_data]
        offset = np.cumsum(offset)
        offset = np.insert(offset, 0, 0)
        offset = offset[:-1]
        batch_data = [i.offset_by_n(n) for i, n in zip(batch_data, offset)]

        return Data(
            x=concat([i.x for i in batch_data], axis=0),
            face=concat([i.face for i in batch_data], axis=1),
            y=concat([i.y for i in batch_data], axis=0),
            pos=concat([i.pos for i in batch_data], axis=0),
            edge_index=concat([i.edge_index for i in batch_data], axis=1),
            edge_attr=concat([i.edge_attr for i in batch_data], axis=0),
        )
```

### 5.2 稀疏操作

torch_scatter中有一个很方便完成稀疏累加的函数`scatter_add`，其使用场景通常是将图上一个点的特征与这个点的邻居的特征相加，参考：[Scatter — pytorch_scatter 2.1.0 documentation (pytorch-scatter.readthedocs.io)](https://pytorch-scatter.readthedocs.io/en/latest/functions/scatter.html)

paddle中相应的实现是：

```python
def scatter_add(src, index, dim_size=None):
    if dim_size is None:
        dim_size = paddle.max(index) + 1

    indices = paddle.unsqueeze(index, axis=1)
    x = paddle.zeros_like(src)
    y = paddle.scatter_nd_add(x, indices, src)
    return y[:dim_size]

```

一个简单的例子是：

```python
import paddle
from transforms.scatter import scatter_add
x = paddle.arange(1,11)
index = paddle.to_tensor([0,0,0,1,1,1,2,2,2,3])
scatter_add(x, index)

# >> Tensor([6 , 15, 24, 10])
```

这里面的意思是：将x中前三项求和，放到结果的第0位置，将第四到第六项求和，放到结果的第1位置，如此类推。所以结果的第一项是1+2+3 = 6，第二项是4+5=6 = 15,第三项是7+8+9=24，第四项是10。

### 5.3 数据变换

一些常用的torch_geometric中的Transformations也被改写成paddle，置入了`transforms`。在train.py中，我们定义原始图数据的变换只需要这样：

```python
transforms = Compose([FaceToEdge(), Cartesian(norm=False), Distance(norm=False),])

graph = transforms(graph)
```





