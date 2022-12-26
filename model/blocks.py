# import torch
# import torch.nn as nn
# import torch.nn.functional as F
# from torch_scatter import scatter_add
# from utils.utils import decompose_graph
# from torch_geometric.data import Data

import paddle
import paddle.nn as nn
import paddle.nn.functional as F
from utils.utils import decompose_graph
from transforms.scatter import scatter_add
import numpy as np


class EdgeBlock(nn.Layer):
    def __init__(self, custom_func=None):

        super(EdgeBlock, self).__init__()
        self.net = custom_func

    def forward(self, graph):

        node_attr, edge_index, edge_attr, _ = decompose_graph(graph)
        senders_idx, receivers_idx = edge_index
        edges_to_collect = []

        senders_attr = node_attr[senders_idx]
        receivers_attr = node_attr[receivers_idx]

        edges_to_collect.append(senders_attr)
        edges_to_collect.append(receivers_attr)
        edges_to_collect.append(edge_attr)
       
        collected_edges = paddle.concat(edges_to_collect, axis=1)

        edge_attr_ = self.net(collected_edges)  # Update

        return {
            "x": node_attr,
            "edge_attr": edge_attr_,
            "edge_index": edge_index,
            "num_nodes": node_attr.shape[0],
        }


class NodeBlock(nn.Layer):
    def __init__(self, custom_func=None):

        super(NodeBlock, self).__init__()

        self.net = custom_func

    def forward(self, graph):
        # Decompose graph
        edge_attr = graph["edge_attr"]
        nodes_to_collect = []

        receivers_idx = graph["edge_index"][1] # correct
        num_nodes = graph["num_nodes"] # 1876, correct
        # OK the scatter add, might need to switch to paddle's scatter_nd_add
        agg_received_edges = scatter_add(edge_attr, receivers_idx, dim_size=num_nodes)
        nodes_to_collect.append(graph["x"])
        nodes_to_collect.append(agg_received_edges)
        collected_nodes = paddle.concat(nodes_to_collect, axis=-1)
        if self.net is not None:
            x = self.net(collected_nodes)
        else:
            x = collected_nodes
        return {"x": x, "edge_attr": edge_attr, "edge_index": graph["edge_index"], "num_nodes": num_nodes}
