import paddle
from paddle import nn
import paddle.nn.functional as F

import enum


class NodeType(enum.IntEnum):
    NORMAL = 0
    OBSTACLE = 1
    AIRFOIL = 2
    HANDLE = 3
    INFLOW = 4
    OUTFLOW = 5
    WALL_BOUNDARY = 6
    SIZE = 9


# see https://github.com/sungyongs/dpgn/blob/master/utils.py
def decompose_graph(graph: dict) -> tuple:
    # graph: torch_geometric.data.data.Data
    # TODO: make it more robust
    x = graph.get("x", None)
    edge_index = graph.get("edge_index", None)
    edge_attr = graph.get("edge_attr", None)
    global_attr = graph.get("global_attr", None)
    return (x, edge_index, edge_attr, global_attr)

# see https://github.com/sungyongs/dpgn/blob/master/utils.py
def copy_geometric_data(graph: dict) -> dict:
    """return a copy of torch_geometric.data.data.Data
    This function should be carefully used based on
    which keys in a given graph.
    BUG: torch_geometric.data.data.Data does not have versions of paddle
    """
    node_attr, edge_index, edge_attr, global_attr = decompose_graph(graph)

    ret = {
        "x": node_attr,
        "edge_index": edge_index,
        "edge_attr": edge_attr,
        "global_attr": global_attr,
    }
    return ret