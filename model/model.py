from .blocks import NodeBlock, EdgeBlock
import paddle
import paddle.nn as nn
import paddle.nn.functional as F
from utils.utils import decompose_graph, copy_geometric_data

# import torch.nn as nn
# from .blocks import EdgeBlock, NodeBlock
# from utils.utils import decompose_graph, copy_geometric_data
# from torch_geometric.data import Data


# rewrite the above in paddle
def build_mlp(in_size, hidden_size, out_size, lay_norm=True):
    """
    Given the input size, hidden size and output size, build a MLP with ReLU activation and LayerNorm
    """
    module = nn.Sequential(
        nn.Linear(in_size, hidden_size),
        nn.ReLU(),
        nn.Linear(hidden_size, hidden_size),
        nn.ReLU(),
        nn.Linear(hidden_size, hidden_size),
        nn.ReLU(),
        nn.Linear(hidden_size, out_size),
    )
    if lay_norm:
        return nn.Sequential(module, nn.LayerNorm(normalized_shape=out_size))
    return module


class Encoder(nn.Layer):
    def __init__(self, edge_input_size=128, node_input_size=128, hidden_size=128):
        super(Encoder, self).__init__()

        self.eb_encoder = build_mlp(edge_input_size, hidden_size, hidden_size)
        self.nb_encoder = build_mlp(node_input_size, hidden_size, hidden_size)

    def forward(self, graph: dict) -> dict:

        # (x, edge_index, edge_attr, global_attr)
        node_attr, _, edge_attr, _ = decompose_graph(graph)
        node_ = self.nb_encoder(node_attr)
        edge_ = self.eb_encoder(edge_attr)

        # return Data(x=node_, edge_attr=edge_, edge_index=graph.edge_index)
        return {
            "x": node_,
            "edge_attr": edge_,
            "edge_index": graph["edge_index"],
            "num_nodes": graph["num_nodes"],
        }


class GnBlock(nn.Layer):
    def __init__(self, hidden_size=128):

        super(GnBlock, self).__init__()

        eb_input_dim = 3 * hidden_size
        nb_input_dim = 2 * hidden_size
        nb_custom_func = build_mlp(nb_input_dim, hidden_size, hidden_size)
        eb_custom_func = build_mlp(eb_input_dim, hidden_size, hidden_size)

        self.eb_module = EdgeBlock(custom_func=eb_custom_func)
        self.nb_module = NodeBlock(custom_func=nb_custom_func)

    def forward(self, graph: dict) -> dict:

        graph_last = copy_geometric_data(graph)
        graph = self.eb_module(graph)
        graph = self.nb_module(graph)
        # edge_attr = graph_last.edge_attr + graph.edge_attr
        edge_attr = graph_last["edge_attr"] + graph["edge_attr"]
        # x = graph_last.x + graph.x
        x = graph_last["x"] + graph["x"]
        # return Data(x=x, edge_attr=edge_attr, edge_index=graph.edge_index)
        return {"x": x, "edge_attr": edge_attr, "edge_index": graph["edge_index"], "num_nodes": graph["num_nodes"]}


class Decoder(nn.Layer):
    def __init__(self, hidden_size=128, output_size=2):
        super(Decoder, self).__init__()
        self.decode_module = build_mlp(
            hidden_size, hidden_size, output_size, lay_norm=False
        )

    def forward(self, graph: dict) -> paddle.Tensor:
        # return self.decode_module(graph.x)
        return self.decode_module(graph["x"])


class EncoderProcesserDecoder(nn.Layer):
    def __init__(
        self, message_passing_num, node_input_size, edge_input_size, hidden_size=128
    ):

        super(EncoderProcesserDecoder, self).__init__()

        self.encoder = Encoder(
            edge_input_size=edge_input_size,
            node_input_size=node_input_size,
            hidden_size=hidden_size,
        )

        processer_list = []
        for _ in range(message_passing_num):
            processer_list.append(GnBlock(hidden_size=hidden_size))
        
        self.processer_list = processer_list

        self.decoder = Decoder(hidden_size=hidden_size, output_size=2)

    def forward(self, graph: dict) -> paddle.Tensor:

        graph = self.encoder(graph)
        for model in self.processer_list:
            graph = model(graph)
        decoded = self.decoder(graph)

        return decoded