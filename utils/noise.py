import paddle

from utils.utils import NodeType


def get_velocity_noise(graph: dict, noise_std: float) -> paddle.Tensor:
    """
    Return a noising tensor for velocity sequence in the graph.
    There are several types of nodes in the graph, and only the normal nodes are noised.
    """
    velocity_sequence = graph.x[:, 1:3]
    type = graph.x[:, 0]
    noise = paddle.normal(mean=0.0, std=noise_std, shape=velocity_sequence.shape)
    mask = type != NodeType.NORMAL
    noise[mask] = 0
    return noise
