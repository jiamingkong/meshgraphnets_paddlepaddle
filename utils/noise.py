import paddle

from utils.utils import NodeType


def get_velocity_noise(graph: dict, noise_std: float) -> paddle.Tensor:
    # velocity_sequence = graph.x[:, 1:3]
    # type = graph.x[:, 0]
    # import pdb; pdb.set_trace()
    velocity_sequence = graph["x"][:, 1:3]
    type = graph["x"][:, 0]

    noise = paddle.normal(mean=0.0, std=noise_std, shape=velocity_sequence.shape)
    mask = type != NodeType.NORMAL
    noise[mask] = 0
    return noise