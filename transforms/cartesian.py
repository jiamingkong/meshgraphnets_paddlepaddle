import paddle
import paddle.nn as nn


class Cartesian(object):
    r"""Saves the relative Cartesian coordinates of linked nodes in its edge
    attributes (functional name: :obj:`cartesian`).

    Args:
        norm (bool, optional): If set to :obj:`False`, the output will not be
            normalized to the interval :math:`{[0, 1]}^D`.
            (default: :obj:`True`)
        max_value (float, optional): If set and :obj:`norm=True`, normalization
            will be performed based on this value instead of the maximum value
            found in the data. (default: :obj:`None`)
        cat (bool, optional): If set to :obj:`False`, all existing edge
            attributes will be replaced. (default: :obj:`True`)
    """

    def __init__(self, norm=True, max_value=None, cat=True):
        self.norm = norm
        self.max = max_value
        self.cat = cat

    def __call__(self, data: dict) -> dict:
        # (row, col), pos, pseudo = data.edge_index, data.pos, data.edge_attr
        # row, col = data["edge_index"][0], data["edge_index"][1]
        row, col = data.edge_index
        # pos = data["pos"]
        pos = data.pos
        # pseudo = data["edge_attr"] if "edge_attr" in data else None
        pseudo = data.edge_attr if data.edge_attr is not None else None

        cart = pos[row] - pos[col]
        cart = cart.view(-1, 1) if cart.dim() == 1 else cart

        if self.norm and cart.numel() > 0:
            max_value = cart.abs().max() if self.max is None else self.max
            cart = cart / (2 * max_value) + 0.5

        if pseudo is not None and self.cat:
            pseudo = pseudo.view(-1, 1) if pseudo.dim() == 1 else pseudo
            data.edge_attr = paddle.concat([pseudo, cart.type_as(pseudo)], axis=-1)
            # data["edge_attr"] = paddle.concat([pseudo, cart.type_as(pseudo)], axis=-1)

        else:
            data.edge_attr = cart
            # data["edge_attr"] = cart

        return data

    def __repr__(self) -> str:
        return f"{self.__class__.__name__}(norm={self.norm}, " f"max_value={self.max})"
