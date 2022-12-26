import paddle

class Distance(object):
    r"""Saves the Euclidean distance of linked nodes in its edge attributes
    (functional name: :obj:`distance`).

    Args:
        norm (bool, optional): If set to :obj:`False`, the output will not be
            normalized to the interval :math:`[0, 1]`. (default: :obj:`True`)
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
        row, col = data["edge_index"][0], data["edge_index"][1]
        pos = data["pos"]
        pseudo = data["edge_attr"] if "edge_attr" in data else None

        dist = paddle.norm(pos[col] - pos[row], p=2, axis=-1).reshape((-1, 1))

        if self.norm and dist.numel() > 0:
            dist = dist / (dist.max() if self.max is None else self.max)

        if pseudo is not None and self.cat:
            pseudo = pseudo.reshape((-1, 1)) if pseudo.dim() == 1 else pseudo
            # data.edge_attr = paddle.concat([pseudo, dist.type_as(pseudo)], axis=-1)
            data["edge_attr"] = paddle.concat([pseudo, dist.astype(pseudo.dtype)], axis=-1)
        else:
            # data.edge_attr = dist
            data["edge_attr"] = dist

        return data

    def __repr__(self) -> str:
        return (f'{self.__class__.__name__}(norm={self.norm}, '
                f'max_value={self.max})')
