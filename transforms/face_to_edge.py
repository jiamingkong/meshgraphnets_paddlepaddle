import paddle
import paddle.nn.functional as F
from paddle import nn
from .basic import BasicDescriptor


class FaceToEdge(object):
    r"""Converts mesh faces :obj:`[3, num_faces]` to edge indices
    :obj:`[2, num_edges]` (functional name: :obj:`face_to_edge`).

    Args:
        remove_faces (bool, optional): If set to :obj:`False`, the face tensor
            will not be removed.
    """

    def __init__(self, remove_faces: bool = True):
        self.remove_faces = remove_faces

    def __call__(self, data):

        # if "face" in data:
        if data.face is not None:
            # face = data["face"]
            # Paddle doesn't have the "cat" api, therefore I have to hand-code this:
            # face: [3,3518]
            face = data.face
            edge_index = paddle.concat([face[:2], face[1:], face[::2]], axis=1)
            edge_index = to_undirected(edge_index, num_nodes=data.num_nodes)

            # data.edge_index = edge_index
            # data["edge_index"] = edge_index
            data.edge_index = edge_index
            if self.remove_faces:
                # del data["face"]
                data.face = None

        return data


def to_undirected(
    edge_index,
    edge_attr=None,
    num_nodes=None,
    reduce: str = "add",
):
    r"""Converts the graph given by :attr:`edge_index` to an undirected graph
    such that :math:`(j,i) \in \mathcal{E}` for every edge :math:`(i,j) \in
    \mathcal{E}`.

    Args:
        edge_index (LongTensor): The edge indices.
        edge_attr (Tensor or List[Tensor], optional): Edge weights or multi-
            dimensional edge features.
            If given as a list, will remove duplicates for all its entries.
            (default: :obj:`None`)
        num_nodes (int, optional): The number of nodes, *i.e.*
            :obj:`max_val + 1` of :attr:`edge_index`. (default: :obj:`None`)
        reduce (string, optional): The reduce operation to use for merging edge
            features (:obj:`"add"`, :obj:`"mean"`, :obj:`"min"`, :obj:`"max"`,
            :obj:`"mul"`). (default: :obj:`"add"`)

    :rtype: :class:`LongTensor` if :attr:`edge_attr` is :obj:`None`, else
        (:class:`LongTensor`, :obj:`Tensor` or :obj:`List[Tensor]]`)

    Examples:

        >>> edge_index = paddle.to_tensor([[0, 1, 1],
        ...                            [1, 0, 2]])
        >>> to_undirected(edge_index)
        tensor([[0, 1, 1, 2],
                [1, 0, 2, 1]])

        >>> edge_index = paddle.to_tensor([[0, 1, 1],
        ...                            [1, 0, 2]])
        >>> edge_weight = paddle.to_tensor([1., 1., 1.])
        >>> to_undirected(edge_index, edge_weight)
        (tensor([[0, 1, 1, 2],
                [1, 0, 2, 1]]),
        tensor([2., 2., 1., 1.]))

        >>> # Use 'mean' operation to merge edge features
        >>>  to_undirected(edge_index, edge_weight, reduce='mean')
        (tensor([[0, 1, 1, 2],
                [1, 0, 2, 1]]),
        tensor([1., 1., 1., 1.]))
    """
    # Maintain backward compatibility to `to_undirected(edge_index, num_nodes)`
    if isinstance(edge_attr, int):
        edge_attr = None
        num_nodes = edge_attr

    row, col = edge_index[0], edge_index[1]
    # print(f"row = {row}, col = {col}")
    row, col = paddle.concat([row, col], axis=0), paddle.concat([col, row], axis=0)
    edge_index = paddle.stack([row, col], axis=0)  # shape (2,n)

    if isinstance(edge_attr, paddle.Tensor):
        edge_attr = paddle.concat([edge_attr, edge_attr], axis=0)
    # elif isinstance(edge_attr, (list, tuple)):
    #     edge_attr = [paddle.concat([e, e], axis=0) for e in edge_attr]

    return coalesce(edge_index, edge_attr, num_nodes, reduce)


def maybe_num_nodes(edge_index, num_nodes=None):
    if num_nodes is not None:
        return num_nodes
    elif isinstance(edge_index, paddle.Tensor):
        return int(edge_index.max()) + 1 if edge_index.numel() > 0 else 0
    else:
        return max(edge_index.size(0), edge_index.size(1))


def coalesce(
    edge_index,
    edge_attr=None,
    num_nodes=None,
    reduce="add",
    is_sorted=False,
    sort_by_row=True,
):
    """Row-wise sorts :obj:`edge_index` and removes its duplicated entries.
    Duplicate entries in :obj:`edge_attr` are merged by scattering them
    together according to the given :obj:`reduce` option.

    Args:
        edge_index (LongTensor): The edge indices.
        edge_attr (Tensor or List[Tensor], optional): Edge weights or multi-
            dimensional edge features.
            If given as a list, will re-shuffle and remove duplicates for all
            its entries. (default: :obj:`None`)
        num_nodes (int, optional): The number of nodes, *i.e.*
            :obj:`max_val + 1` of :attr:`edge_index`. (default: :obj:`None`)
        reduce (string, optional): The reduce operation to use for merging edge
            features (:obj:`"add"`, :obj:`"mean"`, :obj:`"min"`, :obj:`"max"`,
            :obj:`"mul"`). (default: :obj:`"add"`)
        is_sorted (bool, optional): If set to :obj:`True`, will expect
            :obj:`edge_index` to be already sorted row-wise.
        sort_by_row (bool, optional): If set to :obj:`False`, will sort
            :obj:`edge_index` column-wise.

    :rtype: :class:`LongTensor` if :attr:`edge_attr` is :obj:`None`, else
        (:class:`LongTensor`, :obj:`Tensor` or :obj:`List[Tensor]]`)

    Example:

        >>> edge_index = torch.tensor([[1, 1, 2, 3],
        ...                            [3, 3, 1, 2]])
        >>> edge_attr = torch.tensor([1., 1., 1., 1.])
        >>> coalesce(edge_index)
        tensor([[1, 2, 3],
                [3, 1, 2]])

        >>> # Sort `edge_index` column-wise
        >>> coalesce(edge_index, sort_by_row=False)
        tensor([[2, 3, 1],
                [1, 2, 3]])

        >>> coalesce(edge_index, edge_attr)
        (tensor([[1, 2, 3],
                [3, 1, 2]]),
        tensor([2., 1., 1.]))

        >>> # Use 'mean' operation to merge edge features
        >>> coalesce(edge_index, edge_attr, reduce='mean')
        (tensor([[1, 2, 3],
                [3, 1, 2]]),
        tensor([1., 1., 1.]))
    """
    nnz = edge_index.shape[1]
    num_nodes = maybe_num_nodes(edge_index, num_nodes)
    # print(f"num_nodes = {num_nodes}")
    idx = edge_index[1 - int(sort_by_row)]
    idx = paddle.concat([paddle.to_tensor([-1], dtype=idx.dtype), idx], axis=0)
    idx[1:] = idx[1:] * num_nodes + edge_index[int(sort_by_row)]
    # print(f"idx = {idx}")
    if not is_sorted:
        # idx[1:], perm = idx[1:].sort()
        perm = paddle.argsort(idx[1:])
        values = paddle.sort(idx[1:])
        # print(f"perm = {perm}, values = {values}")
        idx[1:] = values
        # print(f"idx = {idx}")
        edge_index = paddle.gather(edge_index, perm, axis=1)
        # print(f"edge_index = {edge_index}")
        if isinstance(edge_attr, paddle.Tensor):
            edge_attr = edge_attr[perm]
        elif isinstance(edge_attr, (list, tuple)):
            edge_attr = [e[perm] for e in edge_attr]

    mask = idx[1:] > idx[:-1]

    # print(f"mask = {mask}") # True, True, False, True
    # Only perform expensive merging in case there exists duplicates:

    if mask.all():
        if isinstance(edge_attr, (paddle.Tensor, list, tuple)):
            return edge_index, edge_attr
        return edge_index

    mask_ = paddle.tile(mask, (edge_index.shape[0], 1))
    edge_index = paddle.masked_select(edge_index, mask_).reshape((2, mask.sum()))

    if edge_attr is None:
        return edge_index

    dim_size = edge_index.shape[1]
    idx = paddle.arange(0, nnz, device=edge_index.device)
    idx.sub_(mask.logical_not_().cumsum(dim=0))

    if isinstance(edge_attr, paddle.Tensor):
        edge_attr = paddle.scatter(edge_attr, idx, 0, None, dim_size, reduce)
        return edge_index, edge_attr
    elif isinstance(edge_attr, (list, tuple)):
        edge_attr = [
            paddle.scatter(e, idx, 0, None, dim_size, reduce) for e in edge_attr
        ]
        return edge_index, edge_attr

    return edge_index


if __name__ == "__main__":
    edge_index = paddle.to_tensor([[1, 1, 2, 3], [3, 3, 1, 2]])
    print(coalesce(edge_index))
    edge_index = paddle.to_tensor([[0, 1, 1], [1, 0, 2]])
    print(to_undirected(edge_index))
