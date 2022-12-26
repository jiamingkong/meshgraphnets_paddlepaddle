import paddle


def scatter_add(src, index, dim_size=None):
    if dim_size is None:
        dim_size = paddle.max(index) + 1

    indices = paddle.unsqueeze(index, axis=1)
    x = paddle.zeros_like(src)
    y = paddle.scatter_nd_add(x, indices, src)
    # trim y's size
    return y[:dim_size]
