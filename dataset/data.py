# end of the day it's just a bit easier to create a Data object for graph composition and decomposing
import numpy as np
import paddle


class Data(object):
    def __init__(
        self,
        x=None,
        face=None,
        y=None,
        pos=None,
        edge_index=None,
        edge_attr=None,
        global_attr=None,
    ):
        self.x = x
        self.face = face
        self.y = y
        self.pos = pos
        self.edge_index = edge_index
        self.edge_attr = edge_attr
        self.global_attr = global_attr

    @property
    def num_nodes(self):
        """
        返回图中节点的数量
        """
        return self.x.shape[0]

    def __repr__(self):
        """
        人类可读的字符串表示形式
        """
        self_x_shape = f"x={self.x.shape}, " if self.x is not None else ""
        self_face_shape = f"face={self.face.shape}, " if self.face is not None else ""
        self_y_shape = f"y={self.y.shape}, " if self.y is not None else ""
        self_pos_shape = f"pos={self.pos.shape}, " if self.pos is not None else ""
        self_edge_index_shape = (
            f"edge_index={self.edge_index.shape}, "
            if self.edge_index is not None
            else ""
        )
        self_edge_attr_shape = (
            f"edge_attr={self.edge_attr.shape}" if self.edge_attr is not None else ""
        )
        self_global_attr_shape = (
            f"global_attr={self.global_attr.shape}"
            if self.global_attr is not None
            else ""
        )
        return f"Data({self_x_shape}{self_face_shape}{self_y_shape}{self_pos_shape}{self_edge_index_shape}{self_edge_attr_shape}{self_global_attr_shape})"

    def offset_by_n(self, n):
        """
        方便构建子图拼接成批成大图的函数
        """
        if self.edge_index is not None:
            self.edge_index = self.edge_index + n
        if self.face is not None:
            self.face = self.face + n
        return self


def concat(list_of_elements, axis=0):
    # if there is None in the list, return None
    if any([i is None for i in list_of_elements]):
        return None
    else:
        return paddle.concat(list_of_elements, axis=axis)


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
        _x = concat([i.x for i in batch_data], axis=0)
        _face = concat([i.face for i in batch_data], axis=1)
        _y = concat([i.y for i in batch_data], axis=0)
        _pos = concat([i.pos for i in batch_data], axis=0)
        _edge_index = concat([i.edge_index for i in batch_data], axis=1)
        _edge_attr = concat([i.edge_attr for i in batch_data], axis=0)
        result = Data(
            x=_x,
            face=_face,
            y=_y,
            pos=_pos,
            edge_index=_edge_index,
            edge_attr=_edge_attr,
        )
        # print(f"Collator: {result}")
        return result
