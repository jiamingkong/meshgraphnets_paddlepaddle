from .model import EncoderProcesserDecoder

# import torch.nn as nn
# import torch
# from torch_geometric.data import Data

import paddle
import paddle.nn.functional as F
from paddle import nn
from utils import normalization
import os


class Simulator(nn.Layer):
    def __init__(
        self,
        message_passing_num,
        node_input_size,
        edge_input_size,
        device,
        model_dir="checkpoint/",
    ) -> None:
        super(Simulator, self).__init__()

        self.node_input_size = node_input_size
        self.edge_input_size = edge_input_size
        self.model_dir = model_dir
        self.model = EncoderProcesserDecoder(
            message_passing_num=message_passing_num,
            node_input_size=node_input_size,
            edge_input_size=edge_input_size,
        )
        self._output_normalizer = normalization.Normalizer(
            size=2, name="output_normalizer", device=device
        )
        self._node_normalizer = normalization.Normalizer(
            size=node_input_size, name="node_normalizer", device=device
        )

        print("Simulator model initialized")

    def update_node_attr(self, frames, types: paddle.Tensor):
        node_feature = []

        node_feature.append(frames)  # velocity
        node_type = paddle.squeeze(types.astype(paddle.int64))
        one_hot = paddle.nn.functional.one_hot(node_type, 9)
        node_feature.append(one_hot)
        node_feats = paddle.concat(node_feature, axis=1)
        attr = self._node_normalizer(node_feats, self.training)

        return attr

    def velocity_to_accelation(self, noised_frames, next_velocity):

        acc_next = next_velocity - noised_frames
        return acc_next

    def forward(self, graph: dict, velocity_sequence_noise):

        if self.training:

            # node_type = graph["x"][:, 0:1]
            # frames = graph["x"][:, 1:3]
            # target = graph["y"]
            node_type = graph.x[:, 0:1]
            frames = graph.x[:, 1:3]
            target = graph.y

            noised_frames = frames + velocity_sequence_noise
            node_attr = self.update_node_attr(noised_frames, node_type)
            # graph["x"] = node_attr
            graph.x = node_attr
            predicted = self.model(graph)

            target_acceration = self.velocity_to_accelation(noised_frames, target)
            target_acceration_normalized = self._output_normalizer(
                target_acceration, self.training
            )

            return predicted, target_acceration_normalized

        else:

            # node_type = graph["x"][:, 0:1]
            # frames = graph["x"][:, 1:3]
            node_type = graph.x[:, 0:1]
            frames = graph.x[:, 1:3]
            node_attr = self.update_node_attr(frames, node_type)
            # graph["x"] = node_attr
            graph.x = node_attr
            predicted = self.model(graph)

            velocity_update = self._output_normalizer.inverse(predicted)
            predicted_velocity = frames + velocity_update

            return predicted_velocity

    def load_checkpoint(self, filename=None):
        if filename is not None and filename.endswith(".pkl"):
            import pickle
            import numpy as np

            with open(filename, "rb") as f:
                dicts = pickle.load(f)
                model = dicts["model"]
                for key, tensor in model.items():
                    # put tensor from numpy to paddle.Tensor
                    model[key] = paddle.to_tensor(tensor)
                self.set_state_dict(model)
                keys = list(dicts.keys())
                keys.remove("model")
                for k in keys:
                    v = dicts[k]
                    object = eval("self." + k)
                    # import pdb; pdb.set_trace()
                    object.set_variables(v)
                    # set the parameters of the normalizer

                print("Simulator model loaded checkpoint %s" % filename)
            return
        if filename is None:
            # search for the largest number
            files = [i for i in os.listdir(self.model_dir) if i.endswith(".pdparams")]
            if len(files) == 0:
                print("No checkpoint found in %s" % self.model_dir)
                return
            files = [int(f.split(".")[0]) for f in files]
            files.sort()
            ckpdir = os.path.join(self.model_dir, str(files[-1]) + ".pdparams")
        else:
            ckpdir = filename
        # dicts = torch.load(ckpdir)
        dicts = paddle.load(ckpdir)
        # self.load_state_dict(dicts["model"])
        self.set_state_dict(dicts["model"])

        keys = list(dicts.keys())
        keys.remove("model")

        for k in keys:
            v = dicts[k]
            for para, value in v.items():
                object = eval("self." + k)
                setattr(object, para, value)

        print("Simulator model loaded checkpoint %s" % ckpdir)

    def save_checkpoint(self, savedir=None):
        if savedir is None:
            savedir = self.model_dir

        os.makedirs(os.path.dirname(savedir), exist_ok=True)
        # see what's in the folder, choose the largest number
        if os.path.exists(savedir):
            files = [i for i in os.listdir(savedir) if i.endswith(".pdparams")]
            if len(files) > 0:
                files = [int(f.split(".")[0]) for f in files]
                files.sort()
                savename = os.path.join(savedir, str(files[-1] + 1) + ".pdparams")
            else:
                savename = os.path.join(savedir, "0.pdparams")

        model = self.state_dict()
        # import pdb; pdb.set_trace()
        for key, tensor in model.items():
            print(f"{key:25s} Tensor={tensor.shape}")
        _output_normalizer = self._output_normalizer.get_variable()
        _node_normalizer = self._node_normalizer.get_variable()
        # _edge_normalizer = self._edge_normalizer.get_variable()

        to_save = {
            "model": model,
            "_output_normalizer": _output_normalizer,
            "_node_normalizer": _node_normalizer,
        }
        # import pdb; pdb.set_trace()
        paddle.save(to_save, savename)
        print("Simulator model saved at %s" % savename)
