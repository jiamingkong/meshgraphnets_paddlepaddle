from dataset import FPC, Data, collate_fn
from paddle.io import DataLoader
from transforms import FaceToEdge, BasicDescriptor, Cartesian, Distance, Compose
from model.blocks import NodeBlock
from utils.noise import get_velocity_noise
from model import Simulator
import paddle
from utils.utils import NodeType

import warnings

warnings.filterwarnings("ignore")


noise_std = 2e-2


if __name__ == "__main__":

    simulator = Simulator(
        message_passing_num=15, node_input_size=11, edge_input_size=3, device="gpu"
    )
    optimizer = paddle.optimizer.Adam(
        parameters=simulator.parameters(), learning_rate=1e-3
    )
    # simulator.cuda()
    train_dataset = FPC(10, "sample_data/small", small_open_tra_num=1)
    # definitely broken because of the different point numbers
    train_loader = DataLoader(train_dataset, batch_size=3, collate_fn=collate_fn)
    # graph = next(iter(train_dataset))

    transforms = Compose([FaceToEdge(), Cartesian(norm=False), Distance(norm=False),])

    # for idx, graph in enumerate(train_dataset):
    #     print(f"idx = {idx}")
    #     graph = transforms(graph)
    #     # print(graph_repr(graph))
    #     print(graph)
    #     import pdb; pdb.set_trace()

    print("Starting the loop")
    # for batch_index, graph in enumerate(train_loader):
    graph = next(iter(train_loader))
    # print(f"Training {batch_index}")
    graph = transforms(graph)
    print(graph)
    import pdb

    pdb.set_trace()
    node_type = graph.x[:, 0]
    velocity_sequence_noise = get_velocity_noise(graph, 2e-2)

    predicted_acc, target_acc = simulator(graph, velocity_sequence_noise)
    mask = paddle.logical_or(
        node_type == NodeType.NORMAL, node_type == NodeType.OUTFLOW
    )

    errors = ((predicted_acc - target_acc) ** 2)[mask]
    loss = paddle.mean(errors)
    print(loss)

    # optimizer.clear_grad()
    # loss.backward()
    # optimizer.step()
    # print(f"batch {batch_index} loss: {loss.numpy()}")
