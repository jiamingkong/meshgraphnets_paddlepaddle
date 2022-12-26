from dataset import FPC
from paddle.io import DataLoader
from transforms import FaceToEdge, BasicDescriptor, Cartesian, Distance, Compose
from model.blocks import NodeBlock
from utils.noise import get_velocity_noise
from model import Simulator
import paddle
from utils.utils import NodeType

noise_std = 2e-2

if __name__ == '__main__':

    simulator = Simulator(
        message_passing_num=15, node_input_size=11, edge_input_size=3, device = "gpu"
    )
    optimizer = paddle.optimizer.Adam(
        parameters=simulator.parameters(), learning_rate=1e-3
    )
    # simulator.cuda()
    train_dataset = FPC(10, 'sample_data/small', small_open_tra_num=1)
    train_loader = DataLoader(train_dataset, batch_size=2)
    # graph = next(iter(train_dataset))

    transforms = Compose([
        BasicDescriptor(),
        FaceToEdge(),
        Cartesian(norm = False),
        Distance(norm = False),
    ])
    for batch_index, graph in enumerate(train_loader):
            
        graph = transforms(graph)
        node_type = graph["x"][:, 0]
        velocity_sequence_noise = get_velocity_noise(graph, 2e-2)

        predicted_acc, target_acc = simulator(graph, velocity_sequence_noise)
        mask = paddle.logical_or(
            node_type == NodeType.NORMAL, node_type == NodeType.OUTFLOW
        )

        errors = ((predicted_acc - target_acc) ** 2)[mask]
        loss = paddle.mean(errors)
        print(loss)
        
        optimizer.clear_grad()
        loss.backward()
        optimizer.step()
        print(f"batch {batch_index} loss: {loss.numpy()}")
