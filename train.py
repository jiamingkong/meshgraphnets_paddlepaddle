from dataset import FPC, Data, Collator, DataLoader

# from paddle.io import DataLoader
from transforms import FaceToEdge, BasicDescriptor, Cartesian, Distance, Compose
from model.blocks import NodeBlock
from utils.noise import get_velocity_noise
from model import Simulator
import paddle
from utils.utils import NodeType

# average meter
from utils.average_meter import AverageMeter

import warnings

warnings.filterwarnings("ignore")

import argparse

parser = argparse.ArgumentParser(description="Implementation of MeshGraphNets")
parser.add_argument("--gpu", action="store_true", help="use gpu")
parser.add_argument("--gpu_id", type=int, default=0, help="gpu id")
parser.add_argument("--noise_std", type=float, default=2e-2)
parser.add_argument("--model_dir", type=str, default="checkpoint")
parser.add_argument("--data_dir", type=str, default="data/cylinder_flow/datapkls")
parser.add_argument("--lr", type=float, default=1e-4)
parser.add_argument("--batch_size", type=int, default=4)
parser.add_argument("--gradient_accumulation_step", type=int, default=32)
parser.add_argument("--save_interval", type=int, default=1000)
parser.add_argument("--max_epoch", type=int, default=20)

args = parser.parse_args()

# setting gpu devices
if args.gpu:
    device = "gpu:" + str(args.gpu_id)
else:
    device = "cpu"
paddle.set_device(device)


if __name__ == "__main__":

    # The model
    am = AverageMeter("Cylinder Flow")

    simulator = Simulator(
        message_passing_num=15, node_input_size=11, edge_input_size=3, device="gpu"
    )

    # this seemed to stable the training a lot, tune it at your own risk
    clip = paddle.nn.ClipGradByNorm(clip_norm=0.5)
    optimizer = paddle.optimizer.Adam(
        parameters=simulator.parameters(), learning_rate=args.lr, grad_clip=clip
    )

    train_dataset = FPC(
        args.max_epoch, args.data_dir, split="train", small_open_tra_num=10,
    )

    # data loader
    collator = Collator()
    train_loader = DataLoader(
        train_dataset, batch_size=args.batch_size, collate_fn=collator
    )

    transforms = Compose([FaceToEdge(), Cartesian(norm=False), Distance(norm=False),])

    simulator.load_checkpoint(args.model_dir)

    print("All set up, start training")

    g_step = args.gradient_accumulation_step

    for batch, graph in enumerate(train_loader):
        graph = transforms(graph)
        node_type = graph.x[:, 0]
        velocity_sequence_noise = get_velocity_noise(graph, args.noise_std)
        predicted_acc, target_acc = simulator(graph, velocity_sequence_noise)
        mask = paddle.logical_or(
            node_type == NodeType.NORMAL, node_type == NodeType.OUTFLOW
        )

        errors = ((predicted_acc - target_acc) ** 2)[mask]
        loss = paddle.mean(errors)

        # do some gradient accumulation
        loss = loss / g_step
        loss.backward()
        if batch % g_step == 0:
            optimizer.step()
            optimizer.clear_grad()

        am.update(loss.item(), batch)
        print(am)
        if batch % args.save_interval == 0 and batch > 0:
            simulator.save_checkpoint(args.model_dir)
            print("Saved checkpoint")

    print("Training finished")
