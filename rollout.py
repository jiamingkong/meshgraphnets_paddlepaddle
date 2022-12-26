from dataset import FPC_ROLLOUT, Data, Collator, DataLoader
import paddle
import argparse
from tqdm import tqdm
import pickle
from transforms import Compose, FaceToEdge, Cartesian, Distance
from utils.utils import NodeType
import numpy as np
from model.simulator import Simulator
import os


parser = argparse.ArgumentParser(description="Implementation of MeshGraphNets")
parser.add_argument("--gpu", action="store_true", help="use gpu")

parser.add_argument("--model_dir", type=str, default="checkpoint/simulator.pth")

parser.add_argument("--test_split", type=str, default="test")
parser.add_argument("--rollout_num", type=int, default=1)

args = parser.parse_args()

# gpu devices
if args.gpu:
    device = "gpu"
else:
    device = "cpu"
paddle.set_device(device)


def rollout_error(predicteds, targets):

    number_len = targets.shape[0]
    squared_diff = np.square(predicteds - targets).reshape(number_len, -1)
    loss = np.sqrt(
        np.cumsum(np.mean(squared_diff, axis=1), axis=0) / np.arange(1, number_len + 1)
    )

    for show_step in range(0, 1000000, 50):
        if show_step < number_len:
            print("testing rmse  @ step %d loss: %.2e" % (show_step, loss[show_step]))
        else:
            break

    return loss


def rollout(model, dataloader, rollout_index=1):

    dataset.change_file(rollout_index)

    predicted_velocity = None
    mask = None
    predicteds = []
    targets = []

    for graph in tqdm(dataloader, total=600):

        graph = transformer(graph)

        if mask is None:
            node_type = graph.x[:, 0]
            mask = paddle.logical_or(
                node_type == NodeType.NORMAL, node_type == NodeType.OUTFLOW
            )
            mask = paddle.logical_not(mask)

        if predicted_velocity is not None:
            graph.x[:, 1:3] = predicted_velocity.detach()

        next_v = graph.y
        predicted_velocity = model(graph, velocity_sequence_noise=None)

        predicted_velocity[mask] = next_v[mask]

        predicteds.append(predicted_velocity.detach().cpu().numpy())
        targets.append(next_v.detach().cpu().numpy())

    crds = graph.pos.cpu().numpy()
    result = [np.stack(predicteds), np.stack(targets)]

    os.makedirs("result", exist_ok=True)
    with open("result/result" + str(rollout_index) + ".pkl", "wb") as f:
        pickle.dump([result, crds], f)

    return result


if __name__ == "__main__":

    simulator = Simulator(
        message_passing_num=15, node_input_size=11, edge_input_size=3, device=device
    )
    simulator.load_checkpoint()
    simulator.eval()

    dataset = FPC_ROLLOUT("sample_data/small", split="valid")
    transformer = Compose(
        [
            FaceToEdge(),
            Cartesian(norm=False),
            Distance(norm=False),
        ]
    )
    test_loader = DataLoader(dataset=dataset, batch_size=1)

    for i in range(args.rollout_num):
        with paddle.no_grad():
            result = rollout(simulator, test_loader, rollout_index=i)
            print("------------------------------------------------------------------")
            rollout_error(result[0], result[1])
