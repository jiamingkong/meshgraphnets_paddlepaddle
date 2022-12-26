# import torch
# import torch.nn as nn
import paddle
import paddle.nn as nn


class Normalizer(nn.Layer):
    def __init__(
        self,
        size,
        max_accumulations=10 ** 6,
        std_epsilon=1e-8,
        name="Normalizer",
        device="cuda",
    ):
        super(Normalizer, self).__init__()
        self.name = name
        self._max_accumulations = max_accumulations
        # torch to paddle migration
        self._std_epsilon = paddle.to_tensor(
            std_epsilon, dtype=paddle.float32, stop_gradient=True
        )
        self._acc_count = paddle.to_tensor(0, dtype=paddle.float32, stop_gradient=True)
        self._num_accumulations = paddle.to_tensor(
            0, dtype=paddle.float32, stop_gradient=True
        )
        self._acc_sum = paddle.zeros((1, size), dtype=paddle.float32)
        self._acc_sum_squared = paddle.zeros((1, size), dtype=paddle.float32)
        # stop gradient for _acc_sum and _acc_sum_squared
        self._acc_sum.stop_gradient = True
        self._acc_sum_squared.stop_gradient = True
        # if device == "cpu":
        #     self._std_epsilon = self._std_epsilon.cpu()
        #     self._acc_count = self._acc_count.cpu()
        #     self._num_accumulations = self._num_accumulations.cpu()
        #     self._acc_sum = self._acc_sum.cpu()
        #     self._acc_sum_squared = self._acc_sum_squared.cpu()
        # else:
        #     self._std_epsilon = self._std_epsilon.cuda()
        #     self._acc_count = self._acc_count.cuda()
        #     self._num_accumulations = self._num_accumulations.cuda()
        #     self._acc_sum = self._acc_sum.cuda()
        #     self._acc_sum_squared = self._acc_sum_squared.cuda()

    def forward(self, batched_data, accumulate=True):
        """Normalizes input data and accumulates statistics."""
        if accumulate:
            # stop accumulating after a million updates, to prevent accuracy issues
            if self._num_accumulations < self._max_accumulations:
                self._accumulate(batched_data.detach())
        return (batched_data - self._mean()) / self._std_with_epsilon()

    def inverse(self, normalized_batch_data):
        """Inverse transformation of the normalizer."""
        return normalized_batch_data * self._std_with_epsilon() + self._mean()

    def _accumulate(self, batched_data):
        """Function to perform the accumulation of the batch_data statistics."""
        count = batched_data.shape[0]
        # torch to paddle migration
        # data_sum = torch.sum(batched_data, axis=0, keepdims=True)
        # squared_data_sum = torch.sum(batched_data ** 2, axis=0, keepdims=True)
        data_sum = paddle.sum(batched_data, axis=0, keepdim=True)
        squared_data_sum = paddle.sum(batched_data ** 2, axis=0, keepdim=True)

        self._acc_sum += data_sum
        self._acc_sum_squared += squared_data_sum
        self._acc_count += count
        self._num_accumulations += 1

    def _mean(self):
        # safe_count = torch.maximum(
        #     self._acc_count,
        #     torch.tensor(1.0, dtype=torch.float32, device=self._acc_count.device),
        # )
        safe_count = paddle.maximum(
            self._acc_count, paddle.to_tensor(1.0, dtype=paddle.float32),
        )

        return self._acc_sum / safe_count

    def _std_with_epsilon(self):
        safe_count = paddle.maximum(
            self._acc_count, paddle.to_tensor(1.0, dtype=paddle.float32),
        )

        std = paddle.sqrt(self._acc_sum_squared / safe_count - self._mean() ** 2)

        return paddle.maximum(std, self._std_epsilon)

    def get_variable(self):

        dict = {
            "_max_accumulations": self._max_accumulations,
            "_std_epsilon": self._std_epsilon,
            "_acc_count": self._acc_count,
            "_num_accumulations": self._num_accumulations,
            "_acc_sum": self._acc_sum,
            "_acc_sum_squared": self._acc_sum_squared,
            "name": self.name,
        }

        return dict
