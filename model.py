import torch.nn as nn
import torch


class Generator(nn.Module):
    def __init__(
        self, input_dim: int, output_dim: int, filter_size: int, seq_len: int, flag: str
    ):
        super().__init__()
        self.input_dim = input_dim
        self.flag = flag

        self.conv_layer_input = nn.Conv1d(
            input_dim,
            input_dim * output_dim,
            int(filter_size / 2),
            padding="valid",
            groups=input_dim,
        )
        self.conv_layer_hidden = nn.Conv1d(
            input_dim * output_dim,
            input_dim * output_dim,
            1,
            padding="valid",
            groups=input_dim,
        )
        self.conv_layer_output = nn.Conv1d(
            input_dim * output_dim,
            input_dim,
            int(filter_size / 2) + 1,
            padding="valid",
            groups=input_dim,
        )
        self.feature_conv_input = nn.Conv2d(
            1,
            4,
            (1, input_dim),
            padding="valid",
        )
        self.feature_conv_hidden = nn.Conv2d(
            4,
            4,
            (1, 1),
            padding="valid",
        )
        self.feature_conv_output = nn.Conv2d(
            4,
            1,
            (1, 1),
            padding="valid",
        )
        self.tanh = nn.Tanh()
        self.relu = nn.ReLU()
        self.leakyrelu = nn.LeakyReLU()
        if self.flag not in {"decoder", "encoder"}:
            raise ValueError("flag can only be encoder or decoder")

    def forward(self, input: torch.Tensor):
        if self.flag == "encoder":
            output = self.conv_layer_input(input)
            output = self.tanh(output)
            output = self.conv_layer_hidden(output)
            output = self.tanh(output)
            output = self.conv_layer_output(output)
            output = self.tanh(output)
            output = output.transpose(1, 2)
            output = nn.functional.pad(output, (self.input_dim - 1, 0))
            output = self.feature_conv_input(output.unsqueeze(1))
            output = self.tanh(output)
            output = self.feature_conv_hidden(output)
            output = self.tanh(output)
            output = self.feature_conv_output(output)
            output = self.tanh(output)
            output = output.squeeze(1)
            output = output.transpose(1, 2)
            output = self.tanh(output)
        elif self.flag == "decoder":
            output = input.transpose(1, 2)
            output = nn.functional.pad(output, (self.input_dim - 1, 0))
            output = self.feature_conv_input(output.unsqueeze(1))
            output = self.leakyrelu(output)
            output = self.feature_conv_hidden(output)
            output = self.leakyrelu(output)
            output = self.feature_conv_output(output)
            output = self.leakyrelu(output)
            output = output.squeeze(1)
            output = output.transpose(1, 2)
            output = self.conv_layer_input(output)
            output = self.leakyrelu(output)
            output = self.conv_layer_hidden(output)
            output = self.leakyrelu(output)
            output = self.conv_layer_output(output)
            output = self.leakyrelu(output)

        return output


class Discriminator(nn.Module):
    def __init__(self, input_dim: int, hidden_dim: int):
        super().__init__()
        self.input_layer = nn.Linear(input_dim, hidden_dim)
        self.hidden_layer = nn.Linear(hidden_dim, hidden_dim)
        self.output_layer = nn.Linear(hidden_dim, 1)
        self.tanh = nn.Tanh()
        self.LeakyRelu = nn.LeakyReLU()

        self.main = nn.Sequential(
            self.input_layer,
            self.tanh,
            self.hidden_layer,
            self.tanh,
            self.output_layer,
            self.LeakyRelu,
        )

    def forward(self, input):
        output = self.main(input)
        output = output.flatten()
        # output = output.mean()
        return output
