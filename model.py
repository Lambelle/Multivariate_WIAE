import torch.nn as nn
import torch


class Generator(nn.Module):
    def __init__(
        self, input_dim: int, output_dim: int, filter_size: int, seq_len: int, flag: str
    ):
        super().__init__()
        self.input_dim = input_dim
        self.input_dim_after_conv = seq_len - filter_size + 1
        self.flag = flag
        self.num_layer = 3

        self.conv_layer_input = nn.Conv1d(
            input_dim,
            100,
            filter_size,
            padding="valid",
            # groups=input_dim,
        )
        nn.init.xavier_uniform_(self.conv_layer_input.weight)
        self.conv_layer_hidden = nn.Conv1d(
            100,
            50,
            1,
            padding="valid",
            # groups=input_dim,
        )
        nn.init.xavier_uniform_(self.conv_layer_hidden.weight)
        self.conv_layer_output = nn.Conv1d(
            50,
            output_dim,
            1,
            padding="valid",
            # groups=input_dim,
        )
        nn.init.xavier_uniform_(self.conv_layer_output.weight)
        self.feature_conv_input = nn.Conv2d(
            1,
            32,
            (1, input_dim),
            padding="valid",
        )
        nn.init.xavier_uniform_(self.feature_conv_input.weight)
        self.feature_conv_hidden = nn.Conv2d(
            32,
            32,
            (1, 1),
            1,
            padding="valid",
        )
        nn.init.xavier_uniform_(self.feature_conv_hidden.weight)
        self.feature_conv_output = nn.Conv2d(
            32,
            1,
            (1, 1),
            padding="valid",
        )
        nn.init.xavier_uniform_(self.feature_conv_output.weight)
        self.feature_rnn = nn.RNN(
            self.input_dim_after_conv,
            self.input_dim_after_conv,
            num_layers=self.num_layer,
        )

        # Activation
        self.tanh = nn.Tanh()
        self.relu = nn.ReLU()
        self.leakyrelu = nn.LeakyReLU()

        # BatchNorm
        self.batchnorm_100 = nn.BatchNorm1d(100)
        self.batchnorm_50 = nn.BatchNorm1d(50)

        if self.flag not in {"decoder", "encoder"}:
            raise ValueError("flag can only be encoder or decoder")

    def forward(self, input: torch.Tensor):
        if self.flag == "encoder":
            output = self.conv_layer_input(input)
            output = self.tanh(output)
            # output = self.batchnorm1D(output)
            output = self.conv_layer_hidden(output)
            output = self.tanh(output)
            # output = self.batchnorm1D(output)
            output = self.conv_layer_output(output)
            # output = self.relu(output)

            # Use RNN
            # h0 = torch.rand(self.num_layer,self.input_dim_after_conv,self.input_dim)
            # output,hn = self.feature_rnn(output,h0)

            # Using 2 CONV
            # output = output.transpose(1, 2)
            # output = nn.functional.pad(output, (self.input_dim - 1, 0))
            # output = self.feature_conv_input(output.unsqueeze(1))
            # output = self.batchnorm2D(output)
            # # output = self.tanh(output)
            # output = self.feature_conv_hidden(output)
            # output = self.batchnorm2D(output)
            # # output = self.tanh(output)
            # output = self.feature_conv_output(output)
            # output = self.tanh(output)
            # output = output.squeeze(1)
            # output = output.transpose(1, 2)
        elif self.flag == "decoder":

            # Use RNN
            # output = input.transpose(1,2)
            # h0 = torch.rand(self.num_layer,self.input_dim_after_conv,self.input_dim)
            # output, hn = self.feature_rnn(output, h0)
            # output = output.transpose(1, 2)

            # Use Conv2D
            # output = input.transpose(1, 2)
            # output = nn.functional.pad(output, (self.input_dim - 1, 0))
            # output = self.feature_conv_input(output.unsqueeze(1))
            # output = self.batchnorm2D(output)
            # output = self.relu(output)
            # output = self.feature_conv_hidden(output)
            # output = self.batchnorm2D(output)
            # output = self.relu(output)
            # output = self.feature_conv_output(output)
            # output = self.relu(output)
            # output = output.squeeze(1)
            # output = output.transpose(1, 2)

            output = input
            output = self.conv_layer_input(output)
            output = self.tanh(output)
            output = self.batchnorm_100(output)
            output = self.conv_layer_hidden(output)
            output = self.batchnorm_50(output)
            # output = self.relu(output)
            output = self.conv_layer_output(output)
            # output = self.leakyrelu(output)

        return output


class Discriminator(nn.Module):
    def __init__(self, input_dim: int, hidden_dim: int):
        super().__init__()
        self.input_layer = nn.Linear(input_dim, hidden_dim)
        self.hidden_layer = nn.Linear(hidden_dim, hidden_dim)
        self.output_layer = nn.Linear(hidden_dim, 1)
        self.tanh = nn.Tanh()
        self.LeakyRelu = nn.LeakyReLU()
        self.relu = nn.ReLU()
        self.sigmoid = nn.Sigmoid()
        self.input_conv = nn.Conv2d(
            1, out_channels=4, kernel_size=(2, 7), padding="same"
        )
        nn.init.xavier_uniform_(self.input_conv.weight)
        self.hidden_conv = nn.Conv2d(
            4, out_channels=4, kernel_size=(2, 7), padding="same"
        )
        nn.init.xavier_uniform_(self.hidden_conv.weight)
        self.output_conv = nn.Conv2d(
            4, out_channels=1, kernel_size=(1, 7), padding="same"
        )
        nn.init.xavier_uniform_(self.output_conv.weight)

        self.main = nn.Sequential(
            self.input_layer,
            self.relu,
            self.hidden_layer,
            self.relu,
            self.output_layer,
            self.LeakyRelu,
            # self.relu,
        )

    def forward(self, input):

        # output = input.flatten(start_dim=1,end_dim=2)
        # output = self.main(output)
        output = self.input_conv(input.unsqueeze(1))
        output = self.tanh(output)
        output = self.hidden_conv(output)
        output = self.output_conv(output)
        output = output.squeeze(1)
        output = output.flatten(start_dim=1, end_dim=2)

        return output
