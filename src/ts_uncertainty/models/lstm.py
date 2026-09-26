import torch
from torch import nn


class MyLSTM(nn.Module):
    def __init__(self, input_size:int, hidden_size:int):
        super().__init__()
        self.input_size = input_size
        self.hidden_size = hidden_size

        # stacked weight matrices for input and hidden states, order: i, f, z, o
        self.weight_ih = nn.Parameter(torch.empty(4 * hidden_size, input_size))
        self.weight_hh = nn.Parameter(torch.empty(4 * hidden_size, hidden_size))
        self.bias_ih = nn.Parameter(torch.empty(4 * hidden_size))
        self.bias_hh = nn.Parameter(torch.empty(4 * hidden_size))
        self.sigmoid = nn.Sigmoid()
        self.tanh = nn.Tanh()

    def cell(self, x_t: torch.Tensor, h: torch.Tensor, c: torch.Tensor):
        # Compute the gates and cell state for a single time step of the LSTM.
        gates = x_t @ self.weight_ih.T + self.bias_ih + h @ self.weight_hh.T + self.bias_hh
        i, f, z, o = gates.chunk(4, dim=1)   # order: i, f, z, o

        i = self.sigmoid(i)
        f = self.sigmoid(f)
        z = self.tanh(z)
        o = self.sigmoid(o)

        c_next = f * c + i * z
        h_next = o * self.tanh(c_next)
        return h_next, c_next

    def forward(self,
                x: torch.Tensor,
                state: tuple[torch.tensor, torch.tensor] | None=None):
        """
        Args:
            x: Input tensor of shape (batch, seq_len, input_size)
            state: Tuple of (h, c) where h and c are the hidden and cell states of shape (batch, hidden_size). If None, initializes to zeros.
            Returns:
                output: Tensor of shape (batch, seq_len, hidden_size)
                (h, c): Tuple of hidden and cell states
        """
        batch, seq_len, _ = x.shape
        if state is None:
            h = x.new_zeros(batch, self.hidden_size)
            c = x.new_zeros(batch, self.hidden_size)
        else:
            h, c = state

        outputs = []
        for t in range(seq_len):
            h, c = self.cell(x[:, t], h, c) # state is carried forward
            outputs.append(h)

        output = torch.stack(outputs, dim=1) # (batch, seq_len, hidden_size)
        return output, (h, c)


class LSTM_Model(nn.Module):
    def __init__(self, input_size, hidden_size, horizon_length, target_size, use_torch=True, init_constant=False):
        super().__init__()
        self.init_constant = init_constant
        self.target_size = target_size
        self.horizon_length = horizon_length
        if use_torch:
            self.lstm = nn.LSTM(input_size, hidden_size, batch_first=True)
        else:
            self.lstm = MyLSTM(input_size, hidden_size)
        self.head = nn.Linear(hidden_size, horizon_length * target_size)
        self.apply(self._init_weights)

    def _init_weights(self, module):
        for name, p in module.named_parameters(recurse=False):
            if "bias" in name:
                nn.init.zeros_(p)
            else:
                if self.init_constant:
                    nn.init.constant_(p, 0.1) # for implementation check
                else:
                    nn.init.xavier_uniform_(p)

    def forward(self, x):
        # x: (batch, seq_len, input_size)                 
        output, (h_n, c_n) = self.lstm(x)
        h_state = self.head(output[:, -1])
        return h_state.reshape(x.size(0), -1, self.target_size)  # (batch, horizon_length, target_size)