# Chat Export

**Exported on:** 9/26/2026, 11:44:43 AM

---

## User

You said: i want to create a similar implementation for lstm from torch.

i want to create a similar implementation for lstm from torch. i want to implement it from scratch myself so i learn the details. and then in the end just make a sanity check to the original implementation of torch. because not sure how to implement the recurrence.

can you guide me step by step

2 hours ago




---

## Assistant

Claude responded: The best way to do this is to build it bottom-up: first a single time step (the "cell"), then the loop over time (the recurrence), then a check against nn.LSTM.
Planning a step-by-step LSTM implementation from scratch in PyTorch.
Planning a step-by-step LSTM implementation from scratch in PyTorch.

The best way to do this is to build it bottom-up: first a single time step (the "cell"), then the loop over time (the recurrence), then a check against nn.LSTM. If you match PyTorch's weight layout exactly, the sanity check is simple because you can copy the weights over.

Step 1: The math PyTorch uses

At each time step t, given input x_t, previous hidden state h_{t-1}, and previous cell state c_{t-1}:


i_t = σ(W_ii x_t + b_ii + W_hi h_{t-1} + b_hi)     # input gate
f_t = σ(W_if x_t + b_if + W_hf h_{t-1} + b_hf)     # forget gate
g_t = tanh(W_ig x_t + b_ig + W_hg h_{t-1} + b_hg)  # candidate cell
o_t = σ(W_io x_t + b_io + W_ho h_{t-1} + b_ho)     # output gate

c_t = f_t * c_{t-1} + i_t * g_t
h_t = o_t * tanh(c_t)

Two details matter for matching PyTorch:

The four gates are stacked into one matrix. PyTorch doesn't store 8 separate weight matrices. It stores weight_ih with shape (4*H, input_size) and weight_hh with shape (4*H, H). The gate order along the first dimension is i, f, g, o. Get this wrong and the sanity check fails.
There are two bias vectors, bias_ih and bias_hh, each of shape (4*H,). This is mathematically redundant, since you could merge them, but you need both to copy weights over.
Step 2: Define the parameters

Try writing this yourself first. You need a module holding four parameters with the shapes above, initialized the way PyTorch does it: uniform in [-1/√H, 1/√H].


python
import torch
import torch.nn as nn
import math

class MyLSTM(nn.Module):
    def __init__(self, input_size, hidden_size):
        super().__init__()
        self.input_size = input_size
        self.hidden_size = hidden_size
        H = hidden_size
        self.weight_ih = nn.Parameter(torch.empty(4 * H, input_size))
        self.weight_hh = nn.Parameter(torch.empty(4 * H, H))
        self.bias_ih = nn.Parameter(torch.empty(4 * H))
        self.bias_hh = nn.Parameter(torch.empty(4 * H))
        self.reset_parameters()

    def reset_parameters(self):
        k = 1.0 / math.sqrt(self.hidden_size)
        for p in self.parameters():
            nn.init.uniform_(p, -k, k)
Step 3: One time step (the cell)

This is the core. Because the gates are stacked, you can compute all four pre-activations with two matrix multiplies and then split the result into four chunks.

Try it yourself with this hint: x_t has shape (batch, input_size) and h has shape (batch, H). What do you multiply with weight_ih (transposed?) to get (batch, 4H)?


python
    def cell(self, x_t, h, c):
        gates = x_t @ self.weight_ih.T + self.bias_ih + h @ self.weight_hh.T + self.bias_hh
        i, f, g, o = gates.chunk(4, dim=1)   # order matters: i, f, g, o

        i = torch.sigmoid(i)
        f = torch.sigmoid(f)
        g = torch.tanh(g)
        o = torch.sigmoid(o)

        c_next = f * c + i * g
        h_next = o * torch.tanh(c_next)
        return h_next, c_next
Step 4: The recurrence

This is the part you were unsure about. The idea is that the recurrence is just a Python for loop over time, where the output state of step t becomes the input state of step t+1. Nothing more magical than that.

Keep these shape conventions in mind. By default (batch_first=False), PyTorch's input is (seq_len, batch, input_size). It returns output, which contains every h_t stacked as (seq_len, batch, H), and (h_n, c_n), the final states, each shaped (num_layers, batch, H), which is (1, batch, H) for a single layer. If no initial state is given, it starts from zeros.


python
    def forward(self, x, state=None):
        seq_len, batch, _ = x.shape
        H = self.hidden_size

        if state is None:
            h = x.new_zeros(batch, H)
            c = x.new_zeros(batch, H)
        else:
            h, c = state
            h, c = h[0], c[0]          # drop the num_layers dim

        outputs = []
        for t in range(seq_len):
            h, c = self.cell(x[t], h, c)   # state is carried forward
            outputs.append(h)

        output = torch.stack(outputs, dim=0)          # (seq_len, batch, H)
        return output, (h.unsqueeze(0), c.unsqueeze(0))

Notice that h and c get overwritten each iteration. That is the recurrence. Autograd tracks the whole chain automatically, so backprop through time comes for free.

Step 5: Sanity check against nn.LSTM

Create both models, copy PyTorch's weights into yours so they're identical, run the same input through both, and compare. PyTorch names its parameters weight_ih_l0, weight_hh_l0, bias_ih_l0, and bias_hh_l0, where l0 means layer 0.


python
torch.manual_seed(0)
input_size, hidden_size, seq_len, batch = 5, 8, 7, 3

ref = nn.LSTM(input_size, hidden_size)
mine = MyLSTM(input_size, hidden_size)

with torch.no_grad():
    mine.weight_ih.copy_(ref.weight_ih_l0)
    mine.weight_hh.copy_(ref.weight_hh_l0)
    mine.bias_ih.copy_(ref.bias_ih_l0)
    mine.bias_hh.copy_(ref.bias_hh_l0)

x = torch.randn(seq_len, batch, input_size)
h0 = torch.randn(1, batch, hidden_size)
c0 = torch.randn(1, batch, hidden_size)

out_ref, (h_ref, c_ref) = ref(x, (h0, c0))
out_mine, (h_mine, c_mine) = mine(x, (h0, c0))

print(torch.allclose(out_ref, out_mine, atol=1e-6))
print(torch.allclose(h_ref, h_mine, atol=1e-6))
print(torch.allclose(c_ref, c_mine, atol=1e-6))

To check gradients as well, which is a good test of your understanding, call .sum().backward() on both outputs and compare ref.weight_ih_l0.grad with mine.weight_ih.grad.

If the check fails, the usual culprits are:

the wrong gate order (it must be i, f, g, o),
forgetting the transpose on the weights,
returning h and c without the leading num_layers dimension,
comparing with a tolerance that's too strict (use atol=1e-6 or 1e-5, not exact equality).
Step 6: Extensions, once the basic version passes
batch_first=True: transpose the input at the start and the output at the end.
Multiple layers: keep a list of weight sets. The output sequence of layer l becomes the input x of layer l+1, and h_n/c_n stack the final states of all layers.
Bidirectional: run a second set of weights over x.flip(0), flip its outputs back, and concatenate along the feature dimension.

Each of these can be sanity-checked the same way against nn.LSTM(..., num_layers=2) or bidirectional=True, using parameter names like weight_ih_l1 and weight_ih_l0_reverse.

I'd suggest writing Steps 3 and 4 yourself before looking at my versions. If your check fails, paste your code and the result and we can debug it together.






2 hours ago

---

## User

You said: ok that sounds good.

ok that sounds good. but later dont need the full set of options like batch first and so on. just a basic version. 

the init of the weights i want to move outside of my version. later i want to create a class called LSTM_Model where we just switch the basic version of mine with the implementation of torch. and in this class we want the init. because init is important for weights and want to define that for the model itself.

whats a standard init function for weights that works on time series models. can we generalize that and use that for every model even transformers?

---

## Assistant

Claude responded: Moving the init out of the LSTM class is a good design choice.
Switching to module-type dispatch for broader initialization coverage.

Switching to module-type dispatch for broader initialization coverage.

Moving the init out of the LSTM class is a good design choice. The LSTM layer then only defines structure, and the model decides how it starts. It also means your version and nn.LSTM get initialized by exactly the same rules, which makes comparing them fairer.

The basic MyLSTM, without its own init

This is the same as before, minus reset_parameters:


python
import torch
import torch.nn as nn

class MyLSTM(nn.Module):
    def __init__(self, input_size, hidden_size):
        super().__init__()
        self.hidden_size = hidden_size
        H = hidden_size
        self.weight_ih = nn.Parameter(torch.empty(4 * H, input_size))
        self.weight_hh = nn.Parameter(torch.empty(4 * H, H))
        self.bias_ih = nn.Parameter(torch.empty(4 * H))
        self.bias_hh = nn.Parameter(torch.empty(4 * H))

    def cell(self, x_t, h, c):
        gates = x_t @ self.weight_ih.T + self.bias_ih + h @ self.weight_hh.T + self.bias_hh
        i, f, g, o = gates.chunk(4, dim=1)
        c = torch.sigmoid(f) * c + torch.sigmoid(i) * torch.tanh(g)
        h = torch.sigmoid(o) * torch.tanh(c)
        return h, c

    def forward(self, x, state=None):
        seq_len, batch, _ = x.shape
        if state is None:
            h = x.new_zeros(batch, self.hidden_size)
            c = x.new_zeros(batch, self.hidden_size)
        else:
            h, c = state[0][0], state[1][0]
        outputs = []
        for t in range(seq_len):
            h, c = self.cell(x[t], h, c)
            outputs.append(h)
        return torch.stack(outputs), (h.unsqueeze(0), c.unsqueeze(0))

torch.empty leaves the memory uninitialized (random garbage), so this class is now only safe to use inside a model that always runs the init.

A standard init for LSTMs

The widely used recipe for recurrent models has four parts:

Input-to-hidden weights (weight_ih): Xavier/Glorot uniform. It keeps the variance of activations roughly constant from layer to layer, which suits sigmoid and tanh.
Hidden-to-hidden weights (weight_hh): orthogonal. This is the important one for time series. The same matrix is multiplied in at every time step, and an orthogonal matrix neither shrinks nor blows up the hidden state over many steps. That helps against vanishing and exploding gradients.
Biases: zero, except the forget gate bias, which is set to 1. This makes the LSTM "remember by default" at the start of training, so gradients can flow through long sequences before the model has learned anything. This trick comes from Jozefowicz et al. (2015) and is Keras' default. Since PyTorch has two bias vectors, set it in only one of them, so the effective bias is 1 rather than 2.
Per-gate application: Xavier and orthogonal are best applied to each gate's H × H block separately, rather than to the stacked 4H × H matrix.
Can one init work for every model, even transformers?

Partly. The general principle carries over: variance-preserving inits (Xavier for tanh/sigmoid/linear, He/Kaiming for ReLU) are a sensible default for almost any layer. But different architectures have specific needs:

Recurrent layers want orthogonal recurrent weights and the forget-bias trick, as above.
Transformers usually use normal(0, 0.02) for Linear and Embedding layers (the GPT/BERT convention) or Xavier uniform (the original Transformer and PyTorch's nn.Transformer). LayerNorm starts at weight 1 and bias 0. GPT-2 additionally scales down the output projections of residual blocks by 1/√(2·n_layers) so the residual stream doesn't grow with depth.

So the practical way to generalize is one function that dispatches on layer type, not one distribution for everything. You apply it with model.apply(...), which visits every submodule. When you later build a transformer, you just add or adjust branches.

This version matches LSTM parameters by name, so it works for both your class (weight_ih) and nn.LSTM (weight_ih_l0, weight_ih_l1, ...):


python
def init_weights(module):
    if isinstance(module, (nn.LSTM, MyLSTM)):
        with torch.no_grad():
            for name, p in module.named_parameters():
                if "weight_ih" in name:
                    for gate in p.chunk(4, dim=0):
                        nn.init.xavier_uniform_(gate)
                elif "weight_hh" in name:
                    for gate in p.chunk(4, dim=0):
                        nn.init.orthogonal_(gate)
                elif "bias_ih" in name:
                    nn.init.zeros_(p)
                    H = p.shape[0] // 4
                    p[H:2 * H].fill_(1.0)      # forget gate (order: i, f, g, o)
                elif "bias_hh" in name:
                    nn.init.zeros_(p)

    elif isinstance(module, nn.Linear):
        nn.init.xavier_uniform_(module.weight)
        if module.bias is not None:
            nn.init.zeros_(module.bias)

    elif isinstance(module, nn.Embedding):
        nn.init.normal_(module.weight, mean=0.0, std=0.02)

    elif isinstance(module, nn.LayerNorm):
        nn.init.ones_(module.weight)
        nn.init.zeros_(module.bias)

The Embedding and LayerNorm branches do nothing for the LSTM model, but they're there for when you reuse this with transformers.

Branching by module type rather than by parameter shape matters. A naive rule like "every 1-D parameter gets zeros" would silently set LayerNorm weights to 0 and kill the layer.

The LSTM_Model wrapper

This version predicts one output from the last time step, which is typical for forecasting:


python
class LSTM_Model(nn.Module):
    def __init__(self, input_size, hidden_size, output_size, use_torch=False):
        super().__init__()
        if use_torch:
            self.lstm = nn.LSTM(input_size, hidden_size)
        else:
            self.lstm = MyLSTM(input_size, hidden_size)
        self.head = nn.Linear(hidden_size, output_size)
        self.apply(init_weights)

    def forward(self, x):                  # x: (seq_len, batch, input_size)
        output, (h_n, c_n) = self.lstm(x)
        return self.head(output[-1])       # last time step -> (batch, output_size)

If you instead want a prediction at every time step, use self.head(output), which gives (seq_len, batch, output_size).

nn.LSTM runs its own default init in its constructor, but self.apply(init_weights) runs afterward and overwrites it. That way both variants really follow your scheme. Your earlier sanity check still works unchanged, since it copies weights from one LSTM to the other directly.

Once this runs, the natural next step is to train both variants on the same small time series (a noisy sine wave works well) and check that the loss curves look the same.

---

## User

Claude responded: Yes.
Comparing autoregressive and chunked prediction strategies.
Comparing autoregressive and chunked prediction strategies.

Yes. That's the autoregressive idea from before, combined with the streaming state. And with your direct 10-step head, you can even do it in chunks: predict 10 steps, feed those 10 predictions back in as if they were observations, predict the next 10, and so on. Many forecasting models extend their horizon this way.

As a method on the Streamer, for a univariate series (input_size=1):


python
    @torch.no_grad()
    def long_forecast(self, n_chunks):
        state = self.state                             # local reference
        pred = self.model.head(state[0][-1])           # (batch, 10), from the current h
        preds = [pred]
        for _ in range(n_chunks - 1):
            output, state = self.model.lstm(pred.unsqueeze(-1), state)  # (batch, 10, 1) fed back
            pred = self.model.head(output[:, -1])
            preds.append(pred)
        return torch.cat(preds, dim=1)                 # (batch, 10 * n_chunks)

This keeps the real stream clean. The loop only reassigns the local state; the LSTM returns new tensors and never modifies the old ones in place, so self.state stays exactly as it was. That's important: predictions must never end up in the stored state, or the next real observation would continue from a memory that contains guesses. When the next real value arrives, you call update as usual and it continues from the true history.

Keep the limits in mind, because they're the same as for autoregression:

Errors compound. Each chunk is built on the previous chunk's guesses, so quality drops the further out you go.
Predictions don't look like real data. Forecasts are usually smoother than real observations, since the model predicts something like the expected value, without noise. Feeding smooth values back is a distribution shift the model never saw in training, which can make long forecasts flatten out or drift.
Extra input features become a problem. With covariates, you'd need their future values for each chunk you feed back.

So it works, and it's a legitimate technique, but a forecast of 10 × 10 steps will be noticeably less reliable than the first 10. A good exercise is to measure that yourself: on a validation series, compare the error of chunk 1, chunk 2, chunk 3, and so on, and watch how fast it grows.






20 minutes ago

---

## Assistant

You said: so in theory i could train my forecast model on different sequences with fixed lengths.

so in theory i could train my forecast model on different sequences with fixed lengths. then i fix the weights. and stream my predictions to whatever forecast length i need

15 minutes ago




---

## User

Claude responded: Yes, that's the workflow, and it's essentially how pretrained forecasting models are used: train once on fixed-length windows, freeze the weights, and at infer…
Explaining zero-shot forecasting with foundation models and their limits.
Explaining zero-shot forecasting with foundation models and their limits.

Yes, that's the workflow, and it's essentially how pretrained forecasting models are used: train once on fixed-length windows, freeze the weights, and at inference stream real data in and roll predictions forward for whatever horizon you need. Nothing about the architecture limits the horizon, because the recurrence can run as long as you like.

"Can run" isn't the same as "stays accurate", though. Two gaps between training and this kind of use are worth closing.

1. The model never practiced on its own predictions. In training it always sees real data as input, but in a long rollout it sees its own smooth, slightly wrong outputs. You can train for that directly. Instead of computing the loss only on the first 10-step chunk, roll the model forward for a few chunks during training, feeding its predictions back, and compute the loss on all chunks:


python
def rollout_loss(model, x, y, n_chunks, loss_fn):
    # x: (batch, seq_len, 1), y: (batch, 10 * n_chunks) real future values
    output, state = model.lstm(x)
    pred = model.head(output[:, -1])
    preds = [pred]
    for _ in range(n_chunks - 1):
        output, state = model.lstm(pred.unsqueeze(-1), state)
        pred = model.head(output[:, -1])
        preds.append(pred)
    return loss_fn(torch.cat(preds, dim=1), y)

Gradients flow back through the fed-back predictions, so the model learns to produce outputs that are good to continue from, not just good in isolation. A common schedule is to start training with 1 chunk and increase n_chunks gradually, since long rollouts early in training are unstable.

2. Streaming runs much longer than training windows. If every training sequence has length 100, the state only ever learned to handle 100 steps of history. Training on varied sequence lengths helps, as does measuring how forecasts behave after very long streams. This is the part TiRex-2 made a point of showing works for their model; for your own LSTM you'd have to verify it.

A realistic expectation is that the approach works well for horizons a few times longer than what you trained on, and it degrades gracefully after that. Rather than guessing where your limit is, measure the error per chunk on held-out data, as mentioned before. That tells you directly how far out your model's forecasts are still useful.






15 minutes ago

