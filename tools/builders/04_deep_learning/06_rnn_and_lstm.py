"""Build DL-06: recurrent networks, gated memory, and honest sequence evaluation."""

import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
from nbbuild import build, code, md  # noqa: E402


cells = [
    md(r"""
    # DL-06 · Recurrent Neural Networks and LSTMs

    **Prerequisites:** FND-01, FND-04, DL-01, DL-02, DL-03, and DL-04  
    **Helpful connection:** DL-05 showed weight sharing across image locations  
    **Estimated mastery time:** 10–13 hours, including exercises  
    **Next lesson:** DL-07 · The Attention Mechanism

    A normal neural network receives all its inputs at once. A recurrent neural
    network, or **RNN**, reads a sequence one step at a time. After every step it keeps
    a small numerical summary called a **hidden state**.

    Think of taking notes during a long conversation. Each new sentence changes your
    notes. You do not store every sound; you update a compact summary. That is useful,
    but it creates a hard question: what if an important fact from the beginning is
    slowly erased? This lesson follows that problem from one recurrence calculation to
    the LSTM's gated memory, then to the bottleneck that motivates attention.

    ### Scope

    We use synthetic numeric sequences so no later NLP lesson is required. Tokenization,
    vocabularies, and word embeddings begin in NLP-01. Here the goal is the reusable
    sequence machinery underneath text, sensor, audio, and time-series models.
    """),

    md(r"""
    ## 1 · What you will be able to do

    By the end, you will be able to:

    - distinguish a sequence step, feature, batch, hidden state, and layer;
    - trace `(batch, time, features)` through `nn.RNN`, `nn.GRU`, and `nn.LSTM`;
    - calculate one RNN update by hand;
    - explain why one parameter set is reused at every time step;
    - unroll an RNN and connect it to backpropagation through time, or **BPTT**;
    - calculate how repeated derivatives can vanish or explode;
    - implement a vanilla RNN forward and backward pass in NumPy;
    - verify its gradients with finite differences;
    - calculate every gate in one LSTM step;
    - explain what the forget, input, candidate, and output paths do;
    - explain why an LSTM helps gradient flow without claiming it guarantees memory;
    - train and compare RNN and LSTM models without using test data for selection;
    - handle variable-length batches with padding, lengths, masks, and packing;
    - decide when recurrence, a temporal CNN, a tree model, or attention is appropriate;
    - identify causal leakage, state leakage, hidden-state shape errors, and unstable training.

    ```mermaid
    flowchart LR
        A[One sequence step] --> B[Shared recurrent cell]
        B --> C[Unrolled computation]
        C --> D[BPTT]
        D --> E[Vanishing or exploding gradients]
        E --> F[LSTM gated cell state]
        F --> G[PyTorch sequence contract]
        G --> H[Honest evaluation]
        H --> I[Padding and streaming state]
        I --> J[Why attention comes next]
    ```
    """),

    md(r"""
    ## 2 · What problem are we solving?

    Some observations only make sense in order:

    - a machine temperature rising for six minutes;
    - the words “not approved” rather than “approved”;
    - electricity use over the last 24 hours;
    - a heartbeat waveform;
    - a customer's actions during one session.

    A fixed-window MLP can concatenate several steps, but its input width and parameter
    count depend on the chosen window. A temporal CNN can reuse local filters and is
    often excellent, but a shallow CNN sees only a limited receptive field. Neither
    statement means MLPs or CNNs “cannot” model sequences. It means their assumptions
    differ.

    An RNN makes this assumption:

    > The same update rule can read each step, while a fixed-size state carries useful
    > information forward.

    That gives variable-length processing and natural streaming. The cost is a serial
    dependency: step $t$ needs the state from step $t-1$.

    ### Common input–output patterns

    | Pattern | Example | Output used |
    |---|---|---|
    | sequence → one label | classify a sensor episode | final or pooled states |
    | sequence → one value | forecast next-hour demand | final or pooled states |
    | sequence → label at each step | tag each time point | every output state |
    | sequence → sequence | translation or generation | decoder outputs over time |

    The architecture is only half the design. You must also define exactly which past
    information is legally available when each prediction is made.
    """),

    code(r"""
    import copy
    import math
    import random

    import matplotlib.pyplot as plt
    import numpy as np
    import pandas as pd
    import torch
    from sklearn.metrics import accuracy_score, log_loss
    from torch import nn
    from torch.utils.data import DataLoader, TensorDataset

    np.set_printoptions(precision=5, suppress=True)
    DEVICE = torch.device("cpu")


    def set_reproducible(seed):
        random.seed(seed)
        np.random.seed(seed)
        torch.manual_seed(seed)
        torch.use_deterministic_algorithms(True)


    def stable_sigmoid(values):
        values = np.asarray(values, dtype=float)
        result = np.empty_like(values)
        positive = values >= 0
        result[positive] = 1.0 / (1.0 + np.exp(-values[positive]))
        negative_exp = np.exp(values[~positive])
        result[~positive] = negative_exp / (1.0 + negative_exp)
        return result


    set_reproducible(7)
    print("PyTorch device:", DEVICE)
    """),

    md(r"""
    ## 3 · The tensor and symbol map

    We use **batch-first** layout throughout:

    $$
    X\in\mathbb{R}^{B\times T\times D}
    $$

    | Symbol | Plain meaning | Example |
    |---|---|---|
    | $B$ | sequences in one batch | 32 patients |
    | $T$ | time steps per padded sequence | 48 hourly readings |
    | $D$ | features at one step | temperature, pressure, vibration |
    | $H$ | numbers in the hidden state | 64 learned memory features |
    | $x_t$ | all input features at step $t$ | shape `(B, D)` |
    | $h_t$ | hidden state after step $t$ | shape `(B, H)` |

    For an RNN with one layer:

    ```text
    input X       (B, T, D)
      ↓ nn.RNN
    all outputs   (B, T, H)
    final h_n     (1, B, H)
    ```

    The leading `1` in `h_n` is the number of recurrent layers. With two layers it is
    `2`; with a bidirectional two-layer RNN it is `4` because each layer has a forward
    and backward direction.
    """),

    md(r"""
    ## 4 · One RNN update, first by hand

    Using row vectors, a vanilla RNN calculates:

    $$
    a_t=x_tW_x+h_{t-1}W_h+b
    $$

    $$
    h_t=\tanh(a_t)
    $$

    Shapes:

    | Quantity | Shape | Role |
    |---|---:|---|
    | $x_t$ | $1\times D$ | current input |
    | $W_x$ | $D\times H$ | input-to-hidden weights |
    | $h_{t-1}$ | $1\times H$ | previous memory |
    | $W_h$ | $H\times H$ | hidden-to-hidden weights |
    | $b$ | $1\times H$ | hidden bias |
    | $h_t$ | $1\times H$ | updated memory |

    Let $D=1$, $H=2$, $x_t=2$, and $h_{t-1}=[0.5,-0.5]$:

    $$
    W_x=[0.4,-0.2],\quad
    W_h=
    \begin{bmatrix}
    0.1&0.3\\
    -0.2&0.4
    \end{bmatrix},\quad
    b=[0.1,0]
    $$

    Input contribution:

    $$
    x_tW_x=2[0.4,-0.2]=[0.8,-0.4]
    $$

    Previous-state contribution:

    $$
    h_{t-1}W_h=[0.5,-0.5]W_h=[0.15,-0.05]
    $$

    Therefore:

    $$
    a_t=[0.8,-0.4]+[0.15,-0.05]+[0.1,0]=[1.05,-0.45]
    $$

    $$
    h_t=\tanh([1.05,-0.45])\approx[0.782,-0.422]
    $$

    The same $W_x$, $W_h$, and $b$ process every step. This **weight sharing across
    time** keeps the parameter count independent of sequence length.
    """),

    code(r"""
    current_input = np.array([[2.0]])
    previous_hidden = np.array([[0.5, -0.5]])
    input_weights = np.array([[0.4, -0.2]])
    recurrent_weights = np.array([[0.1, 0.3], [-0.2, 0.4]])
    hidden_bias = np.array([0.1, 0.0])

    input_contribution = current_input @ input_weights
    memory_contribution = previous_hidden @ recurrent_weights
    preactivation = input_contribution + memory_contribution + hidden_bias
    updated_hidden = np.tanh(preactivation)

    print("input contribution: ", input_contribution)
    print("memory contribution:", memory_contribution)
    print("preactivation:       ", preactivation)
    print("updated hidden state:", updated_hidden)

    assert np.allclose(updated_hidden, [[0.78181, -0.42190]], atol=1e-5)
    """),

    md(r"""
    ## 5 · Unrolling and backpropagation through time

    “Unrolling” does not copy the parameters. It draws the repeated computation so the
    dependency becomes visible:

    ```mermaid
    flowchart LR
        H0[h0 = zeros] --> C1[RNN cell]
        X1[x1] --> C1
        C1 --> H1[h1]
        H1 --> C2[RNN cell: same weights]
        X2[x2] --> C2
        C2 --> H2[h2]
        H2 --> C3[RNN cell: same weights]
        X3[x3] --> C3
        C3 --> H3[h3]
        H3 --> L[loss]
    ```

    **Backpropagation through time (BPTT)** is ordinary backpropagation applied to this
    unrolled graph. For $h_t=\tanh(a_t)$:

    $$
    \frac{\partial h_t}{\partial h_{t-1}}
    =W_h^\top\operatorname{diag}(1-h_t^2)
    $$

    A gradient travelling from step $T$ to step $k$ multiplies a chain of these
    Jacobian matrices:

    $$
    \frac{\partial L}{\partial h_k}
    =\frac{\partial L}{\partial h_T}
    \prod_{t=k+1}^{T}
    \frac{\partial h_t}{\partial h_{t-1}}
    $$

    A **Jacobian** is a table of local derivatives. If its repeated effect has size
    below 1, the signal shrinks exponentially. Above 1, it may grow explosively.
    Tanh saturation makes shrinking especially common because $1-h_t^2$ approaches
    zero when $h_t$ approaches $-1$ or $1$.

    Gradient clipping limits exploding updates. It does not restore a vanished signal.
    """),

    code(r"""
    # A scalar recurrence makes the repeated-product idea visible without hiding it
    # inside a large matrix. Each line is an exact chain-rule calculation.
    def scalar_gradient_path(recurrent_weight, initial_state=0.2, steps=30):
        hidden_values = [initial_state]
        local_derivatives = []
        for _ in range(steps):
            next_hidden = np.tanh(recurrent_weight * hidden_values[-1])
            hidden_values.append(next_hidden)
            local_derivatives.append(
                recurrent_weight * (1.0 - next_hidden**2)
            )

        gradient_to_state = 1.0  # choose L = h_T, so dL/dh_T = 1
        backward_norms = [gradient_to_state]
        for local_derivative in reversed(local_derivatives):
            gradient_to_state *= local_derivative
            backward_norms.append(abs(gradient_to_state))
        return np.array(hidden_values), np.array(backward_norms[::-1])


    _, shrinking_gradient = scalar_gradient_path(0.6)
    _, larger_weight_gradient = scalar_gradient_path(1.4, initial_state=0.01)

    print("|dL/dh_0| with recurrent weight 0.6:", shrinking_gradient[0])
    print("|dL/dh_0| with recurrent weight 1.4:", larger_weight_gradient[0])

    fig, axis = plt.subplots(figsize=(8, 4))
    axis.semilogy(range(31), shrinking_gradient, marker="o", label="weight 0.6")
    axis.semilogy(range(31), larger_weight_gradient, marker="s", label="weight 1.4")
    axis.set_xlabel("state index (0 is the earliest state)")
    axis.set_ylabel("absolute gradient on a log scale")
    axis.set_title("Measured chain-rule gradients through one scalar RNN")
    axis.legend()
    axis.grid(alpha=0.3)
    plt.show()
    """),

    md(r"""
    The chart is measured from the recurrence; it is not a decorative curve. A weight
    above 1 does not guarantee explosion because tanh may saturate and pull the local
    derivative back down. The actual behavior depends on both weights and states.
    """),

    md(r"""
    ## 6 · A vanilla RNN and BPTT in NumPy

    We use a sequence-to-one binary classifier. The loss is binary cross-entropy on the
    final logit. The backward pass accumulates parameter gradients because the same
    parameters were used at every step.
    """),

    code(r"""
    class ScratchRNNBinaryClassifier:
        def __init__(self, input_size, hidden_size, seed=0):
            generator = np.random.default_rng(seed)
            self.input_weights = generator.normal(0, 0.2, (input_size, hidden_size))
            self.recurrent_weights = generator.normal(0, 0.2, (hidden_size, hidden_size))
            self.hidden_bias = np.zeros(hidden_size)
            self.output_weights = generator.normal(0, 0.2, (hidden_size, 1))
            self.output_bias = np.zeros(1)

        def forward(self, sequence_batch):
            batch_size, time_steps, _ = sequence_batch.shape
            hidden_states = [np.zeros((batch_size, self.recurrent_weights.shape[0]))]
            for step in range(time_steps):
                preactivation = (
                    sequence_batch[:, step, :] @ self.input_weights
                    + hidden_states[-1] @ self.recurrent_weights
                    + self.hidden_bias
                )
                hidden_states.append(np.tanh(preactivation))

            logits = hidden_states[-1] @ self.output_weights + self.output_bias
            self.cache = (sequence_batch, hidden_states, logits)
            return logits

        def loss_and_backward(self, labels):
            sequence_batch, hidden_states, logits = self.cache
            batch_size, time_steps, _ = sequence_batch.shape
            probabilities = stable_sigmoid(logits)
            loss = -np.mean(
                labels * np.log(probabilities + 1e-12)
                + (1 - labels) * np.log(1 - probabilities + 1e-12)
            )

            # For sigmoid followed by binary cross-entropy, dL/dlogit = p - y.
            logit_gradient = (probabilities - labels) / batch_size
            output_weight_gradient = hidden_states[-1].T @ logit_gradient
            output_bias_gradient = logit_gradient.sum(axis=0)
            hidden_gradient = logit_gradient @ self.output_weights.T

            input_weight_gradient = np.zeros_like(self.input_weights)
            recurrent_weight_gradient = np.zeros_like(self.recurrent_weights)
            hidden_bias_gradient = np.zeros_like(self.hidden_bias)

            for step in reversed(range(time_steps)):
                tanh_gradient = hidden_gradient * (1 - hidden_states[step + 1] ** 2)
                input_weight_gradient += sequence_batch[:, step, :].T @ tanh_gradient
                recurrent_weight_gradient += hidden_states[step].T @ tanh_gradient
                hidden_bias_gradient += tanh_gradient.sum(axis=0)
                hidden_gradient = tanh_gradient @ self.recurrent_weights.T

            self.gradients = {
                "input_weights": input_weight_gradient,
                "recurrent_weights": recurrent_weight_gradient,
                "hidden_bias": hidden_bias_gradient,
                "output_weights": output_weight_gradient,
                "output_bias": output_bias_gradient,
            }
            return float(loss)


    tiny_sequences = np.array(
        [[[1.0], [0.2], [-0.1]], [[-1.0], [0.1], [0.3]]]
    )
    tiny_labels = np.array([[1.0], [0.0]])
    scratch_rnn = ScratchRNNBinaryClassifier(1, 3, seed=4)
    tiny_logits = scratch_rnn.forward(tiny_sequences)
    tiny_loss = scratch_rnn.loss_and_backward(tiny_labels)

    print("logits shape:", tiny_logits.shape)
    print("loss:", tiny_loss)
    for parameter_name, gradient in scratch_rnn.gradients.items():
        print(f"{parameter_name:20s} gradient shape {gradient.shape}")
    """),

    code(r"""
    # Finite differences ask: if one parameter moves by a tiny amount, does the loss
    # change by the amount predicted by our hand-written backward pass?
    def scratch_loss(model, sequences, labels):
        logits = model.forward(sequences)
        probabilities = stable_sigmoid(logits)
        return float(-np.mean(
            labels * np.log(probabilities + 1e-12)
            + (1 - labels) * np.log(1 - probabilities + 1e-12)
        ))


    epsilon = 1e-5
    row, column = 0, 1
    original_value = scratch_rnn.recurrent_weights[row, column]
    scratch_rnn.recurrent_weights[row, column] = original_value + epsilon
    loss_plus = scratch_loss(scratch_rnn, tiny_sequences, tiny_labels)
    scratch_rnn.recurrent_weights[row, column] = original_value - epsilon
    loss_minus = scratch_loss(scratch_rnn, tiny_sequences, tiny_labels)
    scratch_rnn.recurrent_weights[row, column] = original_value

    numerical_gradient = (loss_plus - loss_minus) / (2 * epsilon)
    scratch_rnn.forward(tiny_sequences)
    scratch_rnn.loss_and_backward(tiny_labels)
    analytical_gradient = scratch_rnn.gradients["recurrent_weights"][row, column]

    print("finite-difference gradient:", numerical_gradient)
    print("BPTT gradient:             ", analytical_gradient)
    print("absolute difference:       ", abs(numerical_gradient - analytical_gradient))

    assert abs(numerical_gradient - analytical_gradient) < 1e-7
    """),

    md(r"""
    ## 7 · LSTM: a memory path with learned gates

    An LSTM keeps two states:

    - $c_t$: the **cell state**, a longer-lived memory path;
    - $h_t$: the **hidden state**, the exposed summary used by the next layer or head.

    First concatenate the current input and previous hidden state:

    $$
    z_t=[x_t,h_{t-1}]
    $$

    Then calculate four vectors:

    $$
    f_t=\sigma(z_tW_f+b_f) \qquad \text{forget gate}
    $$

    $$
    i_t=\sigma(z_tW_i+b_i) \qquad \text{input gate}
    $$

    $$
    \widetilde{c}_t=\tanh(z_tW_c+b_c) \qquad \text{candidate content}
    $$

    $$
    o_t=\sigma(z_tW_o+b_o) \qquad \text{output gate}
    $$

    Update and expose memory:

    $$
    c_t=f_t\odot c_{t-1}+i_t\odot\widetilde{c}_t
    $$

    $$
    h_t=o_t\odot\tanh(c_t)
    $$

    Here $\sigma$ maps values to $(0,1)$, and $\odot$ means element-by-element
    multiplication. A gate near 0 mostly closes a path; a gate near 1 mostly opens it.

    ```mermaid
    flowchart LR
        CP[old cell state] -->|multiply by forget gate| KEEP[kept memory]
        X[current input and hidden state] --> CAND[candidate memory]
        X --> IN[input gate]
        CAND -->|multiply| WRITE[new writing]
        IN --> WRITE
        KEEP --> ADD[add]
        WRITE --> ADD
        ADD --> CT[new cell state]
        CT --> TANH[tanh]
        X --> OUT[output gate]
        TANH -->|multiply| HT[new hidden state]
        OUT --> HT
    ```
    """),

    md(r"""
    ### One LSTM step with actual numbers

    Use one cell dimension with old memory $c_{t-1}=0.40$. Suppose the gate
    preactivations produce:

    $$
    f_t=\sigma(2.0)=0.881,\quad
    i_t=\sigma(-0.5)=0.378
    $$

    $$
    \widetilde c_t=\tanh(1.0)=0.762,\quad
    o_t=\sigma(0.7)=0.668
    $$

    The new cell state is:

    $$
    c_t=(0.881)(0.40)+(0.378)(0.762)=0.640
    $$

    The exposed hidden state is:

    $$
    h_t=(0.668)\tanh(0.640)=0.377
    $$

    The old memory was mostly retained, and some candidate content was added.

    Along the direct cell-state path:

    $$
    \frac{\partial c_t}{\partial c_{t-1}}=f_t
    $$

    Across several steps that direct contribution contains a product of forget gates.
    If they remain near 1, gradients can survive longer than in a vanilla RNN. This is
    an **easier gradient path**, not a guarantee of perfect memory: gates below 1 still
    decay, other derivative paths interact, capacity is finite, and training can fail.
    """),

    code(r"""
    forget_gate = stable_sigmoid(np.array([2.0]))[0]
    input_gate = stable_sigmoid(np.array([-0.5]))[0]
    candidate = np.tanh(1.0)
    output_gate = stable_sigmoid(np.array([0.7]))[0]
    old_cell = 0.40

    new_cell = forget_gate * old_cell + input_gate * candidate
    new_hidden = output_gate * np.tanh(new_cell)

    print("forget gate:", forget_gate)
    print("input gate: ", input_gate)
    print("candidate:  ", candidate)
    print("output gate:", output_gate)
    print("new cell:   ", new_cell)
    print("new hidden: ", new_hidden)
    """),

    code(r"""
    class ScratchLSTMCell:
        def __init__(self, input_size, hidden_size, seed=0):
            generator = np.random.default_rng(seed)
            combined_size = input_size + hidden_size
            self.weights = generator.normal(0, 0.2, (combined_size, 4 * hidden_size))
            self.bias = np.zeros(4 * hidden_size)
            # A positive forget bias is a starting preference, not a learned guarantee.
            self.bias[:hidden_size] = 1.0
            self.hidden_size = hidden_size

        def step(self, current_input, previous_hidden, previous_cell):
            combined = np.concatenate([current_input, previous_hidden], axis=1)
            all_preactivations = combined @ self.weights + self.bias
            forget_pre, input_pre, candidate_pre, output_pre = np.split(
                all_preactivations, 4, axis=1
            )
            forget_gate = stable_sigmoid(forget_pre)
            input_gate = stable_sigmoid(input_pre)
            candidate = np.tanh(candidate_pre)
            output_gate = stable_sigmoid(output_pre)
            new_cell = forget_gate * previous_cell + input_gate * candidate
            new_hidden = output_gate * np.tanh(new_cell)
            return new_hidden, new_cell, {
                "forget": forget_gate,
                "input": input_gate,
                "candidate": candidate,
                "output": output_gate,
            }


    cell = ScratchLSTMCell(input_size=2, hidden_size=3, seed=5)
    sample_input = np.array([[0.2, -0.4], [0.7, 0.1]])
    previous_hidden = np.zeros((2, 3))
    previous_cell = np.zeros((2, 3))
    next_hidden, next_cell, gates = cell.step(
        sample_input, previous_hidden, previous_cell
    )

    print("input shape:      ", sample_input.shape)
    print("next hidden shape:", next_hidden.shape)
    print("next cell shape:  ", next_cell.shape)
    for gate_name, gate_values in gates.items():
        print(f"{gate_name:10s} mean {gate_values.mean():.3f}")
    """),

    md(r"""
    ## 8 · RNN, GRU, and LSTM in PyTorch

    PyTorch performs BPTT automatically, but shape reasoning is still your job.
    """),

    code(r"""
    batch_size, time_steps, input_size, hidden_size = 4, 7, 3, 5
    example_batch = torch.randn(batch_size, time_steps, input_size)

    rnn_layer = nn.RNN(input_size, hidden_size, batch_first=True)
    gru_layer = nn.GRU(input_size, hidden_size, batch_first=True)
    lstm_layer = nn.LSTM(input_size, hidden_size, batch_first=True)

    rnn_outputs, rnn_final_hidden = rnn_layer(example_batch)
    gru_outputs, gru_final_hidden = gru_layer(example_batch)
    lstm_outputs, (lstm_final_hidden, lstm_final_cell) = lstm_layer(example_batch)

    print("input:            ", tuple(example_batch.shape))
    print("RNN outputs / h_n:", tuple(rnn_outputs.shape), tuple(rnn_final_hidden.shape))
    print("GRU outputs / h_n:", tuple(gru_outputs.shape), tuple(gru_final_hidden.shape))
    print("LSTM outputs:     ", tuple(lstm_outputs.shape))
    print("LSTM h_n / c_n:   ", tuple(lstm_final_hidden.shape), tuple(lstm_final_cell.shape))

    assert torch.allclose(lstm_outputs[:, -1, :], lstm_final_hidden[0])
    """),

    md(r"""
    `outputs[:, -1, :]` equals the final hidden state here because every sequence has
    the same length, there is one forward layer, and no padding follows the real data.
    Do not use that shortcut blindly with padded batches or bidirectional layers.

    ### Parameter counts

    A vanilla RNN has one candidate-state calculation. A GRU has three related blocks;
    an LSTM has four. Exact library layouts include biases, but the rough lesson is:
    more gates add parameters and compute in exchange for more control over memory.
    """),

    code(r"""
    def trainable_parameter_count(model):
        return sum(parameter.numel() for parameter in model.parameters() if parameter.requires_grad)


    parameter_table = pd.DataFrame(
        {
            "layer": ["RNN", "GRU", "LSTM"],
            "trainable parameters": [
                trainable_parameter_count(rnn_layer),
                trainable_parameter_count(gru_layer),
                trainable_parameter_count(lstm_layer),
            ],
        }
    )
    display(parameter_table)
    """),

    md(r"""
    ## 9 · A fair delayed-memory experiment

    We build a small diagnostic task: the label is the sign of a value marked at the
    first step, while later steps contain distracting noise. The second feature is a
    marker: `1` means “store this value” and `0` means “this is a distractor.” The task
    isolates delayed credit assignment. It is not proof that one architecture wins in
    general.

    The split contract is strict:

    1. training data updates weights;
    2. validation loss chooses the checkpoint;
    3. model configurations are frozen;
    4. test data is opened once for the final report.

    We use `BCEWithLogitsLoss`, which combines sigmoid and binary cross-entropy in a
    numerically stable calculation. The model returns raw logits during training.
    """),

    code(r"""
    def make_delayed_memory_data(number_of_sequences, time_steps, seed):
        generator = np.random.default_rng(seed)
        labels = generator.integers(0, 2, size=number_of_sequences).astype(np.float32)
        sequences = np.zeros((number_of_sequences, time_steps, 2), dtype=np.float32)
        sequences[:, :, 0] = generator.normal(
            0, 0.25, (number_of_sequences, time_steps)
        )
        sequences[:, 0, 0] = np.where(labels == 1, 1.0, -1.0)
        sequences[:, 0, 1] = 1.0
        return sequences, labels[:, None]


    X_train, y_train = make_delayed_memory_data(900, 35, seed=10)
    X_validation, y_validation = make_delayed_memory_data(300, 35, seed=11)
    X_test, y_test = make_delayed_memory_data(300, 35, seed=12)


    class RecurrentBinaryClassifier(nn.Module):
        def __init__(self, cell_type, hidden_size=24):
            super().__init__()
            cell_classes = {"rnn": nn.RNN, "gru": nn.GRU, "lstm": nn.LSTM}
            self.cell_type = cell_type
            self.recurrent = cell_classes[cell_type](2, hidden_size, batch_first=True)
            self.output = nn.Linear(hidden_size, 1)

            if cell_type == "lstm":
                # PyTorch gate order is input, forget, cell candidate, output. A
                # positive forget bias starts the model willing to retain information.
                with torch.no_grad():
                    self.recurrent.bias_ih_l0[hidden_size:2 * hidden_size].fill_(1.0)
                    self.recurrent.bias_hh_l0[hidden_size:2 * hidden_size].zero_()

        def forward(self, sequence_batch):
            all_hidden_states, _ = self.recurrent(sequence_batch)
            return self.output(all_hidden_states[:, -1, :])


    def binary_metrics(model, features, labels):
        model.eval()
        with torch.no_grad():
            logits = model(torch.from_numpy(features)).squeeze(1)
            probabilities = torch.sigmoid(logits).numpy()
        predictions = (probabilities >= 0.5).astype(int)
        return {
            "accuracy": accuracy_score(labels.ravel(), predictions),
            "log_loss": log_loss(labels.ravel(), probabilities, labels=[0, 1]),
        }


    def fit_recurrent_model(cell_type, seed=21, max_epochs=80, patience=10):
        set_reproducible(seed)
        model = RecurrentBinaryClassifier(cell_type).to(DEVICE)
        optimizer = torch.optim.AdamW(model.parameters(), lr=0.006, weight_decay=1e-4)
        loss_function = nn.BCEWithLogitsLoss()
        loader = DataLoader(
            TensorDataset(torch.from_numpy(X_train), torch.from_numpy(y_train)),
            batch_size=64,
            shuffle=True,
            generator=torch.Generator().manual_seed(seed),
        )

        best_validation_loss = math.inf
        best_state = None
        best_epoch = 0
        stale_epochs = 0
        history = []

        for epoch in range(1, max_epochs + 1):
            model.train()
            running_loss = 0.0
            for sequence_batch, label_batch in loader:
                optimizer.zero_grad(set_to_none=True)
                logits = model(sequence_batch)
                loss = loss_function(logits, label_batch)
                loss.backward()
                gradient_norm = nn.utils.clip_grad_norm_(model.parameters(), max_norm=5.0)
                if not torch.isfinite(gradient_norm):
                    raise RuntimeError("A recurrent gradient became non-finite.")
                optimizer.step()
                running_loss += loss.item() * len(sequence_batch)

            validation = binary_metrics(model, X_validation, y_validation)
            history.append(
                {
                    "epoch": epoch,
                    "train_loss": running_loss / len(X_train),
                    "validation_log_loss": validation["log_loss"],
                    "validation_accuracy": validation["accuracy"],
                }
            )
            if validation["log_loss"] < best_validation_loss - 1e-4:
                best_validation_loss = validation["log_loss"]
                best_epoch = epoch
                best_state = copy.deepcopy(model.state_dict())
                stale_epochs = 0
            else:
                stale_epochs += 1
                if stale_epochs >= patience:
                    break

        model.load_state_dict(best_state)
        return model, history, best_epoch, binary_metrics(model, X_validation, y_validation)


    development_results = {}
    frozen_models = {}
    for cell_type in ("rnn", "lstm"):
        model, history, best_epoch, validation_metrics = fit_recurrent_model(cell_type)
        frozen_models[cell_type] = model
        development_results[cell_type] = {
            "history": history,
            "best_epoch": best_epoch,
            "validation": validation_metrics,
        }
        print(cell_type.upper(), "best epoch", best_epoch, "validation", validation_metrics)
    """),

    code(r"""
    # Every development decision is now frozen. Only this cell opens the test split.
    final_results = {
        cell_type: binary_metrics(model, X_test, y_test)
        for cell_type, model in frozen_models.items()
    }
    final_table = pd.DataFrame(final_results).T
    display(final_table)

    fig, axes = plt.subplots(1, 2, figsize=(11, 4))
    for cell_type, color in (("rnn", "tab:orange"), ("lstm", "tab:blue")):
        history = development_results[cell_type]["history"]
        axes[0].plot(
            [row["epoch"] for row in history],
            [row["train_loss"] for row in history],
            label=cell_type.upper(),
            color=color,
        )
        axes[1].plot(
            [row["epoch"] for row in history],
            [row["validation_log_loss"] for row in history],
            label=cell_type.upper(),
            color=color,
        )
    axes[0].set_title("Training loss")
    axes[1].set_title("Validation log loss")
    for axis in axes:
        axis.set_xlabel("epoch")
        axis.legend()
        axis.grid(alpha=0.3)
    plt.show()
    """),

    md(r"""
    Read the evidence carefully:

    - Test accuracy says how often the frozen decision is correct at threshold 0.5.
    - Log loss also cares about probability confidence.
    - One seed and one synthetic task cannot establish universal superiority.
    - The task can be solved by directly retaining the first value; a feature-engineered
      baseline that exposes that value would be trivial. This experiment is an
      optimization microscope, not a business benchmark.

    A stronger study would repeat fixed configurations across several training seeds
    and report variation before opening test data.
    """),

    md(r"""
    ## 10 · Variable lengths: padding is not data

    Batches want rectangular tensors, but real sequences have different lengths. A
    common workflow is:

    1. pad shorter sequences to the longest sequence in the batch;
    2. preserve each real length;
    3. prevent padded steps from affecting the selected representation or loss.

    If lengths are `[3, 5]`, then `outputs[:, -1, :]` is wrong for the first sequence:
    index `-1` is padding, not its third real step.

    PyTorch options:

    - `pack_padded_sequence` skips padded recurrent work;
    - gather `outputs[batch_index, length - 1]` for the last real forward state;
    - use a Boolean mask when computing a per-step loss;
    - pass `padding_idx` to an embedding so its padding vector is not trained.
    """),

    code(r"""
    from torch.nn.utils.rnn import pack_padded_sequence, pad_packed_sequence

    padded_batch = torch.tensor(
        [
            [[1.0], [2.0], [3.0], [0.0], [0.0]],
            [[4.0], [5.0], [6.0], [7.0], [8.0]],
        ]
    )
    real_lengths = torch.tensor([3, 5])
    packing_rnn = nn.GRU(input_size=1, hidden_size=4, batch_first=True)

    packed = pack_padded_sequence(
        padded_batch, real_lengths.cpu(), batch_first=True, enforce_sorted=False
    )
    packed_outputs, final_hidden = packing_rnn(packed)
    unpacked_outputs, _ = pad_packed_sequence(packed_outputs, batch_first=True)

    batch_indices = torch.arange(len(real_lengths))
    last_real_outputs = unpacked_outputs[batch_indices, real_lengths - 1]

    print("padded batch:     ", tuple(padded_batch.shape))
    print("unpacked outputs: ", tuple(unpacked_outputs.shape))
    print("last real outputs:", tuple(last_real_outputs.shape))
    print("matches packed h_n:", torch.allclose(last_real_outputs, final_hidden[0]))
    """),

    md(r"""
    ## 11 · Important variants and boundaries

    ### GRU

    A gated recurrent unit merges cell and hidden state and uses update and reset gates.
    It usually has fewer parameters than an LSTM. Choose between them with validation
    evidence; “LSTM is always better” and “GRU is always faster” are not reliable rules.

    ### Bidirectional recurrence

    A bidirectional model reads left-to-right and right-to-left, then combines both
    representations. It is useful when the entire sequence exists before prediction,
    such as tagging a completed document.

    Do **not** use the backward direction for causal forecasting or live generation.
    It would expose future information.

    ### Stacked recurrence

    The output sequence of one recurrent layer becomes the input to the next. More
    layers increase representation capacity and optimization difficulty. Recurrent
    dropout, residual paths, and normalization may help, but must be validated.

    ### Truncated BPTT

    For a very long stream, process a window, carry the hidden state forward, and call
    `state = state.detach()` before the next window. Detaching keeps the numerical state
    while ending the backward graph. The tradeoff is deliberate: dependencies longer
    than the truncation window receive no direct gradient.

    ### Teacher forcing and exposure bias

    In an autoregressive decoder, training may feed the correct previous answer while
    inference feeds the model's own previous prediction. The mismatch is called
    exposure bias. Scheduled sampling is one proposed response, but it introduces its
    own bias; sequence-level evaluation remains essential.
    """),

    md(r"""
    ## 12 · When to use recurrence—and when not to

    | Approach | Main strength | Main limitation | Good fit |
    |---|---|---|---|
    | lag features + tree model | strong, simple tabular baseline | manual horizon/window design | moderate structured time series |
    | temporal 1D CNN | parallel local-pattern learning | receptive field is architecture-dependent | signals with local motifs |
    | vanilla RNN | smallest recurrent baseline | weak long-range optimization | short dependencies, teaching |
    | GRU | gated memory with fewer blocks | still sequential | streaming and modest data |
    | LSTM | explicit gated cell-state path | more parameters; still sequential | medium-range stateful tasks |
    | Transformer | direct long-range access and parallel training | attention cost; more data/compute | large-scale and long-context tasks |

    Use an RNN, GRU, or LSTM when order matters and carrying compact state is valuable,
    especially for streaming or modest data. Do not reach for recurrence automatically
    when a feature-engineered baseline is sufficient, when the task has no meaningful
    order, or when parallel long-context processing is central.

    A model choice should follow a baseline and a deployment constraint, not fashion.
    """),

    md(r"""
    ## 13 · Failure modes you should be able to diagnose

    | Symptom | Likely cause | What to inspect | Response |
    |---|---|---|---|
    | early inputs seem ignored | vanishing temporal gradients | gradient norm by time step | LSTM/GRU, shorter path, attention |
    | loss spikes or becomes NaN | exploding gradients or bad scale | raw gradients, inputs, learning rate | clip, normalize, lower rate |
    | validation is impossibly good | future or entity leakage | split timestamps and IDs | rebuild point-in-time split |
    | padded examples behave oddly | last padded state used | lengths, masks, gathered index | pack or gather last real step |
    | streaming predictions contaminate users | state shared across entities | state-store keys and reset logic | isolate and expire state |
    | training memory grows each window | state graph was not detached | `grad_fn` across chunks | detach at truncation boundary |
    | bidirectional forecast wins suspiciously | future steps are visible | direction and prediction timestamp | use forward-only causal model |
    | train good, validation poor | overfit or length/regime shift | curves and length distribution | regularize, improve split/data |

    ### State is part of the serving contract

    A streaming recurrent service must define:

    - which entity owns each hidden state;
    - when a session starts and ends;
    - how state is initialized and expired;
    - what happens after missing or out-of-order events;
    - whether replay produces the same result;
    - how model-version changes invalidate old state.

    State leakage between users is both a correctness and privacy failure.
    """),

    md(r"""
    ## 14 · Real scenario: hourly electricity demand

    A utility forecasts demand using load history, weather, calendar features, and
    planned outages.

    **Why recurrence may help:** the model can update a compact state as every hour
    arrives, which is natural for low-latency streaming.

    **Why it may not be the best first model:** gradient-boosted trees on lagged and
    rolling features are strong, inspectable baselines. A temporal CNN trains in
    parallel. A Transformer may use longer context more directly.

    **Honest evaluation:** split in time order. Every weather forecast and feature must
    reflect what was known at the prediction timestamp. Randomly shuffling hours lets
    future regimes leak backward.

    **Metrics:** MAE by forecast horizon, peak-hour error, interval coverage, latency,
    and cost under asymmetric over- versus under-forecast decisions.

    **Model decision:** choose the simplest approach that meets the operational target
    across several forward-validation windows. A lower average error does not compensate
    for leakage, missed peaks, or an impossible serving-state design.
    """),

    md(r"""
    ## 15 · Check your understanding

    Answer without looking back:

    1. In `(B, T, D)`, what does each letter mean?
    2. Why does an RNN's parameter count not grow with sequence length?
    3. What exactly is multiplied repeatedly during BPTT?
    4. Why can gradient clipping address explosion but not vanishing?
    5. What is the difference between $c_t$ and $h_t$ in an LSTM?
    6. What does each LSTM gate control?
    7. Why is “LSTMs solve vanishing gradients” too strong?
    8. Why can `outputs[:, -1, :]` be wrong for padded sequences?
    9. When does a bidirectional RNN leak future information?
    10. Why must streaming hidden state be keyed and reset per entity?
    """),

    md(r"""
    ## 16 · Practice and mini-project

    ### Beginner

    1. Repeat the hand calculation in Section 4 with $x_t=-1$.
    2. Draw a length-four RNN and label every $x_t$ and $h_t$.

    ### Intermediate

    3. Extend the finite-difference check to every scratch-RNN parameter and report the
       maximum absolute and relative error.
    4. Add a GRU to the delayed-memory experiment. Keep the split, seed, hidden size,
       optimizer, selection metric, and patience fixed.

    ### Challenge

    5. Repeat the delayed-memory experiment at lengths 10, 35, and 80 across three
       training seeds. Choose no architecture using test results. Plot validation loss,
       test only the frozen representative, and state what the evidence does not prove.

    ### Mini-project · Stateful equipment-warning model

    **Goal:** predict whether a machine will enter an alarm state in the next hour.

    **Columns:** `machine_id`, `timestamp`, `temperature`, `pressure`, `vibration`,
    `load`, `maintenance_flag`, and `alarm_next_hour`.

    **Workflow:**

    1. sort within machine and audit timestamp gaps;
    2. split by time before learning transformations;
    3. fit scaling on training data only;
    4. create variable-length windows plus lengths and masks;
    5. compare a majority baseline, lag-feature tree model, GRU, and LSTM;
    6. select with forward validation across at least three seeds;
    7. open test once and report recall, precision, PR AUC, latency, and false alarms;
    8. describe state keys, reset rules, late events, and model-version migration.

    **Expected output:** a reproducible report, learning curves, threshold rationale,
    frozen checkpoint, preprocessing artifact, and state-serving contract.

    **Evaluation:** no temporal/entity leakage; padding handled correctly; baselines are
    fair; selection is validation-only; claims match multi-seed evidence.
    """),

    md(r"""
    ## 17 · Summary and memory aid

    An RNN applies one shared update across time and carries a hidden state. Unrolling
    reveals a deep computation graph, so BPTT repeatedly multiplies local derivatives;
    signals can vanish or explode. An LSTM adds a gated, additive cell-state path that
    can preserve information and gradients longer, but it does not guarantee unlimited
    memory and it remains sequential.

    Production mastery also requires honest time-aware splits, correct padding, stable
    logits and clipping, multi-seed evidence, and safe per-entity state management.

    **Memory aid:** *An RNN rewrites a note each step; an LSTM learns what to erase,
    write, and reveal.*

    ### Why attention is next

    RNNs compress the past into a fixed-size state and process steps serially.
    Attention lets a position retrieve information directly from other positions. DL-07
    will build that operation from weighted averages, then explain the new costs and
    masking rules it introduces.
    """),
]


build("04_deep_learning/06_rnn_and_lstm.ipynb", cells)
