"""Build DL-03: backpropagation and a scalar reverse-mode autodiff engine."""

import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
from nbbuild import build, code, md  # noqa: E402


cells = [
    md(r"""
    # DL-03 · Backpropagation and Autodiff

    **Prerequisites:** FND-04, DL-01, and DL-02  
    **Estimated study time:** 10–12 hours, including practice  
    **Next lesson:** DL-04 · Stable Neural Training

    DL-02 calculated every gradient for one fixed network. That was useful, but adding
    a layer would force us to derive and code another special backward pass.

    This lesson builds the reusable idea underneath `loss.backward()`. We will first
    trace one tiny graph with ordinary numbers. Then we will teach each operation how
    to send a gradient backward, arrange those operations in a safe order, and build a
    small reverse-mode automatic-differentiation engine.

    ### Scope boundary

    We focus on **how gradients are calculated**, not on choosing an optimizer or
    making a deep model train reliably. DL-04 handles initialization experiments,
    optimizers, normalization, regularization, clipping, and stable training.
    """),

    md(r"""
    ## 1 · What you will be able to do

    By the end, you will be able to:

    - separate a forward pass, backpropagation, autodiff, and gradient descent;
    - calculate a small computation graph forward and backward by hand;
    - distinguish an upstream gradient from a local derivative;
    - explain why branches must add their gradient contributions;
    - place graph nodes in topological and reverse-topological order;
    - explain why reverse mode fits one scalar loss and many parameters;
    - build scalar operations that record their own backward rules;
    - implement a reusable reverse-mode autodiff engine;
    - verify every selected leaf gradient with central finite differences;
    - derive matrix-multiplication and broadcast-bias gradients by shape;
    - reproduce scratch gradients with PyTorch autograd;
    - use `requires_grad`, `.grad`, `retain_grad()`, `detach()`, and `no_grad()` safely;
    - identify when gradients vanish, explode, accumulate, or stop;
    - decide when backpropagation is unsuitable or needs another estimator.

    ### Learning path

    ```mermaid
    flowchart LR
        A[One numeric graph] --> B[Local chain rules]
        B --> C[Branches add gradients]
        C --> D[Reverse graph order]
        D --> E[Scalar autodiff engine]
        E --> F[Finite-difference check]
        F --> G[Tensor and broadcast rules]
        G --> H[PyTorch equivalence]
        H --> I[Autodiff mini-project]
        I --> J[DL-04 stable training]
    ```

    DL-02 two-layer gradients  
    → required before general backpropagation  
    → because this lesson turns those repeated chain-rule steps into a reusable graph algorithm.

    Backpropagation  
    → required before stable neural training  
    → because training diagnostics only make sense after we know where gradients come from.
    """),

    md(r"""
    ## 2 · Four jobs that beginners often mix together

    Imagine adjusting a soup recipe.

    - The **forward pass** cooks the soup and measures its taste score.
    - **Backpropagation** traces how each ingredient affected that score.
    - **Automatic differentiation**, or **autodiff**, records the recipe steps and
      performs that tracing automatically.
    - **Gradient descent** changes the ingredient amounts using the calculated effects.

    | Job | Question it answers | Does it update parameters? |
    |---|---|---:|
    | forward pass | What output and loss do these values produce? | no |
    | backpropagation | How does the loss change with each earlier value? | no |
    | reverse-mode autodiff | How can software perform backpropagation for a recorded program? | no |
    | gradient descent | How should parameters move after gradients are known? | yes |
    | finite differences | Does a calculated gradient agree with small numerical changes? | no |

    Backpropagation is therefore **not** the optimizer. `loss.backward()` fills
    gradients; `optimizer.step()` uses them.
    """),

    code(r"""
    import math

    import matplotlib.pyplot as plt
    import numpy as np
    import pandas as pd
    import torch

    np.set_printoptions(precision=5, suppress=True)
    torch.manual_seed(42)
    """),

    md(r"""
    ## 3 · First calculate one graph with ordinary numbers

    We will use a one-neuron expression:

    $$
    z=wx+b
    $$

    $$
    p=\tanh(z)
    $$

    $$
    e=p-y
    $$

    $$
    L=e^2
    $$

    **Symbols:** $x$ is an input; $w$ and $b$ are adjustable parameters; $z$ is the
    neuron's raw score; $p$ is its output; $y$ is the target; $e$ is error; and $L$ is
    scalar loss.

    Use $w=0.5$, $x=2$, $b=-1$, and $y=1$.

    ### Forward pass

    $$
    z=(0.5)(2)-1=0
    $$

    $$
    p=\tanh(0)=0
    $$

    $$
    e=0-1=-1
    $$

    $$
    L=(-1)^2=1
    $$

    The forward pass answers, “What happened?” The backward pass asks, “Which earlier
    values were responsible, and by how much?”
    """),

    code(r"""
    weight_value = 0.5
    input_value = 2.0
    bias_value = -1.0
    target_value = 1.0

    # Follow the four equations in order and keep every intermediate value.
    raw_score = weight_value * input_value + bias_value
    prediction = math.tanh(raw_score)
    error = prediction - target_value
    loss = error ** 2

    forward_table = pd.DataFrame(
        [
            {"node": "z", "calculation": "w*x + b", "value": raw_score},
            {"node": "p", "calculation": "tanh(z)", "value": prediction},
            {"node": "e", "calculation": "p - y", "value": error},
            {"node": "L", "calculation": "e^2", "value": loss},
        ]
    )
    print(forward_table.to_string(index=False))

    assert np.isclose(loss, 1.0)
    """),

    md(r"""
    ## 4 · Walk backward: upstream gradient × local derivative

    Start at the loss. A value changes one-for-one with itself, so the seed is:

    $$
    \frac{\partial L}{\partial L}=1
    $$

    Each node receives an **upstream gradient**: how sensitive the final loss is to
    that node's output. It multiplies that number by its **local derivative**: how
    sensitive the node's output is to one input.

    ```mermaid
    flowchart LR
        W[w = 0.5] --> M["multiply"]
        X[x = 2] --> M
        M --> A["add b"]
        B[b = -1] --> A
        A --> T[tanh]
        T --> S["subtract y"]
        Y[y = 1] --> S
        S --> Q[square]
        Q --> L[L = 1]

        L -. "seed 1" .-> Q
        Q -. "× 2e" .-> S
        S -. "× 1" .-> T
        T -. "× (1-p²)" .-> A
        A -. "× 1" .-> M
        M -. "× x" .-> W
    ```

    At our values:

    $$
    \frac{\partial L}{\partial e}=2e=-2
    $$

    $$
    \frac{\partial L}{\partial p}=(-2)(1)=-2
    $$

    $$
    \frac{\partial L}{\partial z}=(-2)(1-p^2)=(-2)(1-0^2)=-2
    $$

    $$
    \frac{\partial L}{\partial w}=(-2)x=-4
    $$

    $$
    \frac{\partial L}{\partial x}=(-2)w=-1
    $$

    $$
    \frac{\partial L}{\partial b}=(-2)(1)=-2
    $$

    A negative weight gradient means a small increase in $w$ would locally decrease
    the loss. Backpropagation calculates that sensitivity; it does not perform the increase.
    """),

    code(r"""
    # Work backward from the scalar loss, reusing cached forward values.
    loss_to_error = 2 * error
    loss_to_prediction = loss_to_error * 1
    loss_to_raw_score = loss_to_prediction * (1 - prediction ** 2)
    loss_to_weight = loss_to_raw_score * input_value
    loss_to_input = loss_to_raw_score * weight_value
    loss_to_bias = loss_to_raw_score * 1

    backward_table = pd.DataFrame(
        [
            {"gradient": "dL/de", "value": loss_to_error},
            {"gradient": "dL/dp", "value": loss_to_prediction},
            {"gradient": "dL/dz", "value": loss_to_raw_score},
            {"gradient": "dL/dw", "value": loss_to_weight},
            {"gradient": "dL/dx", "value": loss_to_input},
            {"gradient": "dL/db", "value": loss_to_bias},
        ]
    )
    print(backward_table.to_string(index=False))

    assert np.allclose(
        [loss_to_weight, loss_to_input, loss_to_bias],
        [-4.0, -1.0, -2.0],
    )
    """),

    md(r"""
    ## 5 · A branch must add gradients, not overwrite them

    Consider:

    $$
    q=a^2+a
    $$

    The value $a$ reaches $q$ through two paths. The first contributes $2a$; the
    second contributes $1$.

    $$
    \frac{dq}{da}=2a+1
    $$

    At $a=3$, the gradient is $7$.

    Think of two roads delivering water to the same tank. The final amount is the sum
    arriving through both roads. In a graph, gradients from every outgoing path must
    accumulate at the shared node.

    ```mermaid
    flowchart LR
        A[a] --> S[square]
        A --> P[add]
        S --> P
        P --> Q[q]
    ```

    This is why autodiff engines use `+=` when updating an input's gradient.
    """),

    code(r"""
    branch_input = 3.0
    square_path_contribution = 2 * branch_input
    direct_path_contribution = 1.0
    accumulated_gradient = square_path_contribution + direct_path_contribution

    print("square path contribution:", square_path_contribution)
    print("direct path contribution:", direct_path_contribution)
    print("total dq/da:", accumulated_gradient)

    assert accumulated_gradient == 7.0
    """),

    md(r"""
    ## 6 · Graph order makes the backward pass safe

    A **topological order** lists every node after its dependencies. The forward pass
    follows that order because an operation needs its inputs first.

    The backward pass uses the reverse order. A node must receive all gradient
    contributions from later operations before it sends its completed gradient to
    earlier inputs.

    For the graph above:

    ```text
    forward:   leaves → multiply → add → tanh → subtract → square → loss
    backward:  loss → square → subtract → tanh → add → multiply → leaves
    ```

    A graph may branch or reuse values, so “reverse the lines of code” is not a general
    algorithm. We first discover the dependency order, then reverse that order.

    ### The reusable node contract

    Each scalar node will store:

    - its forward value;
    - its accumulated gradient;
    - the nodes that produced it;
    - a small function that knows its local backward rule;
    - a label and operation name for inspection.
    """),

    md(r"""
    ## 7 · Build a scalar reverse-mode autodiff engine

    The class below is intentionally small, but it contains the essential mechanism:

    1. arithmetic creates a result node;
    2. that node remembers its parents;
    3. it stores a local backward function;
    4. `backward()` finds topological order;
    5. the loss receives seed gradient `1`;
    6. local functions run in reverse order and accumulate into parents.

    ReLU is not differentiable exactly at zero. Like many frameworks, this engine
    chooses a derivative of zero there. That is a declared convention, not an ordinary
    derivative.
    """),

    code(r"""
    class Value:
        '''A scalar value and the graph needed for reverse-mode differentiation.'''

        def __init__(self, data, parents=(), operation="", label=""):
            self.data = float(data)
            self.grad = 0.0
            self.parents = tuple(parents)
            self.operation = operation
            self.label = label
            self._backward = lambda: None

        def __repr__(self):
            return f"Value(data={self.data:.5f}, grad={self.grad:.5f}, label={self.label!r})"

        def __add__(self, other):
            other = other if isinstance(other, Value) else Value(other)
            result = Value(self.data + other.data, (self, other), "+")

            def _backward():
                # Addition copies the same upstream gradient to both inputs.
                self.grad += result.grad
                other.grad += result.grad

            result._backward = _backward
            return result

        def __mul__(self, other):
            other = other if isinstance(other, Value) else Value(other)
            result = Value(self.data * other.data, (self, other), "*")

            def _backward():
                # Each input receives upstream gradient times the other forward value.
                self.grad += other.data * result.grad
                other.grad += self.data * result.grad

            result._backward = _backward
            return result

        def __pow__(self, exponent):
            if not isinstance(exponent, (int, float)):
                raise TypeError("This small engine supports only numeric powers.")
            result = Value(self.data ** exponent, (self,), f"**{exponent}")

            def _backward():
                self.grad += exponent * self.data ** (exponent - 1) * result.grad

            result._backward = _backward
            return result

        def tanh(self):
            output = math.tanh(self.data)
            result = Value(output, (self,), "tanh")

            def _backward():
                self.grad += (1 - output ** 2) * result.grad

            result._backward = _backward
            return result

        def exp(self):
            output = math.exp(self.data)
            result = Value(output, (self,), "exp")

            def _backward():
                self.grad += output * result.grad

            result._backward = _backward
            return result

        def log(self):
            if self.data <= 0:
                raise ValueError("log requires a positive value.")
            result = Value(math.log(self.data), (self,), "log")

            def _backward():
                self.grad += (1 / self.data) * result.grad

            result._backward = _backward
            return result

        def relu(self):
            output = max(0.0, self.data)
            result = Value(output, (self,), "ReLU")

            def _backward():
                self.grad += (1.0 if self.data > 0 else 0.0) * result.grad

            result._backward = _backward
            return result

        def sigmoid(self):
            # Choose the algebraic form that avoids a large positive exponential.
            if self.data >= 0:
                output = 1 / (1 + math.exp(-self.data))
            else:
                exponential = math.exp(self.data)
                output = exponential / (1 + exponential)
            result = Value(output, (self,), "sigmoid")

            def _backward():
                self.grad += output * (1 - output) * result.grad

            result._backward = _backward
            return result

        def topological_order(self):
            ordered_nodes = []
            visited_node_ids = set()

            def visit(node):
                if id(node) in visited_node_ids:
                    return
                visited_node_ids.add(id(node))
                for parent in node.parents:
                    visit(parent)
                ordered_nodes.append(node)

            visit(self)
            return ordered_nodes

        def zero_grad(self):
            for node in self.topological_order():
                node.grad = 0.0

        def backward(self, seed=1.0, reset=True):
            ordered_nodes = self.topological_order()
            if reset:
                for node in ordered_nodes:
                    node.grad = 0.0
            self.grad = float(seed)
            for node in reversed(ordered_nodes):
                node._backward()

        def __neg__(self):
            return self * -1

        def __sub__(self, other):
            return self + (-other)

        def __rsub__(self, other):
            return other + (-self)

        def __truediv__(self, other):
            other = other if isinstance(other, Value) else Value(other)
            return self * other ** -1

        def __rtruediv__(self, other):
            return other * self ** -1

        def __radd__(self, other):
            return self + other

        def __rmul__(self, other):
            return self * other
    """),

    md(r"""
    ## 8 · Rebuild the manual graph and inspect the result

    The expression below is the same graph calculated in Sections 3 and 4. This time,
    normal arithmetic constructs the graph automatically.

    Notice what the engine does **not** know:

    - It does not know what a neuron is.
    - It does not contain a special formula for this complete expression.
    - It only knows local rules for multiplication, addition, tanh, subtraction, and power.

    Complex derivatives emerge by composing simple local rules.
    """),

    code(r"""
    weight = Value(0.5, label="w")
    input_feature = Value(2.0, label="x")
    bias = Value(-1.0, label="b")
    target = Value(1.0, label="y")

    # Ordinary operators now build the same graph we previously calculated by hand.
    score = weight * input_feature + bias
    score.label = "z"
    model_output = score.tanh()
    model_output.label = "p"
    model_error = model_output - target
    model_error.label = "e"
    graph_loss = model_error ** 2
    graph_loss.label = "L"
    graph_loss.backward()

    leaf_report = pd.DataFrame(
        [
            {"leaf": node.label, "value": node.data, "gradient": node.grad}
            for node in [weight, input_feature, bias, target]
        ]
    )
    graph_order = [node.label or node.operation or "constant" for node in graph_loss.topological_order()]

    print("forward loss:", graph_loss.data)
    print("topological order:", " -> ".join(graph_order))
    print(leaf_report.to_string(index=False))

    assert np.allclose([weight.grad, input_feature.grad, bias.grad], [-4.0, -1.0, -2.0])
    """),

    md(r"""
    ## 9 · Confirm that reused values accumulate gradients

    Our earlier branch was $q=a^2+a$. In code, `a` appears twice. Both paths reach the
    same object, and both local backward rules use `+=`.

    If either rule overwrote `a.grad`, the result would be `6` or `1` instead of `7`.
    Skip connections, tied weights, recurrent use, and many ordinary expressions rely
    on this same accumulation rule.
    """),

    code(r"""
    shared_value = Value(3.0, label="a")
    branch_result = shared_value ** 2 + shared_value
    branch_result.label = "q"
    branch_result.backward()

    print("q:", branch_result.data)
    print("dq/da:", shared_value.grad)

    assert branch_result.data == 12.0
    assert shared_value.grad == 7.0
    """),

    md(r"""
    ## 10 · Gradient checking is an independent measurement

    Backpropagation gives an analytical gradient. A central finite difference estimates
    the same gradient by measuring two nearby forward passes:

    $$
    g_{\text{numeric}}=
    \frac{L(\theta+\varepsilon)-L(\theta-\varepsilon)}{2\varepsilon}
    $$

    **Symbols:** $\theta$ is one checked scalar; $\varepsilon$ is a small change; and
    $L$ is the same scalar objective used by backpropagation.

    Relative error is:

    $$
    \operatorname{relative\ error}=
    \frac{|g_{\text{autodiff}}-g_{\text{numeric}}|}
    {\max(10^{-8},|g_{\text{autodiff}}|+|g_{\text{numeric}}|)}
    $$

    Gradient checking is too expensive for training because it needs extra forward
    passes for every checked scalar. Its value is independence: a backward bug usually
    disagrees with the measured loss change.

    We use values away from zero and saturation so the test is informative. We check
    every selected leaf, not three convenient entries from one layer.
    """),

    code(r"""
    def numeric_expression(weight_number, input_number, bias_number, target_number):
        prediction_number = math.tanh(weight_number * input_number + bias_number)
        return (prediction_number - target_number) ** 2


    gradient_check_values = {
        "weight": 0.7,
        "input": 1.5,
        "bias": -0.2,
        "target": 0.4,
    }

    # Build the analytical graph once at the unperturbed values.
    checked_weight = Value(gradient_check_values["weight"], label="weight")
    checked_input = Value(gradient_check_values["input"], label="input")
    checked_bias = Value(gradient_check_values["bias"], label="bias")
    checked_target = Value(gradient_check_values["target"], label="target")
    checked_prediction = (checked_weight * checked_input + checked_bias).tanh()
    checked_loss = (checked_prediction - checked_target) ** 2
    checked_loss.backward()

    checked_leaves = {
        "weight": checked_weight,
        "input": checked_input,
        "bias": checked_bias,
        "target": checked_target,
    }
    epsilon = 1e-5
    check_rows = []

    for leaf_name, leaf_node in checked_leaves.items():
        # Change only one leaf so the measured loss difference isolates its derivative.
        plus_values = gradient_check_values.copy()
        minus_values = gradient_check_values.copy()
        plus_values[leaf_name] += epsilon
        minus_values[leaf_name] -= epsilon

        plus_loss = numeric_expression(
            plus_values["weight"], plus_values["input"], plus_values["bias"], plus_values["target"]
        )
        minus_loss = numeric_expression(
            minus_values["weight"], minus_values["input"], minus_values["bias"], minus_values["target"]
        )
        numerical_gradient = (plus_loss - minus_loss) / (2 * epsilon)
        relative_error = abs(leaf_node.grad - numerical_gradient) / max(
            1e-8,
            abs(leaf_node.grad) + abs(numerical_gradient),
        )
        check_rows.append(
            {
                "leaf": leaf_name,
                "autodiff": leaf_node.grad,
                "finite_difference": numerical_gradient,
                "relative_error": relative_error,
            }
        )

    gradient_check_report = pd.DataFrame(check_rows)
    print(gradient_check_report.to_string(index=False))
    print("maximum relative error:", f"{gradient_check_report['relative_error'].max():.2e}")

    assert gradient_check_report["relative_error"].max() < 1e-6
    """),

    md(r"""
    ### What a failed check tells you

    Suppose tanh incorrectly used local derivative `1` everywhere. At the check values,
    the weight gradient would be about `0.874`, while the measured gradient is about
    `0.456`. The disagreement localizes a mathematical or implementation bug.

    Before blaming the backward rule, also verify:

    - both methods use the exact same loss;
    - randomness is frozen;
    - dropout and batch-statistic updates are disabled;
    - checked values are not exactly at a nondifferentiable point;
    - $\varepsilon$ is neither so large that curvature dominates nor so small that
      floating-point rounding dominates.
    """),

    md(r"""
    ## 11 · Why reverse mode fits neural-network training

    For a function with many inputs and many outputs, all first derivatives form a
    **Jacobian**: a table whose rows correspond to outputs and columns to inputs.

    Neural-network training usually has millions of parameter inputs but one scalar
    loss output.

    - **Forward mode** efficiently carries the effect of one chosen input direction
      toward all outputs. This is a Jacobian–vector product, or JVP.
    - **Reverse mode** carries one chosen output sensitivity toward all inputs. This is
      a vector–Jacobian product, or VJP.

    ```mermaid
    flowchart LR
        P[Many parameters] --> G[Recorded computation]
        G --> L[One scalar loss]
        L -. "one reverse seed" .-> G
        G -. "all parameter gradients" .-> P
    ```

    Reverse mode does **not** make computation independent of model size. Both forward
    and backward work scale with the operations and parameters used. Its advantage is
    that one reverse sweep reuses intermediate results instead of perturbing parameters
    one at a time.

    Forward mode can be preferable when inputs are few and outputs are many, or when a
    directional derivative is the desired result. We do not need to materialize a full
    Jacobian for either JVPs or VJPs.
    """),

    md(r"""
    ## 12 · Move from scalars to tensors by following shapes

    For a batched affine layer:

    $$
    Z=XW+b
    $$

    Let:

    - $X$ have shape $(B,d)$;
    - $W$ have shape $(d,h)$;
    - $b$ have shape $(h,)$;
    - $Z$ have shape $(B,h)$;
    - upstream gradient $G=\partial L/\partial Z$ have shape $(B,h)$.

    Then:

    $$
    \frac{\partial L}{\partial X}=GW^T
    $$

    $$
    \frac{\partial L}{\partial W}=X^TG
    $$

    $$
    \frac{\partial L}{\partial b}=\sum_{i=1}^{B}G_i
    $$

    Bias is broadcast across all $B$ rows during the forward pass. Its backward rule
    reverses that broadcast by summing the row contributions. A useful general rule is:

    **Backward must reduce a broadcast gradient back to the original input shape.**

    Matrix multiplication is not elementwise multiplication. The formulas follow the
    dependency paths and must also return arrays shaped exactly like their inputs.
    """),

    code(r"""
    batch_inputs = np.array([[1.0, 2.0], [3.0, 4.0]])
    layer_weights = np.array([[0.5, -1.0, 2.0], [1.5, 0.0, -0.5]])
    layer_bias = np.array([0.1, 0.2, 0.3])
    upstream_gradient = np.array([[1.0, -2.0, 0.5], [0.25, 1.0, -1.0]])

    # Forward broadcasts one bias vector across both batch rows.
    layer_output = batch_inputs @ layer_weights + layer_bias

    # Backward returns one gradient array for each original input shape.
    input_gradient = upstream_gradient @ layer_weights.T
    weight_gradient = batch_inputs.T @ upstream_gradient
    bias_gradient = upstream_gradient.sum(axis=0)

    shape_report = pd.DataFrame(
        [
            {"value": "X", "forward_shape": batch_inputs.shape, "gradient_shape": input_gradient.shape},
            {"value": "W", "forward_shape": layer_weights.shape, "gradient_shape": weight_gradient.shape},
            {"value": "b", "forward_shape": layer_bias.shape, "gradient_shape": bias_gradient.shape},
            {"value": "Z", "forward_shape": layer_output.shape, "gradient_shape": upstream_gradient.shape},
        ]
    )
    print(shape_report.to_string(index=False))
    print("bias gradient:", bias_gradient)

    assert input_gradient.shape == batch_inputs.shape
    assert weight_gradient.shape == layer_weights.shape
    assert bias_gradient.shape == layer_bias.shape
    """),

    md(r"""
    ## 13 · Verify the same graph with PyTorch

    PyTorch autograd uses the same central strategy—record differentiable operations
    during the forward pass and apply local backward rules in reverse—but it is not
    merely our class on a GPU. It additionally handles tensors, broadcasting, devices,
    custom kernels, graph-lifetime rules, complex operators, and performance concerns.

    A tensor created directly with `requires_grad=True` is normally a **leaf**. Model
    parameters are leaves. Results of operations are normally **non-leaf tensors**.
    After backward, leaf gradients are retained in `.grad`. A non-leaf gradient is
    available only if we explicitly call `retain_grad()`.
    """),

    code(r"""
    torch_weight = torch.tensor(0.5, dtype=torch.float64, requires_grad=True)
    torch_input = torch.tensor(2.0, dtype=torch.float64, requires_grad=True)
    torch_bias = torch.tensor(-1.0, dtype=torch.float64, requires_grad=True)
    torch_target = torch.tensor(1.0, dtype=torch.float64)

    # Recreate the scratch expression with identical values and double precision.
    torch_score = torch_weight * torch_input + torch_bias
    torch_score.retain_grad()
    torch_prediction = torch.tanh(torch_score)
    torch_loss = (torch_prediction - torch_target) ** 2
    torch_loss.backward()

    pytorch_comparison = pd.DataFrame(
        [
            {"value": "w", "scratch_gradient": weight.grad, "pytorch_gradient": torch_weight.grad.item()},
            {"value": "x", "scratch_gradient": input_feature.grad, "pytorch_gradient": torch_input.grad.item()},
            {"value": "b", "scratch_gradient": bias.grad, "pytorch_gradient": torch_bias.grad.item()},
            {"value": "z", "scratch_gradient": loss_to_raw_score, "pytorch_gradient": torch_score.grad.item()},
        ]
    )
    print(pytorch_comparison.to_string(index=False))
    print("w is leaf:", torch_weight.is_leaf)
    print("z is leaf:", torch_score.is_leaf)

    assert np.allclose(
        pytorch_comparison["scratch_gradient"],
        pytorch_comparison["pytorch_gradient"],
    )
    """),

    md(r"""
    ## 14 · Gradient state: accumulation, detaching, and no-grad work

    PyTorch accumulates into `.grad` because the same parameter may receive
    contributions from several microbatches or graph branches. In an ordinary training
    step, reset gradients before the next backward pass.

    | Tool | What it does | Common use |
    |---|---|---|
    | `parameter.grad = None` or optimizer zeroing | clears old accumulated gradients | start a new update |
    | `tensor.detach()` | returns a tensor sharing data but disconnected from this graph | logging or intentional stop-gradient |
    | `torch.no_grad()` | disables graph recording inside a block | parameter updates or evaluation |
    | `torch.inference_mode()` | stronger evaluation-oriented graph disabling | model inference |

    Detaching by accident is not a numerical problem: it removes the dependency path,
    so no gradient can cross it.
    """),

    code(r"""
    accumulated_parameter = torch.tensor(3.0, requires_grad=True)

    first_loss = accumulated_parameter ** 2
    first_loss.backward()
    gradient_after_first_backward = accumulated_parameter.grad.item()

    second_loss = accumulated_parameter ** 2
    second_loss.backward()
    gradient_after_second_backward = accumulated_parameter.grad.item()

    # Reset before a logically new gradient calculation.
    accumulated_parameter.grad = None
    third_loss = accumulated_parameter ** 2
    third_loss.backward()
    gradient_after_reset = accumulated_parameter.grad.item()

    detached_value = accumulated_parameter.detach()
    with torch.no_grad():
        updated_value = accumulated_parameter - 0.1 * accumulated_parameter.grad

    print("after first backward:", gradient_after_first_backward)
    print("after second backward without reset:", gradient_after_second_backward)
    print("after reset and backward:", gradient_after_reset)
    print("detached requires_grad:", detached_value.requires_grad)
    print("updated value recorded a graph:", updated_value.requires_grad)

    assert [gradient_after_first_backward, gradient_after_second_backward, gradient_after_reset] == [6.0, 12.0, 6.0]
    """),

    md(r"""
    ## 15 · Gradient flow is repeated multiplication

    Along one path, backpropagation multiplies local derivatives and weight effects.
    If a typical magnitude is $s$ across $k$ steps, the path magnitude behaves like:

    $$
    s^k
    $$

    - If $|s|<1$, repeated multiplication tends to shrink: **vanishing gradients**.
    - If $|s|>1$, it tends to grow: **exploding gradients**.
    - If a path contains an exact zero local derivative, no gradient crosses that step.

    The plot below isolates this multiplication effect. It is not a claim that every
    real network has one constant slope. Actual gradient flow depends on weights,
    activations, data, normalization, residual paths, depth, and floating-point scale.
    DL-04 will test practical remedies under controlled experiments.
    """),

    code(r"""
    path_depths = np.arange(0, 13)
    local_effects = {
        "shrinking effect: 0.25": 0.25,
        "mild shrinking effect: 0.80": 0.80,
        "growing effect: 1.10": 1.10,
    }

    fig, axis = plt.subplots(figsize=(8, 4.5))
    for effect_name, effect_size in local_effects.items():
        # Holding one factor constant reveals only the repeated-product mechanism.
        axis.plot(path_depths, effect_size ** path_depths, marker="o", label=effect_name)

    axis.set_yscale("log")
    axis.set_xlabel("number of backward multiplications")
    axis.set_ylabel("relative path-gradient magnitude")
    axis.set_title("Repeated local effects can shrink or grow a gradient path")
    axis.legend()
    plt.tight_layout()
    plt.show()
    """),

    md(r"""
    ## 16 · Memory: store activations or recompute them

    Backward rules often need forward values. Tanh backward, for example, reused its
    forward output. Saving every activation makes backward fast but consumes memory.

    **Activation checkpointing** saves selected points and recomputes missing forward
    sections during backward.

    ```mermaid
    flowchart TD
        A[Store every activation] --> B[More memory]
        A --> C[Less recomputation]
        D[Store selected checkpoints] --> E[Less memory]
        D --> F[More recomputation]
    ```

    The overhead is not one universal percentage. It depends on which sections are
    saved, their cost, hardware, communication, and implementation. Use checkpointing
    when activation memory is the bottleneck and extra computation is acceptable.
    """),

    md(r"""
    ## 17 · When backpropagation helps—and when it does not directly apply

    ### Use reverse-mode backpropagation when

    - a differentiable model produces a scalar training objective;
    - many parameters influence that objective;
    - exact program derivatives are more useful than parameter-by-parameter estimates;
    - custom differentiable components need verified gradients.

    ### Do not expect ordinary backpropagation to solve

    - a discrete choice with no differentiable path, such as an `argmax` action;
    - a black-box system whose internal computation is unavailable;
    - a truly nondifferentiable objective without a declared surrogate or estimator;
    - model selection, data leakage, or poor evaluation design;
    - optimization by itself—gradients still need an update rule.

    Alternatives depend on the problem: differentiable relaxations, surrogate losses,
    policy-gradient estimators, evolutionary or derivative-free search, or direct
    enumeration for tiny spaces. These methods have different bias, variance, and cost.
    """),

    md(r"""
    ## 18 · Mini-project: use the engine to learn the OR rule

    This project tests whether the engine supports a changing graph across many
    examples—not whether OR is an important machine-learning benchmark.

    **Goal:** train one sigmoid neuron with the scratch engine and confirm that autodiff
    supplies useful gradients.

    **Dataset columns:** `first_bit`, `second_bit`, and `or_target`.

    | first_bit | second_bit | or_target |
    |---:|---:|---:|
    | 0 | 0 | 0 |
    | 0 | 1 | 1 |
    | 1 | 0 | 1 |
    | 1 | 1 | 1 |

    **Workflow:**

    1. initialize two weights and one bias;
    2. build a fresh graph for all four rows;
    3. calculate mean binary cross-entropy;
    4. call `backward()` once;
    5. update only the three parameters;
    6. rebuild the graph on the next step;
    7. inspect probabilities and decisions.

    The first implementation stays deliberately small. A production system would use
    stable logit loss, vectorized tensors, an optimizer, and tested framework kernels.
    """),

    code(r"""
    or_features = [(0.0, 0.0), (0.0, 1.0), (1.0, 0.0), (1.0, 1.0)]
    or_targets = [0.0, 1.0, 1.0, 1.0]

    or_weight_one = Value(0.1, label="w1")
    or_weight_two = Value(-0.1, label="w2")
    or_bias = Value(0.0, label="b")
    or_parameters = [or_weight_one, or_weight_two, or_bias]
    project_losses = []
    project_learning_rate = 0.2

    for training_step in range(300):
        example_losses = []

        for (first_bit, second_bit), target_bit in zip(or_features, or_targets):
            # Each row contributes a fresh prediction path into one shared mean loss.
            example_logit = or_weight_one * first_bit + or_weight_two * second_bit + or_bias
            example_probability = example_logit.sigmoid()
            example_loss = -(
                target_bit * example_probability.log()
                + (1 - target_bit) * (1 - example_probability).log()
            )
            example_losses.append(example_loss)

        mean_project_loss = sum(example_losses) / len(example_losses)
        mean_project_loss.backward()
        project_losses.append(mean_project_loss.data)

        # Parameter updates happen outside the graph, just like a no-grad optimizer step.
        for parameter in or_parameters:
            parameter.data -= project_learning_rate * parameter.grad

    final_or_probabilities = []
    for first_bit, second_bit in or_features:
        final_logit = (
            or_weight_one.data * first_bit
            + or_weight_two.data * second_bit
            + or_bias.data
        )
        final_or_probabilities.append(1 / (1 + math.exp(-final_logit)))

    final_or_decisions = [int(probability >= 0.5) for probability in final_or_probabilities]
    project_report = pd.DataFrame(
        {
            "features": or_features,
            "target": or_targets,
            "probability": final_or_probabilities,
            "decision": final_or_decisions,
        }
    )
    print("first loss:", round(project_losses[0], 5))
    print("final loss:", round(project_losses[-1], 5))
    print(project_report.to_string(index=False))

    assert project_losses[-1] < project_losses[0]
    assert final_or_decisions == [0, 1, 1, 1]
    """),

    md(r"""
    ## 19 · Common mistakes and how to reason through them

    - **Calling backpropagation an optimizer.** Ask whether the operation calculates a
      gradient or changes a parameter.
    - **Starting the loss with gradient zero.** The correct scalar seed is one because
      $\partial L/\partial L=1$.
    - **Overwriting a shared node's gradient.** Contributions from different paths add.
    - **Running backward in forward order.** A node must first receive every later contribution.
    - **Forgetting cached forward values.** Local derivatives often depend on them.
    - **Ignoring broadcast reduction.** Return each gradient to its input's original shape.
    - **Checking only one convenient weight.** Test all parameter families, including biases.
    - **Using different losses for analytical and numerical gradients.** The comparison
      is valid only when objectives match exactly.
    - **Checking at a ReLU kink.** A finite difference can disagree with a chosen
      subgradient exactly at zero without proving the implementation is wrong.
    - **Forgetting to clear framework gradients.** `.grad` accumulates by design.
    - **Detaching a needed path.** No graph edge means no gradient can cross it.
    - **Claiming backward is independent of model size.** Reverse mode is efficient
      because it reuses work, but it still executes the model's graph.
    """),

    md(r"""
    ## 20 · Practice, solutions, and mastery checkpoint

    ### Worked example

    Sections 3 and 4 are the complete worked example: they calculate
    $L=(\tanh(wx+b)-y)^2$ forward and then trace every derivative back to $w$, $x$,
    and $b$. Recalculate that example once without looking before starting the exercises.

    ### Guided practice

    1. For $L=(3w-2)^2$ at $w=1$, calculate the forward loss and $dL/dw$.
    2. For $q=a^2+2a$ at $a=4$, list both path contributions and their sum.

    ### Independent practice

    3. Add a `sin()` operation to `Value` and gradient-check it away from extrema.
    4. Explain why bias gradient for a batch is a sum across rows.
    5. Create a deliberate wrong multiplication rule and show which finite-difference
       checks fail. Restore the correct rule afterward.

    ### Challenge

    Add a scalar `softplus()` operation,

    $$
    \operatorname{softplus}(x)=\log(1+e^x),
    $$

    using existing engine operations. Check inputs `-3`, `0.5`, and `2`; compare every
    gradient with PyTorch; and explain how operation composition creates the backward
    rule without writing a special `softplus` derivative.

    ### Solution and scoring rubric

    1. Forward loss is $(3-2)^2=1$. Let $e=3w-2$; then $dL/dw=2e(3)=6$.
    2. The square path contributes $2a=8$ and the linear path contributes $2$; total is `10`.
    3. `sin()` uses local derivative $\cos(x)$ multiplied by its upstream gradient.
    4. One shared bias affects every batch row, so every row contributes to the same bias element.
    5. A correct independent finite-difference check disagrees wherever the broken rule
       lies on a path from the checked leaf to the loss.

    Award one point for each correct guided or independent answer, three points for a
    correctly checked `sin()` operation, and four points for the softplus challenge.
    Explanations must name both the local derivative and upstream gradient.

    ### Self-check

    Explain without notes:

    1. What is the difference between backpropagation and gradient descent?
    2. What are upstream and local gradients?
    3. Why must graph branches accumulate gradients?
    4. Why does backward use reverse topological order?
    5. Why is the scalar-loss seed one?
    6. Why is reverse mode suitable for many parameters and one loss?
    7. Why does broadcast bias use a sum in backward?
    8. What does gradient checking establish—and what does it not establish?
    9. What is the difference between a PyTorch leaf and non-leaf tensor?
    10. What happens when a tensor is detached?
    11. Why can repeated Jacobian effects shrink or grow gradients?

    ### Readiness threshold

    Award two points for each self-check answer. Score at least **18/22**, and complete
    the OR project plus one fully checked new operation before moving on.
    """),

    md(r"""
    ## Ready to move on?

    ### Quick check

    Without running code, explain why $q=a^2+a$ gives two gradient contributions to
    the same leaf, and state the order in which its graph must run backward.

    ### Teach it back

    Draw a branched computation graph. Calculate it forward, seed the scalar loss with
    one, walk backward in reverse topological order, and show where two gradient paths
    add. Then explain how a `Value` object automates those same steps.

    ### Final readiness gate

    You are ready for DL-04 when you can:

    - calculate one graph manually;
    - distinguish local and upstream gradients;
    - explain accumulation and graph order;
    - implement and gradient-check one new scalar operation;
    - derive affine-layer gradients by shape;
    - match the same graph in PyTorch;
    - explain accumulation, detaching, and no-grad updates;
    - describe vanishing and exploding gradients without claiming one activation fixes everything.

    ### Memory aid

    **Record the forward path, seed the loss with one, reverse the graph, multiply local
    effects, and add every path that meets.**

    ### Next dependency

    Verified gradient flow  
    → required before stable neural training  
    → because DL-04 changes initialization, activations, optimizers, normalization, and
    regularization while measuring how those choices affect optimization and generalization.
    """),
]


build("04_deep_learning/03_backpropagation.ipynb", cells)
