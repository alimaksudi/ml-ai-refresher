"""Build DL-05: convolutional neural networks from pixels to fair evaluation."""

import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
from nbbuild import build, code, md  # noqa: E402


cells = [
    md(r"""
    # DL-05 · Convolutional Neural Networks

    **Prerequisites:** DL-01, DL-02, DL-03, and DL-04  
    **Estimated total mastery time:** 12–15 hours, including practice  
    **Next lesson on the canonical path:** NLP-01 · TF-IDF and Word Embeddings

    A flattened image is just a long row of numbers. That representation hides which
    pixels are neighbors and forces a dense layer to learn separate weights for the same
    pattern at different locations.

    A convolutional neural network, or CNN, makes a useful assumption: nearby pixels
    interact, and the same local detector can be reused across the image. We will earn
    that idea carefully—one patch calculation first, then channels, batches, backward
    gradients, PyTorch, controlled training, and a sealed comparison with an MLP.

    ### Scope boundary

    This lesson covers image classification with small CNNs. Object detection,
    segmentation, large pretrained backbones, and detailed vision deployment are later
    specializations. Transfer learning is introduced as a workflow, not executed with
    an internet-downloaded checkpoint.
    """),

    md(r"""
    ## 1 · What you will be able to do

    By the end, you will be able to:

    - read image tensors in `(batch, channels, height, width)` order;
    - calculate one cross-correlation window manually;
    - explain locality and weight sharing without claiming automatic invariance;
    - calculate output shapes with padding, stride, and dilation;
    - count convolution weights and biases;
    - implement batched multi-channel convolution in NumPy;
    - match the scratch result with PyTorch;
    - derive and implement input, kernel, and bias gradients;
    - verify convolution gradients with finite differences and PyTorch autograd;
    - implement max-pooling forward and backward passes;
    - distinguish translation equivariance, local shift tolerance, and invariance;
    - calculate receptive-field growth through a layer stack;
    - explain `1×1`, grouped, depthwise, dilated, and strided convolution;
    - compare flattening with global average pooling;
    - trace every shape through a PyTorch CNN;
    - compare logistic, MLP, and CNN models under one split and selection contract;
    - run train-only shift augmentation without leaking validation or test rows;
    - inspect learned filters and feature maps without inventing what they represent;
    - evaluate frozen representative checkpoints once on sealed test data.

    ### Learning path

    ```mermaid
    flowchart LR
        A[Image tensor] --> B[One patch]
        B --> C[Sliding filter]
        C --> D[Channels and batches]
        D --> E[Verified backward pass]
        E --> F[Pooling and receptive field]
        F --> G[Modern convolution blocks]
        G --> H[Shape-traced CNN]
        H --> I[Fair MLP/CNN comparison]
        I --> J[Train-only augmentation]
        J --> K[One sealed test]
    ```

    Stable neural training  
    → required before a CNN comparison  
    → because architecture quality cannot be separated from split, optimizer, and checkpoint discipline.
    """),

    md(r"""
    ## 2 · Start with the structure an MLP throws away

    A grayscale `8×8` digit has one channel and 64 pixels. A color photograph normally
    has three channels: red, green, and blue.

    PyTorch image batches use **NCHW** order:

    | Symbol | Meaning |
    |---|---|
    | $N$ | number of images in the batch |
    | $C$ | channels per image |
    | $H$ | image height |
    | $W$ | image width |

    Thus `(32, 3, 224, 224)` means 32 color images, not 32 channels.

    A dense layer from a `224×224×3` image to 1,000 hidden units has about 150 million
    weights. A convolution with 64 filters of shape `3×3×3` has 1,728 weights before
    biases. Convolution has far fewer weights because each filter is reused at every
    spatial location.

    The **weight count** does not grow with image height and width. Compute cost and
    activation memory do grow because the shared filter is still applied at many positions.
    """),

    code(r"""
    import copy
    import math
    import random

    import matplotlib.pyplot as plt
    import numpy as np
    import pandas as pd
    import torch
    import torch.nn.functional as functional
    from sklearn.datasets import load_digits
    from sklearn.dummy import DummyClassifier
    from sklearn.linear_model import LogisticRegression
    from sklearn.metrics import accuracy_score, f1_score, log_loss
    from sklearn.model_selection import train_test_split
    from torch import nn
    from torch.utils.data import DataLoader, TensorDataset

    np.set_printoptions(precision=4, suppress=True)
    DEVICE = torch.device("cpu")


    def set_reproducible(seed):
        random.seed(seed)
        np.random.seed(seed)
        torch.manual_seed(seed)
        torch.use_deterministic_algorithms(True)


    digit_data = load_digits()

    # A fixed instrument range gives both neural models the same pixel scale.
    digit_images = digit_data.images.astype(np.float32) / 16.0
    digit_targets = digit_data.target.astype(np.int64)

    digit_batch_nchw = digit_images[:32, None, :, :]
    dense_weight_count = 224 * 224 * 3 * 1000
    convolution_weight_count = 3 * 3 * 3 * 64

    print("digit image collection:", digit_images.shape)
    print("NCHW example batch:", digit_batch_nchw.shape)
    print("dense weights:", f"{dense_weight_count:,}")
    print("convolution weights before bias:", f"{convolution_weight_count:,}")
    print("weight-count ratio:", f"{dense_weight_count / convolution_weight_count:,.1f}x")
    """),

    md(r"""
    ## 3 · Calculate one filter window by hand

    Deep-learning libraries implement **cross-correlation** and usually call it
    convolution. Mathematical convolution flips the kernel before sliding; neural
    networks do not need that flip because the kernel values are learned.

    Take this image patch:

    $$
    X=
    \begin{bmatrix}
    1&2\\
    3&4
    \end{bmatrix}
    $$

    and this kernel:

    $$
    K=
    \begin{bmatrix}
    1&0\\
    0&-1
    \end{bmatrix}
    $$

    Multiply matching positions and add:

    $$
    (1)(1)+(2)(0)+(3)(0)+(4)(-1)=-3
    $$

    Add bias $b=0.5$:

    $$
    -3+0.5=-2.5
    $$

    That single number says how strongly this patch matches the kernel pattern. Sliding
    the same kernel across the image creates a **feature map**.
    """),

    code(r"""
    manual_patch = np.array([[1.0, 2.0], [3.0, 4.0]])
    manual_kernel = np.array([[1.0, 0.0], [0.0, -1.0]])
    manual_bias = 0.5

    elementwise_products = manual_patch * manual_kernel
    manual_response = elementwise_products.sum() + manual_bias

    print("elementwise products:")
    print(elementwise_products)
    print("sum plus bias:", manual_response)

    assert np.isclose(manual_response, -2.5)
    """),

    md(r"""
    ## 4 · Sliding creates locality and weight sharing

    For a single-channel input $X$, kernel $K$, and bias $b$:

    $$
    Y[i,j]=b+
    \sum_{a=0}^{k_h-1}\sum_{c=0}^{k_w-1}
    X[i+a,j+c]K[a,c]
    $$

    **Symbols:** $i,j$ locate an output; $a,c$ locate a kernel element; $k_h,k_w$ are
    kernel height and width; and $Y$ is the feature map.

    - **Local connectivity:** one output uses a small input neighborhood.
    - **Weight sharing:** every output location uses the same kernel values.
    - **Translation equivariance:** under suitable stride and boundary conditions,
      shifting the input shifts the feature map.

    Equivariance is not invariance. The response moves; it does not disappear. Padding
    boundaries, stride, cropping, and pooling can prevent exact equivariance.
    """),

    code(r"""
    def correlate2d_single(image, kernel, bias=0.0, stride=1, padding=0, dilation=1):
        '''Cross-correlate one 2D image with one 2D kernel.'''
        # Padding happens once; every later index refers to this padded coordinate system.
        padded_image = np.pad(image, ((padding, padding), (padding, padding)))
        kernel_height, kernel_width = kernel.shape
        effective_height = dilation * (kernel_height - 1) + 1
        effective_width = dilation * (kernel_width - 1) + 1
        output_height = (padded_image.shape[0] - effective_height) // stride + 1
        output_width = (padded_image.shape[1] - effective_width) // stride + 1
        output = np.zeros((output_height, output_width), dtype=float)

        for output_row in range(output_height):
            for output_column in range(output_width):
                input_row = output_row * stride
                input_column = output_column * stride
                patch = padded_image[
                    input_row:input_row + effective_height:dilation,
                    input_column:input_column + effective_width:dilation,
                ]
                output[output_row, output_column] = np.sum(patch * kernel) + bias
        return output


    sliding_image = np.array(
        [[1.0, 2.0, 0.0], [3.0, 4.0, 1.0], [0.0, 2.0, 3.0]]
    )
    sliding_output = correlate2d_single(sliding_image, manual_kernel, manual_bias)

    print("input shape:", sliding_image.shape)
    print("output shape:", sliding_output.shape)
    print(sliding_output)
    assert np.isclose(sliding_output[0, 0], manual_response)
    """),

    md(r"""
    ## 5 · Output shape comes before architecture code

    For one spatial dimension:

    $$
    N_{out}=
    \left\lfloor
    \frac{N+2p-d(k-1)-1}{s}
    \right\rfloor+1
    $$

    **Symbols:** $N$ is input size; $p$ is padding on each side; $d$ is dilation;
    $k$ is kernel size; and $s$ is stride.

    Apply the formula separately to height and width.

    - **Padding** adds a border and controls shrinkage and boundary treatment.
    - **Stride** controls how far the kernel moves and can downsample.
    - **Dilation** spaces kernel elements apart, increasing the covered area without
      adding weights.

    “Same” output size is simple for odd kernels, stride one, and symmetric padding.
    Other combinations may need asymmetric padding, so calculate rather than memorize.
    """),

    code(r"""
    def convolution_output_size(input_size, kernel_size, padding=0, stride=1, dilation=1):
        # Dilation changes the covered width without changing the number of kernel values.
        effective_kernel = dilation * (kernel_size - 1) + 1
        return (input_size + 2 * padding - effective_kernel) // stride + 1


    shape_examples = pd.DataFrame(
        [
            {"input": 28, "kernel": 3, "padding": 0, "stride": 1, "dilation": 1},
            {"input": 28, "kernel": 3, "padding": 1, "stride": 1, "dilation": 1},
            {"input": 28, "kernel": 3, "padding": 1, "stride": 2, "dilation": 1},
            {"input": 28, "kernel": 3, "padding": 2, "stride": 1, "dilation": 2},
        ]
    )
    shape_examples["output"] = shape_examples.apply(
        lambda row: convolution_output_size(
            row["input"], row["kernel"], row["padding"], row["stride"], row["dilation"]
        ),
        axis=1,
    )
    print(shape_examples.to_string(index=False))
    """),

    md(r"""
    ## 6 · Real convolution mixes channels and produces channels

    Let input have shape $(N,C_{in},H,W)$ and weights have shape:

    $$
    (C_{out},C_{in},k_h,k_w)
    $$

    Every output filter contains a small kernel for **every input channel**. Their patch
    responses are summed, then one output bias is added. One filter produces one output
    channel.

    Parameter count including bias is:

    $$
    C_{out}(C_{in}k_hk_w+1)
    $$

    For 16 filters, 3 input channels, and `5×5` kernels:

    $$
    16(3\times5\times5+1)=1216
    $$

    The output shape is $(N,C_{out},H_{out},W_{out})$. Bias broadcasts across batch,
    height, and width.
    """),

    code(r"""
    def correlate2d_nchw(inputs, weights, bias, stride=1, padding=0, dilation=1):
        '''Batched multi-channel cross-correlation in NCHW order.'''
        # Validate channel compatibility before entering the deliberately explicit loops.
        batch_size, input_channels, input_height, input_width = inputs.shape
        output_channels, weight_input_channels, kernel_height, kernel_width = weights.shape
        if input_channels != weight_input_channels:
            raise ValueError("input channels and kernel channels must match")

        output_height = convolution_output_size(input_height, kernel_height, padding, stride, dilation)
        output_width = convolution_output_size(input_width, kernel_width, padding, stride, dilation)
        padded_inputs = np.pad(inputs, ((0, 0), (0, 0), (padding, padding), (padding, padding)))
        outputs = np.zeros((batch_size, output_channels, output_height, output_width), dtype=float)

        for batch_index in range(batch_size):
            for output_channel in range(output_channels):
                for output_row in range(output_height):
                    for output_column in range(output_width):
                        input_row = output_row * stride
                        input_column = output_column * stride
                        patch = padded_inputs[
                            batch_index,
                            :,
                            input_row:input_row + dilation * (kernel_height - 1) + 1:dilation,
                            input_column:input_column + dilation * (kernel_width - 1) + 1:dilation,
                        ]
                        outputs[batch_index, output_channel, output_row, output_column] = (
                            np.sum(patch * weights[output_channel]) + bias[output_channel]
                        )
        return outputs


    comparison_generator = np.random.default_rng(4)
    comparison_inputs = comparison_generator.normal(size=(2, 2, 5, 4))
    comparison_weights = comparison_generator.normal(size=(3, 2, 3, 2))
    comparison_bias = comparison_generator.normal(size=3)
    scratch_convolution = correlate2d_nchw(
        comparison_inputs,
        comparison_weights,
        comparison_bias,
        stride=1,
        padding=1,
    )
    torch_convolution = functional.conv2d(
        torch.tensor(comparison_inputs, dtype=torch.float64),
        torch.tensor(comparison_weights, dtype=torch.float64),
        torch.tensor(comparison_bias, dtype=torch.float64),
        stride=1,
        padding=1,
    ).numpy()

    print("scratch output shape:", scratch_convolution.shape)
    print("maximum PyTorch difference:", np.max(np.abs(scratch_convolution - torch_convolution)))
    assert np.allclose(scratch_convolution, torch_convolution, atol=1e-10)
    """),

    md(r"""
    ## 7 · Backpropagation through convolution reuses every patch

    Let upstream gradient be:

    $$
    G=\frac{\partial L}{\partial Y}
    $$

    Each output location used one input patch and one filter. During backward:

    - the filter gradient adds `patch × upstream value` from every batch and location;
    - the input gradient adds `filter × upstream value` into every contributing patch;
    - the bias gradient sums upstream values across batch and spatial positions.

    Weight sharing causes accumulation: one shared weight affects many output positions,
    so all of those paths contribute to its gradient.

    We implement the rules directly and then check every scalar in a tiny example using
    central finite differences. Training does not begin until this independent check passes.
    """),

    code(r"""
    def correlate2d_backward_nchw(inputs, weights, upstream, stride=1, padding=0, dilation=1):
        '''Return gradients for inputs, weights, and bias of correlate2d_nchw.'''
        # Allocate one destination for each original forward input.
        batch_size, input_channels, input_height, input_width = inputs.shape
        output_channels, _, kernel_height, kernel_width = weights.shape
        _, _, output_height, output_width = upstream.shape
        padded_inputs = np.pad(inputs, ((0, 0), (0, 0), (padding, padding), (padding, padding)))
        padded_input_gradient = np.zeros_like(padded_inputs, dtype=float)
        weight_gradient = np.zeros_like(weights, dtype=float)
        bias_gradient = upstream.sum(axis=(0, 2, 3))

        for batch_index in range(batch_size):
            for output_channel in range(output_channels):
                for output_row in range(output_height):
                    for output_column in range(output_width):
                        input_row = output_row * stride
                        input_column = output_column * stride
                        upstream_value = upstream[batch_index, output_channel, output_row, output_column]
                        patch = padded_inputs[
                            batch_index,
                            :,
                            input_row:input_row + dilation * (kernel_height - 1) + 1:dilation,
                            input_column:input_column + dilation * (kernel_width - 1) + 1:dilation,
                        ]
                        weight_gradient[output_channel] += patch * upstream_value
                        padded_input_gradient[
                            batch_index,
                            :,
                            input_row:input_row + dilation * (kernel_height - 1) + 1:dilation,
                            input_column:input_column + dilation * (kernel_width - 1) + 1:dilation,
                        ] += weights[output_channel] * upstream_value

        if padding == 0:
            input_gradient = padded_input_gradient
        else:
            input_gradient = padded_input_gradient[:, :, padding:-padding, padding:-padding]
        return input_gradient, weight_gradient, bias_gradient


    check_generator = np.random.default_rng(9)
    check_inputs = check_generator.normal(size=(1, 1, 4, 4))
    check_weights = check_generator.normal(size=(1, 1, 2, 2))
    check_bias = check_generator.normal(size=1)
    check_output = correlate2d_nchw(check_inputs, check_weights, check_bias)
    check_upstream = check_generator.normal(size=check_output.shape)
    analytic_input_gradient, analytic_weight_gradient, analytic_bias_gradient = correlate2d_backward_nchw(
        check_inputs,
        check_weights,
        check_upstream,
    )
    """),

    code(r"""
    def convolution_scalar_loss(inputs, weights, bias, upstream):
        output = correlate2d_nchw(inputs, weights, bias)
        return float(np.sum(output * upstream))


    epsilon = 1e-5
    gradient_check_rows = []
    checked_arrays = [
        ("input", check_inputs, analytic_input_gradient),
        ("weight", check_weights, analytic_weight_gradient),
        ("bias", check_bias, analytic_bias_gradient),
    ]

    for array_name, checked_array, analytic_gradient in checked_arrays:
        for scalar_index in np.ndindex(checked_array.shape):
            original_value = checked_array[scalar_index]

            # Perturb one scalar while every other input to the objective remains fixed.
            checked_array[scalar_index] = original_value + epsilon
            plus_loss = convolution_scalar_loss(check_inputs, check_weights, check_bias, check_upstream)
            checked_array[scalar_index] = original_value - epsilon
            minus_loss = convolution_scalar_loss(check_inputs, check_weights, check_bias, check_upstream)
            checked_array[scalar_index] = original_value

            numerical_gradient = (plus_loss - minus_loss) / (2 * epsilon)
            analytical_value = analytic_gradient[scalar_index]
            relative_error = abs(analytical_value - numerical_gradient) / max(
                1e-8,
                abs(analytical_value) + abs(numerical_gradient),
            )
            gradient_check_rows.append(
                {"array": array_name, "index": scalar_index, "relative_error": relative_error}
            )

    convolution_gradient_report = pd.DataFrame(gradient_check_rows)
    print(convolution_gradient_report.groupby("array")["relative_error"].max().to_string())
    print("maximum relative error:", f"{convolution_gradient_report['relative_error'].max():.2e}")
    assert convolution_gradient_report["relative_error"].max() < 1e-6
    """),

    code(r"""
    torch_check_inputs = torch.tensor(check_inputs, dtype=torch.float64, requires_grad=True)
    torch_check_weights = torch.tensor(check_weights, dtype=torch.float64, requires_grad=True)
    torch_check_bias = torch.tensor(check_bias, dtype=torch.float64, requires_grad=True)
    torch_check_upstream = torch.tensor(check_upstream, dtype=torch.float64)

    # A dot product with arbitrary upstream values tests the same vector-Jacobian product.
    torch_check_output = functional.conv2d(torch_check_inputs, torch_check_weights, torch_check_bias)
    torch_check_loss = torch.sum(torch_check_output * torch_check_upstream)
    torch_check_loss.backward()

    print("input gradient matches PyTorch:", np.allclose(analytic_input_gradient, torch_check_inputs.grad.numpy()))
    print("weight gradient matches PyTorch:", np.allclose(analytic_weight_gradient, torch_check_weights.grad.numpy()))
    print("bias gradient matches PyTorch:", np.allclose(analytic_bias_gradient, torch_check_bias.grad.numpy()))

    assert np.allclose(analytic_input_gradient, torch_check_inputs.grad.numpy(), atol=1e-10)
    assert np.allclose(analytic_weight_gradient, torch_check_weights.grad.numpy(), atol=1e-10)
    assert np.allclose(analytic_bias_gradient, torch_check_bias.grad.numpy(), atol=1e-10)
    """),

    md(r"""
    ## 8 · Max-pooling stores a routing decision

    Max-pooling has no learned weights. It takes the maximum inside each window and
    usually downsamples.

    During backward, the upstream value goes only to the input position selected as the
    maximum. Every other position receives zero from that output.

    If several values tie, the mathematical max is nondifferentiable at that point.
    Libraries choose a convention. Our small implementation sends the gradient to the
    first maximum in row-major order. Do not gradient-check exactly at a tie unless you
    account for the declared convention.
    """),

    code(r"""
    def max_pool2d_forward(image, pool_size=2, stride=2):
        # Cache the selected input index because backward cannot recover it from the max alone.
        output_height = (image.shape[0] - pool_size) // stride + 1
        output_width = (image.shape[1] - pool_size) // stride + 1
        output = np.zeros((output_height, output_width), dtype=float)
        selected_indices = {}

        for output_row in range(output_height):
            for output_column in range(output_width):
                input_row = output_row * stride
                input_column = output_column * stride
                window = image[input_row:input_row + pool_size, input_column:input_column + pool_size]
                flat_index = int(np.argmax(window))
                local_row, local_column = np.unravel_index(flat_index, window.shape)
                output[output_row, output_column] = window[local_row, local_column]
                selected_indices[(output_row, output_column)] = (
                    input_row + local_row,
                    input_column + local_column,
                )
        return output, selected_indices


    def max_pool2d_backward(upstream, input_shape, selected_indices):
        input_gradient = np.zeros(input_shape, dtype=float)
        for output_index, input_index in selected_indices.items():
            input_gradient[input_index] += upstream[output_index]
        return input_gradient


    pooling_input = np.array(
        [[1.0, 5.0, 2.0, 4.0], [3.0, 0.0, 7.0, 6.0], [8.0, 2.0, 1.0, 3.0], [4.0, 9.0, 5.0, 0.0]]
    )
    pooled_output, pooling_switches = max_pool2d_forward(pooling_input)
    pooling_upstream = np.ones_like(pooled_output)
    pooling_input_gradient = max_pool2d_backward(pooling_upstream, pooling_input.shape, pooling_switches)

    print("pooled output:")
    print(pooled_output)
    print("input gradient routes:")
    print(pooling_input_gradient)
    assert pooling_input_gradient.sum() == pooling_upstream.sum()
    """),

    md(r"""
    ## 9 · Equivariance, tolerance, and invariance are different promises

    - **Equivariance:** shifting the input shifts the feature map correspondingly.
    - **Local shift tolerance:** a small shift may leave a pooled or downstream response
      similar, but not necessarily identical.
    - **Invariance:** a representation or prediction remains unchanged under a transformation.

    A stride-one shared filter is approximately translation equivariant away from
    boundaries. Stride and pooling discard positions, so a one-pixel shift can change
    which sampling window receives a feature. Global max or average pooling removes
    spatial position more aggressively, but even a complete CNN is not guaranteed to
    classify every translated, rotated, or scaled image identically.

    Augmentation teaches selected transformations from data; it also does not create a
    universal guarantee.
    """),

    code(r"""
    first_canvas = np.zeros((12, 12))
    second_canvas = np.zeros((12, 12))
    first_canvas[3:6, 3:6] = 1.0
    second_canvas[5:8, 6:9] = 1.0
    bright_patch_detector = np.ones((3, 3))

    first_response = correlate2d_single(first_canvas, bright_patch_detector)
    second_response = correlate2d_single(second_canvas, bright_patch_detector)
    first_peak = np.unravel_index(np.argmax(first_response), first_response.shape)
    second_peak = np.unravel_index(np.argmax(second_response), second_response.shape)

    print("first response peak:", first_peak)
    print("second response peak:", second_peak)
    print("observed peak shift:", tuple(np.subtract(second_peak, first_peak)))
    print("global maximums:", first_response.max(), second_response.max())

    assert tuple(np.subtract(second_peak, first_peak)) == (2, 3)
    assert first_response.max() == second_response.max()
    """),

    md(r"""
    ## 10 · Receptive field grows through kernels and strides

    A unit's **theoretical receptive field** is the input region that can influence it.
    Track two numbers:

    - $r_l$: receptive-field size after layer $l$;
    - $j_l$: input-space jump between adjacent outputs after layer $l$.

    Starting with $r_0=1$ and $j_0=1$:

    $$
    r_l=r_{l-1}+[d_l(k_l-1)]j_{l-1}
    $$

    $$
    j_l=j_{l-1}s_l
    $$

    **Symbols:** $k_l$, $s_l$, and $d_l$ are kernel, stride, and dilation at layer $l$.

    Padding changes alignment and boundary coverage but not this theoretical size.
    Actual learned influence—the **effective receptive field**—may occupy only part of
    the theoretical region.
    """),

    code(r"""
    def receptive_field_trace(layer_specs):
        # Jump must be updated after receptive field because the current kernel uses the old spacing.
        receptive_field = 1
        jump = 1
        rows = []

        for layer_name, kernel_size, stride, dilation in layer_specs:
            receptive_field = receptive_field + dilation * (kernel_size - 1) * jump
            jump = jump * stride
            rows.append(
                {
                    "layer": layer_name,
                    "kernel": kernel_size,
                    "stride": stride,
                    "dilation": dilation,
                    "receptive_field": receptive_field,
                    "jump": jump,
                }
            )
        return pd.DataFrame(rows)


    receptive_field_report = receptive_field_trace(
        [
            ("conv 3x3", 3, 1, 1),
            ("pool 2x2", 2, 2, 1),
            ("conv 3x3", 3, 1, 1),
            ("dilated conv", 3, 1, 2),
        ]
    )
    print(receptive_field_report.to_string(index=False))
    assert receptive_field_report.iloc[2]["receptive_field"] == 8
    """),

    md(r"""
    ## 11 · Modern convolution changes connectivity, not the core idea

    | Variant | What changes | Why use it |
    |---|---|---|
    | `1×1` convolution | mixes channels at each location | change channel width cheaply |
    | strided convolution | moves more than one pixel | learned downsampling |
    | dilated convolution | spaces kernel positions apart | larger field without more weights |
    | grouped convolution | splits channels into groups | lower compute or structured mixing |
    | depthwise convolution | one spatial kernel per input channel | very low-cost spatial processing |
    | depthwise + `1×1` | spatial filtering then channel mixing | mobile-efficient separable block |

    For groups $g$, parameter count before bias is:

    $$
    C_{out}\left(\frac{C_{in}}{g}\right)k_hk_w
    $$

    Depthwise convolution uses $g=C_{in}$, usually with $C_{out}=C_{in}$ or a channel
    multiplier. A following `1×1` layer mixes information across channels.

    These variants save weights and multiply-adds, but hardware latency also depends on
    memory movement and kernel implementation. Fewer parameters does not always mean
    proportionally faster execution.
    """),

    code(r"""
    standard_parameters = 64 * 64 * 3 * 3
    depthwise_parameters = 64 * 1 * 3 * 3
    pointwise_parameters = 64 * 64 * 1 * 1
    separable_parameters = depthwise_parameters + pointwise_parameters

    variant_report = pd.DataFrame(
        [
            {"operation": "standard 3x3", "weights": standard_parameters},
            {"operation": "depthwise 3x3", "weights": depthwise_parameters},
            {"operation": "pointwise 1x1", "weights": pointwise_parameters},
            {"operation": "depthwise + pointwise", "weights": separable_parameters},
        ]
    )
    print(variant_report.to_string(index=False))
    print("separable / standard ratio:", round(separable_parameters / standard_parameters, 3))
    """),

    md(r"""
    ## 12 · Flattening and global average pooling make different heads

    **Flattening** preserves every remaining spatial position and connects all of them
    to a dense classifier. It can add many parameters and makes the head depend on a
    fixed feature-map size.

    **Global average pooling**, or GAP, averages each channel across height and width.
    It returns one value per channel, sharply reducing head parameters and discarding
    explicit position.

    GAP can regularize and accept different spatial sizes, but it may remove location
    information needed by the task. Choose it because the task supports spatial
    summarization, not because it is always more modern.
    """),

    md(r"""
    ## 13 · Trace a complete PyTorch CNN before training it

    Our digit CNN uses two convolution blocks and a flattening head:

    ```text
    (N,1,8,8)
      → Conv 1→8, same padding
      → ReLU
      → MaxPool: 8×8 to 4×4
      → Conv 8→16, same padding
      → ReLU
      → MaxPool: 4×4 to 2×2
      → Flatten: 16×2×2 = 64
      → Linear: 10 logits
    ```

    We print every shape and compare the CNN's parameter count with a roughly matched
    MLP. A close parameter budget does not make their computations identical; that is
    the point of testing the spatial inductive bias.
    """),

    code(r"""
    class DigitCNN(nn.Module):
        def __init__(self):
            super().__init__()
            # Two small blocks make all spatial changes easy to trace on an 8x8 image.
            self.first_convolution = nn.Conv2d(1, 8, kernel_size=3, padding=1)
            self.first_pool = nn.MaxPool2d(2)
            self.second_convolution = nn.Conv2d(8, 16, kernel_size=3, padding=1)
            self.second_pool = nn.MaxPool2d(2)
            self.classifier = nn.Linear(16 * 2 * 2, 10)

        def forward_with_shapes(self, inputs):
            shape_trace = [("input", tuple(inputs.shape))]
            features = torch.relu(self.first_convolution(inputs))
            shape_trace.append(("first conv + ReLU", tuple(features.shape)))
            features = self.first_pool(features)
            shape_trace.append(("first pool", tuple(features.shape)))
            features = torch.relu(self.second_convolution(features))
            shape_trace.append(("second conv + ReLU", tuple(features.shape)))
            features = self.second_pool(features)
            shape_trace.append(("second pool", tuple(features.shape)))
            flattened = features.flatten(start_dim=1)
            shape_trace.append(("flatten", tuple(flattened.shape)))
            logits = self.classifier(flattened)
            shape_trace.append(("logits", tuple(logits.shape)))
            return logits, shape_trace

        def forward(self, inputs):
            features = torch.relu(self.first_convolution(inputs))
            features = self.first_pool(features)
            features = torch.relu(self.second_convolution(features))
            features = self.second_pool(features)
            return self.classifier(features.flatten(start_dim=1))


    class MatchedDigitMLP(nn.Module):
        def __init__(self):
            super().__init__()
            self.network = nn.Sequential(nn.Flatten(), nn.Linear(64, 25), nn.ReLU(), nn.Linear(25, 10))

        def forward(self, inputs):
            return self.network(inputs)


    shape_model = DigitCNN()
    example_logits, example_shape_trace = shape_model.forward_with_shapes(torch.zeros(4, 1, 8, 8))
    cnn_parameter_count = sum(parameter.numel() for parameter in shape_model.parameters())
    mlp_parameter_count = sum(parameter.numel() for parameter in MatchedDigitMLP().parameters())

    print(pd.DataFrame(example_shape_trace, columns=["stage", "shape"]).to_string(index=False))
    print("CNN parameters:", cnn_parameter_count)
    print("matched MLP parameters:", mlp_parameter_count)
    assert example_logits.shape == (4, 10)
    """),

    md(r"""
    ## 14 · Freeze the data and training contract

    We now reuse DL-04's discipline:

    - stratified 60% train, 20% validation, and 20% sealed test;
    - pixels divided by their known maximum `16` before the split-independent reshape;
    - identical train rows, minibatch size, AdamW recipe, epoch budget, patience, and
      validation log-loss selection for MLP and CNN;
    - no test access inside training;
    - three declared seeds after the augmentation choice;
    - representative median-validation seed for final reporting.

    Dividing by a known instrument scale does not estimate statistics from held-out
    rows. Dataset-dependent means or variances would need to be fitted on training only.
    """),

    code(r"""
    all_image_indices = np.arange(len(digit_targets))

    # Seal test first, then derive validation only from the remaining development rows.
    development_indices, sealed_test_indices = train_test_split(
        all_image_indices,
        test_size=0.20,
        stratify=digit_targets,
        random_state=42,
    )
    train_indices, validation_indices = train_test_split(
        development_indices,
        test_size=0.25,
        stratify=digit_targets[development_indices],
        random_state=42,
    )

    train_images = digit_images[train_indices, None, :, :]
    train_labels = digit_targets[train_indices]
    validation_images = digit_images[validation_indices, None, :, :]
    validation_labels = digit_targets[validation_indices]
    sealed_test_images = digit_images[sealed_test_indices, None, :, :]
    sealed_test_labels = digit_targets[sealed_test_indices]

    train_image_tensor = torch.tensor(train_images)
    train_label_tensor = torch.tensor(train_labels)
    validation_image_tensor = torch.tensor(validation_images)
    validation_label_tensor = torch.tensor(validation_labels)
    sealed_test_image_tensor = torch.tensor(sealed_test_images)
    sealed_test_label_tensor = torch.tensor(sealed_test_labels)

    split_report = pd.DataFrame(
        [
            {"split": "train", "rows": len(train_indices), "use": "fit parameters"},
            {"split": "validation", "rows": len(validation_indices), "use": "select augmentation and checkpoint"},
            {"split": "sealed test", "rows": len(sealed_test_indices), "use": "one final report"},
        ]
    )
    print(split_report.to_string(index=False))
    print("test status: sealed")
    """),

    md(r"""
    ## 15 · Augmentation belongs only inside the training boundary

    A one-pixel shift is a realistic nuisance for small digit images. We create one
    deterministic shifted copy per training image and combine it with the originals.
    Validation and test remain untouched so they continue to measure the original
    target distribution.

    Augmentation encodes a belief that the label should survive the transformation.
    Horizontal flipping is sensible for many natural objects but can change letters,
    digits, medical laterality, or traffic direction. Always justify the transformation.
    """),

    code(r"""
    def shift_image_without_wrap(image, row_shift, column_shift):
        # Copy only the overlapping source rectangle; vacated pixels remain zero.
        shifted = np.zeros_like(image)
        source_row_start = max(0, -row_shift)
        source_row_end = min(image.shape[0], image.shape[0] - row_shift)
        source_column_start = max(0, -column_shift)
        source_column_end = min(image.shape[1], image.shape[1] - column_shift)
        target_row_start = max(0, row_shift)
        target_column_start = max(0, column_shift)
        shifted[
            target_row_start:target_row_start + (source_row_end - source_row_start),
            target_column_start:target_column_start + (source_column_end - source_column_start),
        ] = image[source_row_start:source_row_end, source_column_start:source_column_end]
        return shifted


    def augmented_training_tensors(seed):
        random_generator = np.random.default_rng(seed)
        shift_choices = np.array([[-1, 0], [1, 0], [0, -1], [0, 1]])
        shifted_images = []

        for image in train_images[:, 0]:
            row_shift, column_shift = shift_choices[random_generator.integers(len(shift_choices))]
            shifted_images.append(shift_image_without_wrap(image, int(row_shift), int(column_shift)))

        combined_images = np.concatenate([train_images, np.asarray(shifted_images)[:, None, :, :]], axis=0)
        combined_labels = np.concatenate([train_labels, train_labels], axis=0)
        return torch.tensor(combined_images), torch.tensor(combined_labels)


    example_shifted = shift_image_without_wrap(train_images[0, 0], 0, 1)
    fig, axes = plt.subplots(1, 2, figsize=(5, 2.5))
    axes[0].imshow(train_images[0, 0], cmap="gray")
    axes[0].set_title("original training row")
    axes[1].imshow(example_shifted, cmap="gray")
    axes[1].set_title("one-pixel shift")
    for axis in axes:
        axis.axis("off")
    plt.tight_layout()
    plt.show()
    """),

    md(r"""
    ## 16 · Train with validation checkpoints, never test feedback

    The helper initializes convolution and hidden ReLU layers with He initialization
    and output logits with Xavier initialization. It records training and validation
    loss, restores an independent best-validation state, and returns without seeing test.

    Logistic regression is a useful non-neural baseline. The MLP and CNN have similar
    parameter counts but different connectivity. The CNN is not required to win on tiny
    `8×8` images; the experiment asks whether its spatial bias earns its complexity here.
    """),

    code(r"""
    def initialize_classifier(model):
        # The final Linear is the logits layer regardless of how the model names it.
        linear_layers = [layer for layer in model.modules() if isinstance(layer, nn.Linear)]
        output_layer = linear_layers[-1]

        for _, layer in model.named_modules():
            if not isinstance(layer, (nn.Conv2d, nn.Linear)):
                continue
            if layer is output_layer:
                nn.init.xavier_normal_(layer.weight)
            else:
                nn.init.kaiming_normal_(layer.weight, nonlinearity="relu")
            if layer.bias is not None:
                nn.init.zeros_(layer.bias)


    def evaluate_classifier(model, image_tensor, label_tensor):
        model.eval()
        with torch.inference_mode():
            logits = model(image_tensor.to(DEVICE))
            probabilities = torch.softmax(logits, dim=1).cpu().numpy()
        predictions = probabilities.argmax(axis=1)
        labels = label_tensor.numpy()
        return {
            "loss": float(log_loss(labels, probabilities, labels=list(range(10)))),
            "accuracy": float(accuracy_score(labels, predictions)),
            "macro_f1": float(f1_score(labels, predictions, average="macro")),
        }


    def train_image_classifier(model_factory, seed=42, augment=False, maximum_epochs=50, patience=7):
        # This function receives no test tensor, making test-based selection impossible here.
        set_reproducible(seed)
        model = model_factory().to(DEVICE)
        initialize_classifier(model)
        optimizer = torch.optim.AdamW(model.parameters(), lr=0.002, weight_decay=1e-4)

        if augment:
            training_features, training_targets = augmented_training_tensors(seed)
        else:
            training_features, training_targets = train_image_tensor, train_label_tensor
        training_loader = DataLoader(
            TensorDataset(training_features, training_targets),
            batch_size=64,
            shuffle=True,
            generator=torch.Generator().manual_seed(seed),
        )

        best_validation_loss = float("inf")
        best_state = None
        best_epoch = None
        stale_epochs = 0
        history_rows = []

        for epoch in range(maximum_epochs):
            model.train()
            training_loss_sum = 0.0

            for batch_images, batch_labels in training_loader:
                optimizer.zero_grad(set_to_none=True)
                logits = model(batch_images.to(DEVICE))
                loss = functional.cross_entropy(logits, batch_labels.to(DEVICE))
                if not torch.isfinite(loss):
                    raise RuntimeError("non-finite training loss")
                loss.backward()
                optimizer.step()
                training_loss_sum += loss.item() * len(batch_labels)

            validation_metrics = evaluate_classifier(model, validation_image_tensor, validation_label_tensor)
            history_rows.append(
                {
                    "epoch": epoch,
                    "training_loss": training_loss_sum / len(training_features),
                    "validation_loss": validation_metrics["loss"],
                    "validation_accuracy": validation_metrics["accuracy"],
                }
            )

            if validation_metrics["loss"] < best_validation_loss - 1e-4:
                best_validation_loss = validation_metrics["loss"]
                best_state = copy.deepcopy(model.state_dict())
                best_epoch = epoch
                stale_epochs = 0
            else:
                stale_epochs += 1
                if stale_epochs >= patience:
                    break

        if best_state is None:
            raise RuntimeError("training produced no validation checkpoint")
        model.load_state_dict(best_state)
        return model, pd.DataFrame(history_rows), {
            "best_epoch": best_epoch,
            "best_validation_loss": best_validation_loss,
            "seed": seed,
            "augment": augment,
        }
    """),

    md(r"""
    ## 17 · Baselines and one-variable augmentation choice

    The majority and logistic models establish task difficulty without neural training.
    The first MLP and CNN runs use the same seed. Then the augmentation ablation changes
    only whether deterministic shifted training copies are included.

    We select augmentation by validation log loss, not by whether it supports a preferred
    story. An augmentation may hurt if it crops important strokes or mismatches deployment.
    """),

    code(r"""
    flattened_train_images = train_images.reshape(len(train_images), -1)
    flattened_validation_images = validation_images.reshape(len(validation_images), -1)

    # Classical baselines see the same normalized pixels and the same split.
    majority_baseline = DummyClassifier(strategy="most_frequent")
    majority_baseline.fit(flattened_train_images, train_labels)
    majority_validation_probabilities = majority_baseline.predict_proba(flattened_validation_images)
    majority_validation_predictions = majority_baseline.predict(flattened_validation_images)

    logistic_baseline = LogisticRegression(max_iter=3000, random_state=42)
    logistic_baseline.fit(flattened_train_images, train_labels)
    logistic_validation_probabilities = logistic_baseline.predict_proba(flattened_validation_images)
    logistic_validation_predictions = logistic_baseline.predict(flattened_validation_images)

    mlp_model, mlp_history, mlp_metadata = train_image_classifier(MatchedDigitMLP, seed=42)
    cnn_plain_model, cnn_plain_history, cnn_plain_metadata = train_image_classifier(DigitCNN, seed=42, augment=False)
    cnn_augmented_model, cnn_augmented_history, cnn_augmented_metadata = train_image_classifier(
        DigitCNN,
        seed=42,
        augment=True,
    )

    validation_comparison = pd.DataFrame(
        [
            {
                "candidate": "majority",
                "validation_loss": log_loss(validation_labels, majority_validation_probabilities, labels=list(range(10))),
                "validation_accuracy": accuracy_score(validation_labels, majority_validation_predictions),
                "parameters": 0,
            },
            {
                "candidate": "logistic",
                "validation_loss": log_loss(validation_labels, logistic_validation_probabilities, labels=list(range(10))),
                "validation_accuracy": accuracy_score(validation_labels, logistic_validation_predictions),
                "parameters": logistic_baseline.coef_.size + logistic_baseline.intercept_.size,
            },
            {
                "candidate": "matched MLP",
                "validation_loss": mlp_metadata["best_validation_loss"],
                "validation_accuracy": evaluate_classifier(mlp_model, validation_image_tensor, validation_label_tensor)["accuracy"],
                "parameters": mlp_parameter_count,
            },
            {
                "candidate": "CNN, original rows",
                "validation_loss": cnn_plain_metadata["best_validation_loss"],
                "validation_accuracy": evaluate_classifier(cnn_plain_model, validation_image_tensor, validation_label_tensor)["accuracy"],
                "parameters": cnn_parameter_count,
            },
            {
                "candidate": "CNN, shifted augmentation",
                "validation_loss": cnn_augmented_metadata["best_validation_loss"],
                "validation_accuracy": evaluate_classifier(cnn_augmented_model, validation_image_tensor, validation_label_tensor)["accuracy"],
                "parameters": cnn_parameter_count,
            },
        ]
    ).sort_values("validation_loss")

    selected_augmentation = bool(
        cnn_augmented_metadata["best_validation_loss"] < cnn_plain_metadata["best_validation_loss"]
    )
    print(validation_comparison.round(4).to_string(index=False))
    print("validation-selected CNN augmentation:", selected_augmentation)
    print("test status: sealed")
    """),

    md(r"""
    ## 18 · Curves reveal optimization; validation compares candidates

    A worse CNN result would not prove convolution is useless. Digits are only `8×8`,
    a flattened MLP is already strong, and the architecture and budget are small. A
    fair conclusion stays local to this dataset and experiment.

    We can still ask:

    - Did each neural model optimize normally?
    - Did augmentation improve the declared validation metric?
    - Are parameter budgets comparable?
    - Does the result remain similar across seeds?
    """),

    code(r"""
    fig, axes = plt.subplots(1, 2, figsize=(12, 4))
    for model_name, history in {
        "matched MLP": mlp_history,
        "CNN original": cnn_plain_history,
        "CNN augmented": cnn_augmented_history,
    }.items():
        axes[0].plot(history["epoch"], history["training_loss"], label=model_name)
        axes[1].plot(history["epoch"], history["validation_loss"], label=model_name)
    axes[0].set_title("Training loss")
    axes[1].set_title("Validation loss")
    for axis in axes:
        axis.set_xlabel("epoch")
        axis.set_ylabel("cross-entropy")
        axis.legend()
    plt.tight_layout()
    plt.show()
    """),

    md(r"""
    ## 19 · Inspect learned filters without assigning imaginary meanings

    The first CNN layer contains eight learned `3×3` filters. We can plot them and their
    feature maps for a validation image.

    A bright or dark pattern shows numerical response, not a guaranteed human concept.
    A filter may resemble an edge detector, stroke fragment, or something harder to
    name. Interpretation needs evidence across many inputs and interventions; one
    attractive picture is not proof of what the model “understands.”
    """),

    code(r"""
    inspection_model = cnn_augmented_model if selected_augmentation else cnn_plain_model
    inspection_model.eval()
    inspection_image = validation_image_tensor[:1]

    with torch.inference_mode():
        first_feature_maps = torch.relu(inspection_model.first_convolution(inspection_image))[0].numpy()
    learned_filters = inspection_model.first_convolution.weight.detach().cpu().numpy()[:, 0]

    fig, axes = plt.subplots(2, 8, figsize=(16, 4))
    for filter_index in range(8):
        axes[0, filter_index].imshow(learned_filters[filter_index], cmap="coolwarm")
        axes[0, filter_index].set_title(f"filter {filter_index}")
        axes[1, filter_index].imshow(first_feature_maps[filter_index], cmap="viridis")
        axes[1, filter_index].set_title("feature map")
        axes[0, filter_index].axis("off")
        axes[1, filter_index].axis("off")
    plt.suptitle("Learned first-layer filters and one validation response")
    plt.tight_layout()
    plt.show()
    """),

    md(r"""
    ## 20 · Freeze the recipe and measure seed sensitivity

    We freeze the augmentation choice and repeat both neural architectures for seeds
    `11`, `22`, and `33`. Validation loss closest to each architecture's seed median
    chooses its representative checkpoint. This avoids presenting the luckiest seed as
    normal behavior.

    Test remains sealed throughout these runs.
    """),

    code(r"""
    declared_seeds = [11, 22, 33]
    seed_rows = []
    seed_artifacts = {"MLP": {}, "CNN": {}}

    for declared_seed in declared_seeds:
        # Architecture is the only difference inside each seed; the CNN augmentation is frozen.
        for architecture_name, model_factory, augmentation_choice in [
            ("MLP", MatchedDigitMLP, False),
            ("CNN", DigitCNN, selected_augmentation),
        ]:
            seed_model, seed_history, seed_metadata = train_image_classifier(
                model_factory,
                seed=declared_seed,
                augment=augmentation_choice,
            )
            validation_metrics = evaluate_classifier(seed_model, validation_image_tensor, validation_label_tensor)
            seed_rows.append(
                {
                    "architecture": architecture_name,
                    "seed": declared_seed,
                    "validation_loss": validation_metrics["loss"],
                    "validation_accuracy": validation_metrics["accuracy"],
                    "best_epoch": seed_metadata["best_epoch"],
                }
            )
            seed_artifacts[architecture_name][declared_seed] = seed_model

    seed_report = pd.DataFrame(seed_rows)
    representative_seeds = {}
    for architecture_name in ["MLP", "CNN"]:
        architecture_rows = seed_report[seed_report["architecture"] == architecture_name]
        median_loss = architecture_rows["validation_loss"].median()
        representative_index = (architecture_rows["validation_loss"] - median_loss).abs().idxmin()
        representative_seeds[architecture_name] = int(seed_report.loc[representative_index, "seed"])

    seed_summary = seed_report.groupby("architecture")["validation_loss"].agg(["mean", "std", "min", "max"])
    print(seed_report.round(4).to_string(index=False))
    print("validation-loss summary:")
    print(seed_summary.round(5).to_string())
    print("representative seeds:", representative_seeds)
    print("test status: sealed")
    """),

    md(r"""
    ## 21 · Mini-project: compare frozen models on sealed test once

    **Goal:** decide whether the CNN's spatial bias earns its complexity on tiny digits.

    **Dataset columns:** one `8×8` grayscale image and one digit target.

    **Completed workflow:**

    1. stratified train, validation, and sealed-test split;
    2. majority and logistic baselines;
    3. roughly matched MLP and CNN parameter budgets;
    4. identical neural optimizer and checkpoint contract;
    5. train-only shift augmentation ablation;
    6. three frozen seeds;
    7. representative median-validation checkpoints;
    8. one test evaluation below.

    **Metrics:** accuracy, macro F1, multiclass log loss, and parameter count.

    **Interpretation rule:** the CNN is not required to win. Report what happened and
    name image resolution, architecture choice, training budget, and augmentation as
    plausible limitations rather than inventing a universal conclusion.
    """),

    code(r"""
    majority_test_probabilities = majority_baseline.predict_proba(
        sealed_test_images.reshape(len(sealed_test_images), -1)
    )

    # Test is consumed only in this final reporting cell; nothing below can retrain a model.
    majority_test_predictions = majority_baseline.predict(
        sealed_test_images.reshape(len(sealed_test_images), -1)
    )
    logistic_test_probabilities = logistic_baseline.predict_proba(
        sealed_test_images.reshape(len(sealed_test_images), -1)
    )
    logistic_test_predictions = logistic_test_probabilities.argmax(axis=1)

    final_rows = [
        {
            "candidate": "majority",
            "test_loss": log_loss(sealed_test_labels, majority_test_probabilities, labels=list(range(10))),
            "test_accuracy": accuracy_score(sealed_test_labels, majority_test_predictions),
            "test_macro_f1": f1_score(sealed_test_labels, majority_test_predictions, average="macro"),
            "parameters": 0,
        },
        {
            "candidate": "logistic",
            "test_loss": log_loss(sealed_test_labels, logistic_test_probabilities, labels=list(range(10))),
            "test_accuracy": accuracy_score(sealed_test_labels, logistic_test_predictions),
            "test_macro_f1": f1_score(sealed_test_labels, logistic_test_predictions, average="macro"),
            "parameters": logistic_baseline.coef_.size + logistic_baseline.intercept_.size,
        },
    ]

    for architecture_name, parameter_count in [("MLP", mlp_parameter_count), ("CNN", cnn_parameter_count)]:
        representative_model = seed_artifacts[architecture_name][representative_seeds[architecture_name]]
        test_metrics = evaluate_classifier(representative_model, sealed_test_image_tensor, sealed_test_label_tensor)
        final_rows.append(
            {
                "candidate": architecture_name,
                "test_loss": test_metrics["loss"],
                "test_accuracy": test_metrics["accuracy"],
                "test_macro_f1": test_metrics["macro_f1"],
                "parameters": parameter_count,
            }
        )

    final_test_report = pd.DataFrame(final_rows)
    print(final_test_report.round(4).to_string(index=False))
    print("final report only; test did not change architecture, augmentation, seed policy, or checkpoint")

    assert final_test_report["test_accuracy"].between(0, 1).all()
    """),

    md(r"""
    ## 22 · Transfer learning when local labels are limited

    For larger natural images, training from scratch is often wasteful. A pretrained
    backbone already contains reusable visual features.

    A careful workflow is:

    1. use the exact preprocessing expected by the pretrained weights;
    2. replace the task-specific classifier head;
    3. freeze the backbone and train the new head;
    4. validate;
    5. optionally unfreeze later blocks with a smaller learning rate;
    6. compare against the frozen baseline under the same split;
    7. preserve checkpoint and model-version compatibility.

    Do not assume ImageNet pretraining transfers equally to medical scans, satellite
    bands, industrial sensors, or drawings. Domain mismatch is an empirical question.
    Frozen BatchNorm layers and their running statistics also need deliberate mode handling.
    """),

    code(r"""
    transfer_demo = DigitCNN()

    # Treat both convolution layers as a pretend pretrained backbone.
    for parameter in list(transfer_demo.first_convolution.parameters()) + list(transfer_demo.second_convolution.parameters()):
        parameter.requires_grad = False

    frozen_trainable_count = sum(parameter.numel() for parameter in transfer_demo.parameters() if parameter.requires_grad)
    total_transfer_count = sum(parameter.numel() for parameter in transfer_demo.parameters())
    print("trainable while backbone is frozen:", frozen_trainable_count)
    print("total parameters:", total_transfer_count)

    # Fine-tuning later re-enables selected backbone parameters with a smaller learning rate.
    for parameter in transfer_demo.second_convolution.parameters():
        parameter.requires_grad = True
    partially_unfrozen_count = sum(parameter.numel() for parameter in transfer_demo.parameters() if parameter.requires_grad)
    print("trainable after unfreezing the last convolution:", partially_unfrozen_count)
    assert frozen_trainable_count < partially_unfrozen_count < total_transfer_count
    """),

    md(r"""
    ## 23 · Common mistakes and how to inspect them

    - **Mixing NHWC and NCHW.** Print every boundary shape before blaming the layer.
    - **Forgetting the channel sum.** One output filter spans all channels in its group.
    - **Omitting bias from parameter counts.** Add one bias per output channel when enabled.
    - **Calling cross-correlation a flipped convolution in code.** State the library convention.
    - **Using the wrong output formula.** Include dilation and floor behavior.
    - **Claiming weight count means constant compute.** Spatial positions still cost work and memory.
    - **Calling pooling full translation invariance.** Test actual transformations and boundaries.
    - **Gradient-checking max-pool at a tie.** The chosen subgradient convention matters.
    - **Claiming a filter is an edge detector from one plot.** Inspect behavior across inputs.
    - **Applying unsafe augmentation.** Confirm that the transformation preserves the label.
    - **Augmenting validation or test.** Keep selection and final evidence on the declared distribution.
    - **Using a sequential, unstratified split.** Preserve class balance and randomness explicitly.
    - **Reporting the final epoch.** Restore the best validation checkpoint.
    - **Opening test for architecture choice.** Freeze every decision first.
    - **Assuming a CNN must beat an MLP on every image dataset.** Inductive bias is useful evidence, not a guarantee.
    """),

    md(r"""
    ## 24 · Practice, solutions, and mastery checkpoint

    ### Worked example

    Input `(N,3,32,32)`, 16 filters, kernel `5×5`, padding 2, stride 1:

    - output spatial size is `32×32`;
    - output tensor is `(N,16,32,32)`;
    - parameters including bias are $16(3\times5\times5+1)=1216$.

    ### Guided practice

    1. Calculate output size for input 28, kernel 3, padding 1, stride 2, dilation 1.
    2. Count parameters for 32 input channels, 64 output channels, `3×3`, with bias.
    3. Trace the receptive field through `conv3/s1 → pool2/s2 → conv3/s1`.

    ### Independent practice

    4. Extend scratch convolution to asymmetric height/width stride and padding.
    5. Finite-difference check convolution with two input and two output channels.
    6. Implement average-pooling forward and backward passes.
    7. Replace the CNN flattening head with GAP and compare parameters plus validation evidence.
    8. Add two-pixel shifts to the augmentation candidates without touching test.

    ### Challenge

    Build a depthwise-separable block from `groups=C_in` plus `1×1` convolution. Match
    its output shapes against a standard convolution, count parameters and multiply-adds,
    compare measured CPU latency carefully, and explain why theoretical savings may not
    translate proportionally to wall-clock speed.

    ### Solution and scoring rubric

    1. Output size is $\lfloor(28+2-3)/2\rfloor+1=14$.
    2. Parameters are $64(32\times3\times3+1)=18{,}496$.
    3. Receptive fields are 3 after convolution, 4 after pooling, and 8 after the second convolution.

    Award two points for each self-check below. Full-credit code must pass numerical
    gradient checks and preserve validation/test boundaries.

    ### Self-check

    1. Why is PyTorch image order NCHW important?
    2. What does one convolution output channel calculate?
    3. Why does weight sharing reduce parameters but not eliminate spatial compute?
    4. How do stride, padding, and dilation affect output shape?
    5. Why do shared kernel gradients accumulate across locations?
    6. How does max-pool route its backward gradient?
    7. How do equivariance, tolerance, and invariance differ?
    8. How is receptive field calculated?
    9. What do depthwise and `1×1` convolution each do?
    10. Why must augmentation be label-preserving and train-only?
    11. What does a fair MLP/CNN comparison require?

    ### Readiness threshold

    Score at least **18/22**, pass one complete multi-channel convolution gradient
    check, and defend the MLP/CNN conclusion without test-driven tuning.
    """),

    md(r"""
    ## Ready to move on?

    ### Quick check

    Explain how `(N,3,32,32)` becomes `(N,16,16,16)` through a `3×3`, stride-2,
    padding-1 convolution. State the weight shape, parameter count with bias, and one
    condition under which exact translation equivariance fails.

    ### Teach it back

    Start with one patch dot product. Expand it to channels and batches, trace its three
    backward gradients, explain pooling switches and receptive-field growth, then defend
    a fair CNN-versus-MLP experiment from split through sealed test.

    ### Memory aid

    **A convolution learns one local channel-mixing pattern, shares it across space,
    accumulates every reused gradient path, and earns its value only under fair evidence.**

    ### Next dependency

    Spatial inductive bias and shared weights  
    → useful before sequence models  
    → because later architectures reuse the idea of sharing transformations across
    positions while changing how information moves between those positions.
    """),
]


build("04_deep_learning/05_convolutional_neural_networks.ipynb", cells)
