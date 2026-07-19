"""Build DL-04: controlled, observable, and reproducible neural training."""

import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
from nbbuild import build, code, md  # noqa: E402


cells = [
    md(r"""
    # DL-04 · Stable Neural Training

    **Prerequisites:** DL-01, DL-02, and DL-03  
    **Estimated total mastery time:** 12–15 hours, including practice  
    **Next lesson:** DL-05 · Convolutional Neural Networks

    A correct network can still fail to learn. Its activations may collapse, gradients
    may explode, updates may be too timid, or training loss may improve while validation
    performance gets worse.

    This lesson turns training into a controlled experiment. We will measure what flows
    forward, what flows backward, how large each update is, and whether validation
    evidence supports a change. We will alter one control at a time and keep test data
    sealed until one final configuration is frozen.

    ### Scope boundary

    DL-03 explained where gradients come from. Here we use PyTorch autograd rather than
    deriving it again. Architecture-specific tools such as convolution, residual blocks,
    recurrent clipping, and transformer LayerNorm return in later lessons.
    """),

    md(r"""
    ## 1 · What you will be able to do

    By the end, you will be able to:

    - distinguish optimization failure, underfitting, and overfitting;
    - establish a reproducible train/validation/test experiment;
    - monitor loss, accuracy, learning rate, activation statistics, gradient norm,
      parameter norm, and update-to-parameter ratio;
    - calculate Xavier and He initialization scales;
    - match initialization to the following activation;
    - diagnose learning rates that are too small or too large;
    - calculate SGD, momentum, RMSProp, Adam, and AdamW updates;
    - compare optimizer recipes under equal epoch budgets;
    - calculate and apply global gradient-norm clipping;
    - distinguish input scaling, BatchNorm, and LayerNorm;
    - use BatchNorm and dropout correctly in training and evaluation modes;
    - separate L2 regularization from AdamW's decoupled weight decay;
    - implement validation scheduling, early stopping, and checkpoint restoration;
    - compare controlled ablations across several seeds;
    - evaluate a frozen model once on sealed test data.

    ### Dependency path

    ```mermaid
    flowchart LR
        A[Fixed data contract] --> B[Observable baseline]
        B --> C[Initialization]
        C --> D[Learning rate]
        D --> E[Optimizer recipe]
        E --> F[Clipping and normalization]
        F --> G[Regularization]
        G --> H[Early stopping]
        H --> I[Multi-seed evidence]
        I --> J[One sealed test]
    ```

    Verified gradient flow  
    → required before training diagnostics  
    → because gradient norms and update sizes are meaningless until the backward path is understood.

    Stable neural training  
    → required before CNNs, RNNs, and transformers  
    → because a new architecture cannot be judged fairly through a broken training procedure.
    """),

    md(r"""
    ## 2 · Three failures can look like “the model is bad”

    Imagine preparing for an examination:

    - **Optimization failure:** your study method is broken—you never absorb the material.
    - **Underfitting:** the plan is too simple or too short—even your practice score stays low.
    - **Overfitting:** you memorize practice answers—practice rises while the real exam stalls.

    | Evidence | Likely diagnosis | First controlled question |
    |---|---|---|
    | training and validation losses both high | underfitting or optimization problem | does training loss decrease normally? |
    | training loss falls, validation loss rises | overfitting | does one regularizer improve validation? |
    | loss jumps, gradients or updates grow | instability | is learning rate or scale excessive? |
    | activations become nearly constant | signal collapse or saturation | does activation-aware initialization help? |
    | one run works and another fails | seed sensitivity | is the conclusion stable across declared seeds? |

    Change one thing, predict what should happen, and state what result would reject the
    diagnosis. Adding dropout, clipping, normalization, and a new optimizer together may
    improve a score, but it teaches us almost nothing about the cause.
    """),

    md(r"""
    ## 3 · Establish the experiment before touching the model

    We use scikit-learn's handwritten digits: 1,797 rows, 64 pixel features, and one
    target from 0 through 9. The dataset is local, small, and nonlinear enough for a
    multilayer perceptron.

    Our contract is:

    1. seal 20% as test data;
    2. use 20% as validation and 60% as training;
    3. fit standardization on training rows only;
    4. keep architecture width and epoch budget fixed within each comparison;
    5. select controls and checkpoints using validation loss only;
    6. run declared seeds after choosing the recipe;
    7. open test once for the frozen representative checkpoint.

    Validation is reusable development evidence, so repeated choices can still overfit
    it. We limit the comparison list in advance and keep the final test independent.
    """),

    code(r"""
    import copy
    import math
    import random
    from dataclasses import dataclass

    import matplotlib.pyplot as plt
    import numpy as np
    import pandas as pd
    import torch
    from sklearn.datasets import load_digits
    from sklearn.model_selection import train_test_split
    from sklearn.preprocessing import StandardScaler
    from torch import nn
    from torch.utils.data import DataLoader, TensorDataset

    DEVICE = torch.device("cpu")


    def set_reproducible(seed):
        '''Seed the random sources used by this CPU lesson.'''
        random.seed(seed)
        np.random.seed(seed)
        torch.manual_seed(seed)
        torch.use_deterministic_algorithms(True)


    digit_data = load_digits()
    all_features = digit_data.data.astype(np.float32)
    all_targets = digit_data.target.astype(np.int64)

    development_features, sealed_test_features, development_targets, sealed_test_targets = train_test_split(
        all_features,
        all_targets,
        test_size=0.20,
        stratify=all_targets,
        random_state=42,
    )
    train_features_raw, validation_features_raw, train_targets, validation_targets = train_test_split(
        development_features,
        development_targets,
        test_size=0.25,
        stratify=development_targets,
        random_state=42,
    )

    # Fit preprocessing on training rows, then reuse those frozen statistics everywhere.
    feature_scaler = StandardScaler()
    train_features = feature_scaler.fit_transform(train_features_raw).astype(np.float32)
    validation_features = feature_scaler.transform(validation_features_raw).astype(np.float32)
    sealed_test_features_scaled = feature_scaler.transform(sealed_test_features).astype(np.float32)

    train_feature_tensor = torch.tensor(train_features)
    train_target_tensor = torch.tensor(train_targets)
    validation_feature_tensor = torch.tensor(validation_features)
    validation_target_tensor = torch.tensor(validation_targets)
    sealed_test_feature_tensor = torch.tensor(sealed_test_features_scaled)
    sealed_test_target_tensor = torch.tensor(sealed_test_targets)

    split_report = pd.DataFrame(
        [
            {"split": "train", "rows": len(train_targets), "used_for": "fit parameters and preprocessing"},
            {"split": "validation", "rows": len(validation_targets), "used_for": "select controls and checkpoint"},
            {"split": "sealed test", "rows": len(sealed_test_targets), "used_for": "one final report"},
        ]
    )
    print(split_report.to_string(index=False))
    print("test status: sealed")

    assert len(train_targets) + len(validation_targets) + len(sealed_test_targets) == len(all_targets)
    """),

    md(r"""
    ## 4 · Build a model whose controls are explicit

    Every hidden block uses:

    ```text
    Linear → optional normalization → ReLU → optional dropout
    ```

    The final linear layer produces logits, so it is not followed by ReLU. Its
    initialization should not blindly copy the hidden ReLU rule.

    - `normalization="batch"` normalizes each hidden feature using batch statistics.
    - `normalization="layer"` normalizes the hidden features within each row.
    - `dropout_probability=0` disables dropout without changing the architecture.

    Keeping these choices explicit makes an ablation auditable.
    """),

    code(r"""
    class DigitMLP(nn.Module):
        def __init__(self, hidden_width=64, normalization="none", dropout_probability=0.0):
            super().__init__()
            # Keep width fixed so normalization and dropout remain the only ablated controls.
            self.first_linear = nn.Linear(64, hidden_width)
            self.first_normalization = self._normalization(normalization, hidden_width)
            self.second_linear = nn.Linear(hidden_width, hidden_width)
            self.second_normalization = self._normalization(normalization, hidden_width)
            self.dropout = nn.Dropout(dropout_probability)
            self.output_linear = nn.Linear(hidden_width, 10)

        @staticmethod
        def _normalization(normalization, width):
            if normalization == "batch":
                return nn.BatchNorm1d(width)
            if normalization == "layer":
                return nn.LayerNorm(width)
            if normalization == "none":
                return nn.Identity()
            raise ValueError(f"unknown normalization: {normalization}")

        def forward(self, inputs):
            hidden = torch.relu(self.first_normalization(self.first_linear(inputs)))
            hidden = self.dropout(hidden)
            hidden = torch.relu(self.second_normalization(self.second_linear(hidden)))
            hidden = self.dropout(hidden)
            return self.output_linear(hidden)


    def initialize_model(model, scheme="he"):
        '''Initialize hidden ReLU layers and the linear logits layer deliberately.'''
        for layer_name, layer in model.named_modules():
            if not isinstance(layer, nn.Linear):
                continue
            if scheme == "small":
                nn.init.normal_(layer.weight, mean=0.0, std=0.01)
            elif scheme == "large":
                nn.init.normal_(layer.weight, mean=0.0, std=1.0)
            elif scheme == "zero":
                nn.init.zeros_(layer.weight)
            elif "output_linear" in layer_name:
                nn.init.xavier_normal_(layer.weight)
            else:
                nn.init.kaiming_normal_(layer.weight, nonlinearity="relu")
            nn.init.zeros_(layer.bias)
    """),

    md(r"""
    ## 5 · Decide what to measure before training

    Loss alone does not tell us why training behaves badly.

    ### Global gradient norm

    For parameter-gradient tensors $g_1,\ldots,g_K$:

    $$
    \lVert g\rVert_2=
    \sqrt{\sum_{k=1}^{K}\sum_j g_{k,j}^2}
    $$

    **Symbols:** $K$ is the number of parameter tensors and $j$ visits every scalar in
    one tensor.

    ### Update-to-parameter ratio

    $$
    r=\frac{\lVert\Delta\theta\rVert_2}
    {\max(10^{-12},\lVert\theta\rVert_2)}
    $$

    Here $\theta$ is the parameter vector before an epoch and $\Delta\theta$ is its
    change. A tiny ratio can explain slow learning; a huge ratio can signal destructive
    steps. There is no universal ideal value—the trend and task context matter.

    We will also record training and validation loss, accuracy, learning rate, selected
    epoch, and whether non-finite values appeared.
    """),

    md(r"""
    ## 6 · A reusable diagnostic trainer

    The trainer below is longer because it makes hidden training decisions visible.
    Read it in five blocks:

    1. deterministic data loading;
    2. model, optimizer, and optional scheduler construction;
    3. one training epoch with gradient measurement and optional clipping;
    4. validation without graph recording;
    5. independent best-checkpoint copying and restoration.

    Test tensors are intentionally absent from this function.
    """),

    code(r"""
    @dataclass(frozen=True)
    class TrainingConfig:
        learning_rate: float = 0.03
        optimizer_name: str = "momentum"
        momentum: float = 0.9
        weight_decay: float = 0.0
        normalization: str = "none"
        dropout_probability: float = 0.0
        initialization: str = "he"
        batch_size: int = 128
        maximum_epochs: int = 40
        clipping_threshold: float | None = None
        scheduler_patience: int | None = None
        early_stopping_patience: int | None = None
        minimum_improvement: float = 1e-4


    def make_training_loader(batch_size, seed):
        generator = torch.Generator().manual_seed(seed)
        dataset = TensorDataset(train_feature_tensor, train_target_tensor)
        return DataLoader(dataset, batch_size=batch_size, shuffle=True, generator=generator)


    def make_optimizer(model, config):
        if config.optimizer_name == "sgd":
            return torch.optim.SGD(model.parameters(), lr=config.learning_rate, weight_decay=config.weight_decay)
        if config.optimizer_name == "momentum":
            return torch.optim.SGD(
                model.parameters(),
                lr=config.learning_rate,
                momentum=config.momentum,
                weight_decay=config.weight_decay,
            )
        if config.optimizer_name == "adam":
            return torch.optim.Adam(model.parameters(), lr=config.learning_rate, weight_decay=config.weight_decay)
        if config.optimizer_name == "adamw":
            return torch.optim.AdamW(model.parameters(), lr=config.learning_rate, weight_decay=config.weight_decay)
        raise ValueError(f"unknown optimizer: {config.optimizer_name}")


    def global_norm(tensors):
        squared_sum = sum(
            float(torch.sum(tensor.detach() ** 2))
            for tensor in tensors
            if tensor is not None
        )
        return math.sqrt(squared_sum)


    def evaluate_model(model, features, targets):
        model.eval()
        with torch.inference_mode():
            logits = model(features.to(DEVICE))
            loss = nn.functional.cross_entropy(logits, targets.to(DEVICE)).item()
            accuracy = (logits.argmax(dim=1) == targets.to(DEVICE)).float().mean().item()
        return loss, accuracy


    def train_diagnostic_model(config, seed=42):
        set_reproducible(seed)
        model = DigitMLP(
            normalization=config.normalization,
            dropout_probability=config.dropout_probability,
        ).to(DEVICE)
        initialize_model(model, config.initialization)
        optimizer = make_optimizer(model, config)
        scheduler = None
        if config.scheduler_patience is not None:
            scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
                optimizer,
                patience=config.scheduler_patience,
                factor=0.5,
            )

        training_loader = make_training_loader(config.batch_size, seed)
        history_rows = []
        best_validation_loss = float("inf")
        best_epoch = None
        best_state = None
        epochs_without_improvement = 0
        status = "completed budget"

        for epoch in range(config.maximum_epochs):
            model.train()
            parameter_values_before = [parameter.detach().clone() for parameter in model.parameters()]
            running_loss = 0.0
            correct_predictions = 0
            seen_rows = 0
            batch_gradient_norms = []

            for batch_features, batch_targets in training_loader:
                optimizer.zero_grad(set_to_none=True)
                logits = model(batch_features.to(DEVICE))
                batch_loss = nn.functional.cross_entropy(logits, batch_targets.to(DEVICE))

                if not torch.isfinite(batch_loss):
                    status = "non-finite loss"
                    break

                batch_loss.backward()
                gradients = [parameter.grad for parameter in model.parameters()]
                gradient_norm_before_clip = global_norm(gradients)
                batch_gradient_norms.append(gradient_norm_before_clip)

                if config.clipping_threshold is not None:
                    nn.utils.clip_grad_norm_(model.parameters(), config.clipping_threshold)
                optimizer.step()

                running_loss += batch_loss.item() * len(batch_targets)
                correct_predictions += (logits.argmax(dim=1).cpu() == batch_targets).sum().item()
                seen_rows += len(batch_targets)

            if status != "completed budget":
                break

            parameter_values_after = [parameter.detach() for parameter in model.parameters()]
            parameter_norm = global_norm(parameter_values_before)
            update_norm = global_norm(
                [after - before for before, after in zip(parameter_values_before, parameter_values_after)]
            )
            update_ratio = update_norm / max(1e-12, parameter_norm)
            validation_loss, validation_accuracy = evaluate_model(
                model,
                validation_feature_tensor,
                validation_target_tensor,
            )
            current_learning_rate = optimizer.param_groups[0]["lr"]

            history_rows.append(
                {
                    "epoch": epoch,
                    "training_loss": running_loss / seen_rows,
                    "training_accuracy": correct_predictions / seen_rows,
                    "validation_loss": validation_loss,
                    "validation_accuracy": validation_accuracy,
                    "gradient_norm": float(np.mean(batch_gradient_norms)),
                    "parameter_norm": parameter_norm,
                    "update_ratio": update_ratio,
                    "learning_rate": current_learning_rate,
                }
            )

            if validation_loss < best_validation_loss - config.minimum_improvement:
                best_validation_loss = validation_loss
                best_epoch = epoch
                best_state = copy.deepcopy(model.state_dict())
                epochs_without_improvement = 0
            else:
                epochs_without_improvement += 1

            # Plateau scheduling consumes validation evidence after the epoch is recorded.
            if scheduler is not None:
                scheduler.step(validation_loss)

            if (
                config.early_stopping_patience is not None
                and epochs_without_improvement >= config.early_stopping_patience
            ):
                status = "early stopped"
                break

        if best_state is None:
            return model, pd.DataFrame(history_rows), {"status": status, "best_epoch": None}

        # Restore the independent validation-selected checkpoint, not the final weights.
        model.load_state_dict(best_state)
        metadata = {
            "status": status,
            "best_epoch": best_epoch,
            "best_validation_loss": best_validation_loss,
            "best_state": best_state,
            "seed": seed,
        }
        return model, pd.DataFrame(history_rows), metadata
    """),

    md(r"""
    ## 7 · Initialization controls the first signal, not the final answer

    Random weights serve two jobs:

    - break symmetry so hidden units can learn different features;
    - keep forward and backward magnitudes in a useful range at the start.

    For fan-in $d$:

    **Xavier normal**, commonly paired with tanh or roughly linear activations:

    $$
    W_{ij}\sim\mathcal N\left(0,\frac{1}{d}\right)
    $$

    **He normal**, commonly paired with ReLU:

    $$
    W_{ij}\sim\mathcal N\left(0,\frac{2}{d}\right)
    $$

    These second arguments are variances. Standard deviations are $\sqrt{1/d}$ and
    $\sqrt{2/d}$. With $d=64$, He standard deviation is:

    $$
    \sqrt{\frac{2}{64}}\approx0.1768
    $$

    He initialization approximately preserves the **second moment** through an idealized
    ReLU stack under distribution assumptions. It does not promise an exact sample
    variance, successful optimization, or good generalization.
    """),

    code(r"""
    def activation_trace(initialization, activation_name, depth=12, width=64, seed=7):
        set_reproducible(seed)
        activations = torch.randn(512, width)
        rows = []

        for layer_index in range(depth):
            # Recreate each layer independently while carrying its activations forward.
            weights = torch.empty(width, width)
            if initialization == "small":
                nn.init.normal_(weights, std=0.01)
            elif initialization == "large":
                nn.init.normal_(weights, std=1.0)
            elif initialization == "xavier":
                nn.init.xavier_normal_(weights)
            elif initialization == "he":
                nn.init.kaiming_normal_(weights, nonlinearity="relu")
            else:
                raise ValueError(initialization)

            preactivation = activations @ weights
            activations = torch.tanh(preactivation) if activation_name == "tanh" else torch.relu(preactivation)
            rows.append(
                {
                    "layer": layer_index + 1,
                    "standard_deviation": activations.std().item(),
                    "zero_fraction": (activations == 0).float().mean().item(),
                }
            )
        return pd.DataFrame(rows)


    initialization_recipes = {
        "small + ReLU": ("small", "relu"),
        "large + ReLU": ("large", "relu"),
        "Xavier + tanh": ("xavier", "tanh"),
        "He + ReLU": ("he", "relu"),
    }

    fig, axes = plt.subplots(1, 2, figsize=(12, 4))
    for recipe_name, (initialization_name, activation_name) in initialization_recipes.items():
        recipe_trace = activation_trace(initialization_name, activation_name)
        axes[0].plot(recipe_trace["layer"], recipe_trace["standard_deviation"], marker="o", label=recipe_name)
        axes[1].plot(recipe_trace["layer"], recipe_trace["zero_fraction"], marker="o", label=recipe_name)

    axes[0].set_yscale("log")
    axes[0].set_title("Activation scale across depth")
    axes[0].set_ylabel("activation standard deviation")
    axes[1].set_title("Exact-zero fraction across depth")
    axes[1].set_ylabel("fraction equal to zero")
    for axis in axes:
        axis.set_xlabel("layer")
        axis.legend()
    plt.tight_layout()
    plt.show()
    """),

    md(r"""
    Small weights collapse signal; very large weights can explode it; zero hidden
    weights preserve symmetry. Xavier+tanh and He+ReLU are matched recipes, not rival
    formulas to rank without an activation context.

    Initialization is diagnosed at step zero. If training later becomes unstable, also
    inspect learning rate, data scale, normalization, and gradient flow.
    """),

    md(r"""
    ## 8 · Learning rate controls step size

    Plain gradient descent updates a parameter vector by:

    $$
    \theta_{t+1}=\theta_t-\eta g_t
    $$

    **Symbols:** $t$ is the update number; $\theta_t$ is the current parameter vector;
    $g_t$ is its gradient; and $\eta$ is learning rate.

    For scalar $\theta=2$, gradient $g=3$, and $\eta=0.1$:

    $$
    \theta_{new}=2-(0.1)(3)=1.7
    $$

    Too small can look like underfitting because the epoch budget ends before useful
    progress. Too large may oscillate, increase loss, or produce non-finite values.
    The experiment below changes only learning rate.
    """),

    code(r"""
    learning_rate_histories = {}
    learning_rate_rows = []

    for candidate_learning_rate in [0.0001, 0.03, 3.0]:
        # Only learning rate changes; seed, model, SGD, data, and epochs stay fixed.
        candidate_config = TrainingConfig(
            learning_rate=candidate_learning_rate,
            optimizer_name="sgd",
            maximum_epochs=25,
        )
        _, candidate_history, candidate_metadata = train_diagnostic_model(candidate_config, seed=42)
        learning_rate_histories[candidate_learning_rate] = candidate_history
        learning_rate_rows.append(
            {
                "learning_rate": candidate_learning_rate,
                "status": candidate_metadata["status"],
                "best_validation_loss": candidate_metadata.get("best_validation_loss", np.nan),
                "last_update_ratio": candidate_history["update_ratio"].iloc[-1] if len(candidate_history) else np.nan,
                "last_parameter_norm": candidate_history["parameter_norm"].iloc[-1] if len(candidate_history) else np.nan,
            }
        )

    learning_rate_report = pd.DataFrame(learning_rate_rows)
    print(learning_rate_report.round(5).to_string(index=False))

    fig, axes = plt.subplots(1, 2, figsize=(12, 4))
    for learning_rate, history in learning_rate_histories.items():
        axes[0].plot(history["epoch"], history["training_loss"], label=f"lr={learning_rate}")
        axes[1].plot(history["epoch"], history["update_ratio"], label=f"lr={learning_rate}")
    axes[0].set_title("Training loss under one-variable LR changes")
    axes[0].set_ylabel("cross-entropy")
    axes[1].set_title("Update-to-parameter ratio")
    axes[1].set_yscale("log")
    for axis in axes:
        axis.set_xlabel("epoch")
        axis.legend()
    plt.tight_layout()
    plt.show()
    """),

    md(r"""
    The smallest candidate barely changes the parameters within the budget. The middle
    candidate learns steadily. The largest candidate produces destructive parameter
    scale and poor validation loss even if floating-point values remain finite. A run
    does not need to reach `NaN` before it is unstable or unusable.
    """),

    md(r"""
    ## 9 · Optimizers remember different summaries of past gradients

    ### Momentum

    $$
    v_t=\beta v_{t-1}+g_t
    $$

    $$
    \theta_{t+1}=\theta_t-\eta v_t
    $$

    $v_t$ is accumulated direction and $\beta$ controls memory. Momentum can smooth
    zig-zagging and accelerate persistent directions.

    ### RMSProp

    $$
    s_t=\rho s_{t-1}+(1-\rho)g_t^2
    $$

    $$
    \theta_{t+1}=\theta_t-
    \eta\frac{g_t}{\sqrt{s_t}+\epsilon}
    $$

    $s_t$ tracks recent squared gradients; $\rho$ is its memory; and $\epsilon$ avoids
    division by zero. Coordinates with persistently large gradients receive smaller
    effective steps.

    Adam combines momentum-like first moments and RMSProp-like second moments, with
    bias correction early in training:

    $$
    m_t=\beta_1m_{t-1}+(1-\beta_1)g_t
    $$

    $$
    v_t=\beta_2v_{t-1}+(1-\beta_2)g_t^2
    $$

    $$
    \hat m_t=\frac{m_t}{1-\beta_1^t},\qquad
    \hat v_t=\frac{v_t}{1-\beta_2^t}
    $$

    $$
    \theta_{t+1}=\theta_t-
    \eta\frac{\hat m_t}{\sqrt{\hat v_t}+\epsilon}
    $$

    $m_t$ is the first-moment estimate, $v_t$ is the second-moment estimate, and hats
    indicate bias-corrected estimates. AdamW additionally applies weight decay as a
    separate shrinkage step.
    """),

    code(r"""
    manual_gradients = [2.0, -1.0]
    manual_learning_rate = 0.1
    manual_beta = 0.9
    manual_parameter = 3.0
    manual_velocity = 0.0
    manual_rows = []

    for update_index, current_gradient in enumerate(manual_gradients, start=1):
        manual_velocity = manual_beta * manual_velocity + current_gradient
        manual_parameter = manual_parameter - manual_learning_rate * manual_velocity
        manual_rows.append(
            {
                "update": update_index,
                "gradient": current_gradient,
                "velocity": manual_velocity,
                "parameter": manual_parameter,
            }
        )

    print(pd.DataFrame(manual_rows).to_string(index=False))
    assert np.isclose(manual_parameter, 2.72)

    # On Adam's first step, bias correction recovers the observed gradient moments.
    adam_gradient = 2.0
    adam_beta_one = 0.9
    adam_beta_two = 0.999
    adam_learning_rate = 0.1
    adam_first_moment = (1 - adam_beta_one) * adam_gradient
    adam_second_moment = (1 - adam_beta_two) * adam_gradient ** 2
    corrected_first_moment = adam_first_moment / (1 - adam_beta_one)
    corrected_second_moment = adam_second_moment / (1 - adam_beta_two)
    adam_parameter = 3.0 - adam_learning_rate * corrected_first_moment / math.sqrt(corrected_second_moment)
    adamw_parameter = (1 - adam_learning_rate * 0.01) * 3.0 - 0.1

    print("Adam first-step parameter:", round(adam_parameter, 4))
    print("AdamW first-step parameter with decay 0.01:", round(adamw_parameter, 4))
    assert np.isclose(adam_parameter, 2.9)
    assert np.isclose(adamw_parameter, 2.897)
    """),

    md(r"""
    ### Compare optimizer recipes fairly

    One learning rate is not equally scaled for every optimizer, so we compare
    **predeclared recipes**, not claim a pure optimizer-only causal effect. Each recipe
    receives the same architecture, initialization, data, seed, and epoch budget.

    Use SGD when simplicity, explicit control, or a carefully tuned large-data recipe
    matters. Momentum is a strong default for many vision tasks. AdamW is often a
    forgiving starting point for transformers and sparse or differently scaled
    gradients. Validation evidence—not optimizer reputation—decides here.
    """),

    code(r"""
    optimizer_recipes = {
        "SGD": TrainingConfig(learning_rate=0.03, optimizer_name="sgd", maximum_epochs=35),
        "momentum": TrainingConfig(learning_rate=0.03, optimizer_name="momentum", maximum_epochs=35),
        "AdamW": TrainingConfig(learning_rate=0.002, optimizer_name="adamw", maximum_epochs=35),
    }
    optimizer_histories = {}
    optimizer_rows = []

    for recipe_name, recipe_config in optimizer_recipes.items():
        # Each declared recipe receives the same model, data, seed, and epoch budget.
        _, recipe_history, recipe_metadata = train_diagnostic_model(recipe_config, seed=42)
        optimizer_histories[recipe_name] = recipe_history
        optimizer_rows.append(
            {
                "recipe": recipe_name,
                "best_epoch": recipe_metadata["best_epoch"],
                "validation_loss": recipe_metadata["best_validation_loss"],
                "validation_accuracy": recipe_history.loc[
                    recipe_history["epoch"] == recipe_metadata["best_epoch"], "validation_accuracy"
                ].iloc[0],
            }
        )

    optimizer_report = pd.DataFrame(optimizer_rows).sort_values("validation_loss")
    selected_optimizer_recipe = optimizer_report.iloc[0]["recipe"]
    print(optimizer_report.round(4).to_string(index=False))
    print("validation-selected recipe:", selected_optimizer_recipe)

    fig, axis = plt.subplots(figsize=(8, 4))
    for recipe_name, history in optimizer_histories.items():
        axis.plot(history["epoch"], history["validation_loss"], label=recipe_name)
    axis.set_xlabel("epoch")
    axis.set_ylabel("validation cross-entropy")
    axis.set_title("Equal-budget optimizer recipe comparison")
    axis.legend()
    plt.tight_layout()
    plt.show()
    """),

    md(r"""
    ## 10 · Gradient clipping is an emergency brake, not steering

    Global norm clipping with threshold $c$ uses:

    $$
    g_{clipped}=g\min\left(1,\frac{c}{\lVert g\rVert_2}\right)
    $$

    For $g=[3,4]$, its norm is $5$. With $c=2$, scale is $2/5=0.4$ and:

    $$
    g_{clipped}=[1.2,1.6]
    $$

    Its new norm is exactly $2$. Clipping preserves direction while limiting magnitude.
    It is valuable for occasional spikes, especially in recurrent or very deep models.
    If nearly every step clips, inspect learning rate, initialization, loss scaling, and
    the architecture instead of hiding the cause.
    """),

    code(r"""
    manual_gradient = torch.tensor([3.0, 4.0])
    clipping_threshold = 2.0
    original_norm = torch.linalg.vector_norm(manual_gradient)
    clipping_scale = min(1.0, clipping_threshold / original_norm.item())
    clipped_gradient = manual_gradient * clipping_scale

    clipping_parameter = nn.Parameter(torch.tensor([0.0, 0.0]))
    clipping_parameter.grad = manual_gradient.clone()
    returned_preclip_norm = nn.utils.clip_grad_norm_([clipping_parameter], clipping_threshold)

    print("original norm:", original_norm.item())
    print("manual clipped gradient:", clipped_gradient.numpy())
    print("PyTorch clipped gradient:", clipping_parameter.grad.numpy())
    print("PyTorch returned pre-clip norm:", returned_preclip_norm.item())

    assert np.allclose(clipped_gradient.numpy(), [1.2, 1.6])
    assert np.isclose(torch.linalg.vector_norm(clipping_parameter.grad).item(), clipping_threshold)
    """),

    md(r"""
    ## 11 · Input scaling, BatchNorm, and LayerNorm solve different problems

    | Method | Statistics come from | Normalizes | Train/eval difference? |
    |---|---|---|---:|
    | input standardization | training dataset | each input feature | no |
    | BatchNorm | current batch during training; running estimates at evaluation | each hidden feature across batch rows | yes |
    | LayerNorm | each individual row | hidden features within that row | no running statistics |

    For a hidden batch $H\in\mathbb R^{B\times d}$, BatchNorm feature $j$ uses batch
    mean $\mu_j$ and variance $\sigma_j^2$:

    $$
    \hat H_{ij}=\frac{H_{ij}-\mu_j}{\sqrt{\sigma_j^2+\epsilon}}
    $$

    LayerNorm instead calculates mean and variance across the $d$ hidden features of
    each row $i$. Both learn an elementwise scale and shift after normalization.

    BatchNorm can become noisy with tiny or nonrepresentative batches and depends on
    correct `train()` and `eval()` modes. LayerNorm is independent of other batch rows,
    which suits variable batches and transformer token representations.
    """),

    code(r"""
    normalization_example = torch.tensor(
        [[1.0, 10.0, -2.0], [3.0, 14.0, 0.0], [5.0, 18.0, 2.0], [7.0, 22.0, 4.0]]
    )

    # Disable learned affine terms so the displayed values show normalization itself.
    batch_normalization = nn.BatchNorm1d(3, affine=False)
    layer_normalization = nn.LayerNorm(3, elementwise_affine=False)

    batch_normalization.train()
    batch_normalized = batch_normalization(normalization_example)
    layer_normalized = layer_normalization(normalization_example)

    print("BatchNorm feature means:", batch_normalized.mean(dim=0).numpy().round(5))
    print("LayerNorm row means:", layer_normalized.mean(dim=1).numpy().round(5))
    print("BatchNorm running mean after training call:", batch_normalization.running_mean.numpy().round(3))

    batch_normalization.eval()
    evaluation_output = batch_normalization(normalization_example[:1])
    print("BatchNorm evaluation uses stored running statistics:", evaluation_output.numpy().round(3))

    assert np.allclose(batch_normalized.mean(dim=0).numpy(), 0.0, atol=1e-6)
    assert np.allclose(layer_normalized.mean(dim=1).numpy(), 0.0, atol=1e-6)
    """),

    md(r"""
    ## 12 · Regularization changes what “fit well” means

    Regularization deliberately makes training harder when that trade improves unseen
    performance.

    ### L2 penalty

    Add squared weights to the data loss:

    $$
    J(\theta)=L_{data}(\theta)+\frac{\lambda}{2}\lVert\theta\rVert_2^2
    $$

    The gradient gains $\lambda\theta$. In plain SGD this is equivalent to multiplicative
    shrinkage plus the data-gradient step.

    ### Decoupled weight decay

    AdamW separates shrinkage from the adaptive gradient transformation:

    $$
    \theta_{t+1}=(1-\eta\lambda)\theta_t-\eta\,\operatorname{AdamUpdate}(g_t)
    $$

    In adaptive optimizers, adding an L2 term to the gradient is generally not equivalent
    to this decoupled decay.

    ### Dropout

    During training, inverted dropout keeps each activation with probability $1-p$ and
    scales kept values by $1/(1-p)$. This preserves expected activation value. During
    evaluation, dropout is disabled. It is not automatically beneficial; it can cause
    underfitting or duplicate other regularization.
    """),

    code(r"""
    dropout_layer = nn.Dropout(p=0.5)
    repeated_input = torch.ones(12)

    dropout_layer.train()
    set_reproducible(5)
    first_training_output = dropout_layer(repeated_input)
    second_training_output = dropout_layer(repeated_input)

    dropout_layer.eval()
    evaluation_dropout_output = dropout_layer(repeated_input)

    print("training output 1:", first_training_output.numpy())
    print("training output 2:", second_training_output.numpy())
    print("evaluation output:", evaluation_dropout_output.numpy())
    print("kept training values are scaled to:", 1 / (1 - dropout_layer.p))

    assert not torch.equal(first_training_output, second_training_output)
    assert torch.equal(evaluation_dropout_output, repeated_input)
    """),

    md(r"""
    ## 13 · Run one-variable normalization and dropout ablations

    We start from the validation-selected optimizer recipe, then compare normalization
    while keeping every other declared control fixed. After selecting normalization, we
    compare dropout probabilities `0` and `0.25`.

    This sequential search is intentionally small. It does not prove the controls have
    no interactions; it prevents an unlimited validation search from becoming disguised
    test-set tuning.
    """),

    code(r"""
    selected_base_config = optimizer_recipes[selected_optimizer_recipe]
    normalization_rows = []

    for normalization_name in ["none", "batch", "layer"]:
        # Carry the optimizer decision forward and change only normalization.
        normalization_config = TrainingConfig(
            learning_rate=selected_base_config.learning_rate,
            optimizer_name=selected_base_config.optimizer_name,
            normalization=normalization_name,
            maximum_epochs=35,
        )
        _, normalization_history, normalization_metadata = train_diagnostic_model(normalization_config, seed=42)
        normalization_rows.append(
            {
                "normalization": normalization_name,
                "validation_loss": normalization_metadata["best_validation_loss"],
                "best_epoch": normalization_metadata["best_epoch"],
            }
        )

    normalization_report = pd.DataFrame(normalization_rows).sort_values("validation_loss")
    selected_normalization = normalization_report.iloc[0]["normalization"]
    print(normalization_report.round(4).to_string(index=False))
    print("validation-selected normalization:", selected_normalization)

    dropout_rows = []
    dropout_runs = {}
    for dropout_probability in [0.0, 0.25]:
        # Add the same mild decay to both candidates so dropout is the only difference.
        dropout_config = TrainingConfig(
            learning_rate=selected_base_config.learning_rate,
            optimizer_name=selected_base_config.optimizer_name,
            normalization=selected_normalization,
            dropout_probability=dropout_probability,
            weight_decay=1e-4,
            maximum_epochs=50,
            scheduler_patience=3,
            early_stopping_patience=8,
        )
        dropout_model, dropout_history, dropout_metadata = train_diagnostic_model(dropout_config, seed=42)
        dropout_runs[dropout_probability] = (dropout_model, dropout_history, dropout_metadata, dropout_config)
        dropout_rows.append(
            {
                "dropout": dropout_probability,
                "validation_loss": dropout_metadata["best_validation_loss"],
                "best_epoch": dropout_metadata["best_epoch"],
                "status": dropout_metadata["status"],
            }
        )

    dropout_report = pd.DataFrame(dropout_rows).sort_values("validation_loss")
    selected_dropout = float(dropout_report.iloc[0]["dropout"])
    selected_final_config = dropout_runs[selected_dropout][3]
    print(dropout_report.round(4).to_string(index=False))
    print("validation-selected dropout:", selected_dropout)
    print("test status: still sealed")
    """),

    md(r"""
    ## 14 · Read real curves before prescribing a fix

    The selected run recorded checkpoint and diagnostic evidence. Common patterns:

    - **Underfit:** training and validation losses remain high and close. Increase useful
      capacity or training budget only after confirming optimization is healthy.
    - **Overfit:** training improves while validation worsens. Test one regularizer,
      more data, or lower capacity.
    - **Unstable:** loss, gradient norm, or update ratio spikes. Reduce learning rate or
      correct scale before treating clipping as a permanent cure.
    - **Candidate convergence:** both losses improve and validation eventually flattens.
      Restore the best validation checkpoint rather than the last epoch.

    Accuracy may remain unchanged while cross-entropy improves because probability
    confidence changes. Use the selection metric declared before training.
    """),

    code(r"""
    selected_model, selected_history, selected_metadata, _ = dropout_runs[selected_dropout]

    # Put selection, gradient scale, and update scale beside each other for diagnosis.
    fig, axes = plt.subplots(1, 3, figsize=(15, 4))
    axes[0].plot(selected_history["epoch"], selected_history["training_loss"], label="training")
    axes[0].plot(selected_history["epoch"], selected_history["validation_loss"], label="validation")
    axes[0].axvline(selected_metadata["best_epoch"], color="black", linestyle="--", label="restored epoch")
    axes[0].set_title("Loss and selected checkpoint")
    axes[0].legend()

    axes[1].plot(selected_history["epoch"], selected_history["gradient_norm"], label="gradient norm")
    axes[1].plot(selected_history["epoch"], selected_history["parameter_norm"], label="parameter norm")
    axes[1].set_yscale("log")
    axes[1].set_title("Backward and parameter scale")
    axes[1].legend()

    axes[2].plot(selected_history["epoch"], selected_history["update_ratio"], label="update ratio")
    axes[2].plot(selected_history["epoch"], selected_history["learning_rate"], label="learning rate")
    axes[2].set_yscale("log")
    axes[2].set_title("Step size and schedule")
    axes[2].legend()

    for axis in axes:
        axis.set_xlabel("epoch")
    plt.tight_layout()
    plt.show()

    print("training ended:", selected_metadata["status"])
    print("last epoch:", int(selected_history["epoch"].iloc[-1]))
    print("restored validation epoch:", selected_metadata["best_epoch"])
    assert selected_metadata["best_epoch"] <= int(selected_history["epoch"].iloc[-1])
    """),

    md(r"""
    ## 15 · One run is an anecdote; declared seeds show sensitivity

    Random initialization, dropout masks, and minibatch order can change results. A
    single successful seed does not prove a robust recipe.

    We now freeze the selected configuration and run three seeds. We report mean,
    standard deviation, minimum, and maximum validation loss. We choose the checkpoint
    with validation loss closest to the seed median—not the luckiest seed—for the final
    sealed-test report.

    Exact bitwise reproduction may depend on device, library version, and deterministic
    kernel availability. A serious run records code version, data fingerprint, split,
    seed, framework version, device, configuration, selected epoch, and checkpoint hash.
    """),

    code(r"""
    declared_seeds = [11, 22, 33]
    seed_rows = []
    seed_artifacts = {}

    for declared_seed in declared_seeds:
        # Configuration is frozen; only declared randomness changes across runs.
        seed_model, seed_history, seed_metadata = train_diagnostic_model(
            selected_final_config,
            seed=declared_seed,
        )
        best_row = seed_history.loc[seed_history["epoch"] == seed_metadata["best_epoch"]].iloc[0]
        seed_rows.append(
            {
                "seed": declared_seed,
                "validation_loss": seed_metadata["best_validation_loss"],
                "validation_accuracy": best_row["validation_accuracy"],
                "best_epoch": seed_metadata["best_epoch"],
            }
        )
        seed_artifacts[declared_seed] = (seed_model, seed_history, seed_metadata)

    seed_report = pd.DataFrame(seed_rows)
    validation_loss_median = seed_report["validation_loss"].median()
    representative_index = (seed_report["validation_loss"] - validation_loss_median).abs().idxmin()
    representative_seed = int(seed_report.loc[representative_index, "seed"])
    frozen_model = seed_artifacts[representative_seed][0]

    seed_summary = seed_report["validation_loss"].agg(["mean", "std", "min", "max"])
    print(seed_report.round(4).to_string(index=False))
    print("validation-loss summary:")
    print(seed_summary.round(5).to_string())
    print("representative median seed:", representative_seed)
    print("test status: still sealed")
    """),

    md(r"""
    ## 16 · Mini-project: open the sealed test exactly once

    **Goal:** demonstrate an end-to-end training decision without test-driven tuning.

    **Dataset columns:** 64 standardized digit pixels and one class label from 0 to 9.

    **Completed workflow:**

    1. fixed stratified splits before preprocessing;
    2. train-only feature standardization;
    3. activation-aware initialization;
    4. learning-rate diagnosis;
    5. equal-budget optimizer recipe comparison;
    6. controlled normalization and dropout ablations;
    7. scheduling, early stopping, and restored validation checkpoint;
    8. three declared seeds;
    9. representative median-seed selection using validation evidence;
    10. one frozen test evaluation below.

    **Evaluation criteria:** test is not required to beat validation or reach an
    arbitrary score. Success means correct data boundaries, finite diagnostics,
    independently copied checkpoints, transparent selection, and no retuning after test.
    """),

    code(r"""
    final_test_loss, final_test_accuracy = evaluate_model(
        frozen_model,
        sealed_test_feature_tensor,
        sealed_test_target_tensor,
    )

    # This report consumes test once and contains no branch that can alter the artifact.
    representative_validation_row = seed_report.loc[seed_report["seed"] == representative_seed].iloc[0]

    final_report = pd.DataFrame(
        [
            {
                "artifact": "representative frozen checkpoint",
                "seed": representative_seed,
                "validation_loss": representative_validation_row["validation_loss"],
                "validation_accuracy": representative_validation_row["validation_accuracy"],
                "test_loss": final_test_loss,
                "test_accuracy": final_test_accuracy,
            }
        ]
    )
    print(final_report.round(4).to_string(index=False))
    print("final report only; test did not change the configuration or checkpoint")

    assert np.isfinite(final_test_loss)
    assert 0.0 <= final_test_accuracy <= 1.0
    """),

    md(r"""
    ## 17 · Common mistakes and what evidence to inspect

    - **Changing several controls together.** Run a one-variable ablation first.
    - **Using test to choose an epoch or optimizer.** Keep it sealed until the recipe is frozen.
    - **Reporting the last epoch.** Restore the independent best-validation state.
    - **Applying He initialization to every layer automatically.** Match the rule to what follows.
    - **Calling exact ReLU variance preservation a guarantee.** The derivation uses distribution assumptions.
    - **Treating clipping as a learning-rate fix.** Record how often and how strongly clipping activates.
    - **Forgetting `eval()`.** BatchNorm and dropout change behavior across modes.
    - **Confusing input standardization with hidden normalization.** Their statistics and purposes differ.
    - **Treating Adam with L2 as identical to AdamW.** Adaptive scaling breaks that equivalence.
    - **Comparing optimizers with hidden budget differences.** Declare learning-rate recipe, seed, and epochs.
    - **Calling one lucky run robust.** Repeat a frozen recipe across declared seeds.
    - **Ignoring non-finite values.** Stop and diagnose rather than silently continuing.
    - **Assuming lower training loss means a better model.** Selection uses held-out validation evidence.
    """),

    md(r"""
    ## 18 · Practice, solutions, and mastery checkpoint

    ### Worked example

    For fan-in 256, He standard deviation is:

    $$
    \sqrt{\frac{2}{256}}\approx0.0884
    $$

    For gradient $[6,8]$ and clipping threshold $5$, original norm is $10$, scale is
    $0.5$, clipped gradient is $[3,4]$, and clipped norm is $5$.

    ### Guided practice

    1. Calculate Xavier and He standard deviations for fan-in 100.
    2. Apply one momentum update with $v_{old}=2$, $g=3$, $\beta=0.9$, $\eta=0.1$,
       and $\theta=5$.
    3. Diagnose: training loss falls smoothly while validation loss rises after epoch 12.

    ### Independent practice

    4. Add the fraction of exactly zero hidden ReLU activations to the trainer.
    5. Add a clipping-frequency metric and explain what persistent clipping suggests.
    6. Compare batch sizes 32 and 256 under equal numbers of optimizer updates.
    7. Repeat the normalization ablation with a deliberately tiny batch and explain the change.

    ### Challenge

    Design a two-stage study that first diagnoses optimization and only then tests
    regularization. Predeclare candidates, selection metric, seed policy, stopping rule,
    and the exact point at which test may be opened. Explain why each boundary prevents
    a misleading conclusion.

    ### Solution and scoring rubric

    1. Xavier is $\sqrt{1/100}=0.1$; He is $\sqrt{2/100}\approx0.1414$.
    2. New velocity is $(0.9)(2)+3=4.8$; new parameter is $5-(0.1)(4.8)=4.52$.
    3. This is evidence of overfitting after epoch 12, assuming optimization diagnostics
       remain healthy. Restore the best validation checkpoint and test one controlled regularizer.

    Award two points for each of eleven self-check questions below. Full-credit code
    must preserve test sealing and compare equal declared budgets.

    ### Self-check

    1. How do optimization failure, underfitting, and overfitting differ?
    2. Why must initialization match the following activation?
    3. What does update-to-parameter ratio measure?
    4. Why is clipping not a cure for a bad learning rate?
    5. How do momentum and adaptive optimizers remember gradients differently?
    6. Why can Adam with L2 differ from AdamW?
    7. How do BatchNorm and LayerNorm choose their statistics?
    8. Why do BatchNorm and dropout require correct model mode?
    9. Which split selects the checkpoint?
    10. Why report multiple frozen seeds?
    11. When may sealed test data be opened?

    ### Readiness threshold

    Score at least **18/22**, complete one independent instrumentation task, and defend
    a controlled ablation without using test evidence for any choice.
    """),

    md(r"""
    ## Ready to move on?

    ### Quick check

    Given a run with falling training loss, rising validation loss, stable gradient
    norms, and moderate update ratios, name the likely diagnosis, one controlled change,
    and the evidence that would reject your proposal.

    ### Teach it back

    Explain the full loop without notes: initialize according to activation, measure
    forward and backward scale, choose a learning-rate/optimizer recipe using validation,
    test normalization or regularization one variable at a time, restore the best
    checkpoint, repeat the frozen recipe across seeds, and open test once.

    ### Memory aid

    **Measure first, change one control, select on validation, restore the best state,
    repeat across seeds, and test only after freezing.**

    ### Next dependency

    Stable, observable training  
    → required before convolutional networks  
    → because DL-05 should measure whether spatial weight sharing helps, not whether an
    unmonitored optimizer happened to succeed.
    """),
]


build("04_deep_learning/04_stable_neural_training.ipynb", cells)
