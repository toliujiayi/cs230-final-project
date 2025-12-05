# Design Document: Contrastive Safety Learning with Gated Trajectories

---

## 1. Executive Summary

### 1.1 Problem Statement
In the SimLingo model, safe trajectory generation relies on the model implicitly "switching" behavior based on safety context. While the language head correctly identifies unsafe situations (e.g., "Unsafe..."), the trajectory head often fails to switch from the *instruction path* to the *safe default path*, especially when instructions are ambiguous (e.g., "Drive to coordinate X").

### 1.2 Proposed Solution
We propose a dual-mechanism approach to enforce safety at both the **feature level** and **action level**:

1.  **Contrastive Safety Learning**: A supervised contrastive loss that forces the driving query features to be geometrically distinct for safe vs. unsafe situations.
2.  **Auxiliary Safety Gating**: An explicit gating mechanism that predicts a safety score from these features and modulates the trajectory head's input, forcing a "switch" in behavior.
3.  **Consistency Loss**: An additional loss that encourages consistency between the outputs of the language and trajectory heads, especially in ambiguous scenarios. This penalizes cases where the language head predicts "unsafe" or adversarial situations, but the trajectory output fails to change (i.e., still proposes an unsafe trajectory).

### 1.3 Key Benefits
-   **Discriminative Features**: Contrastive learning ensures the model *knows* the difference between safe and unsafe visually.
-   **Explicit Control**: The gate ensures this knowledge is *used* to change the trajectory output.
-   **Verifiable**: We can inspect both the embedding separation (t-SNE) and the gate activation values.

---

## 2. Technical Architecture

### 2.1 High-Level Diagram

```
                    [Image] + [Prompt Tokens] + [Driving Query Tokens]
                                        │
                                        ▼
                              ┌───────────────────┐
                              │    Transformer    │
                              │    (InternVL2)    │
                              └─────────┬─────────┘
                                        │
                  ┌─────────────────────┴─────────────────────┐
                  │                                           │
                  ▼                                           ▼
       ┌─────────────────────┐                     ┌─────────────────────┐
       │  Language Features  │                     │   Driving Features  │
       │  [B, Tokens, Dim]   │                     │  [B, Queries, Dim]  │
       └──────────┬──────────┘                     └──────────┬──────────┘
                  │                                           │
                  ▼                     ┌──────────────────────┼──────────────────────┐
       ┌─────────────────────┐         │                      │                      │
       │   Language Head     │         ▼                      ▼                      │
       │     (LM Head)       │  ┌─────────────────┐  ┌─────────────────┐             │
       └──────────┬──────────┘  │ Contrastive Head│  │   Safety Gate   │             │
                  │             │ (Training Only) │  │    (Sigmoid)    │             │
                  ▼             └────────┬────────┘  └────────┬────────┘             │
       ┌─────────────────────┐          │                     │                      │
       │    Text Output      │          ▼                     ▼                      │
       │  "Unsafe because    │  ┌─────────────────┐    Safety Prob (σ)               │
       │   it would lead     │  │   SupCon Loss   │           │                      │
       │   to collision..."  │  │ (Separate Safe/ │           │                      │
       └─────────────────────┘  │ Unsafe features)│           │                      │
                                └─────────────────┘           │                      │
                                                              ▼                      │
                                               ┌───────────────────────┐             │
                                               │  Feature Modulation   │◄────────────┘
                                               │ F' = F × (1 + (1-σ))  │  Raw Features
                                               └───────────┬───────────┘
                                                           │
                                                           ▼ Gated Features
                                               ┌───────────────────────┐
                                               │    Trajectory Head    │
                                               │    (Route + Speed)    │
                                               └───────────┬───────────┘
                                                           │
                                                           ▼
                                               ┌───────────────────────┐
                                               │      Waypoints        │
                                               │     [B, 20+10, 2]     │
                                               └───────────────────────┘
```

**Data Flow Summary:**
1. **Transformer** processes image + prompt tokens + driving query tokens together.
2. **Language Features** → Language Head → **Text Output** (e.g., "Unsafe because...").
3. **Driving Features** branch into three paths:
   - **Contrastive Head** → SupCon Loss (forces safe/unsafe separation, training only).
   - **Safety Gate** → Safety probability σ ∈ [0, 1].
   - **Raw Features** → Feature Modulation.
4. **Feature Modulation** scales raw features: F' = F × (1 + (1-σ) × mask).
5. **Trajectory Head** predicts waypoints from gated features.

### 2.2 Mechanism 1: Supervised Contrastive Learning

We apply **Supervised Contrastive Loss (SupCon)** on the driving features.
-   **Scope**: Only applied to samples in `<SAFETY>` mode.
-   **Label**: `safe_to_execute` (True/False).
-   **Goal**: Ensure $Sim(F_{unsafe}, F_{unsafe}) \gg Sim(F_{unsafe}, F_{safe})$.
-   **Temperature**: τ = 0.07 (default from SupCon paper).

### 2.3 Mechanism 2: Safety Gating

We introduce a learnable gate that explicitly modulates features based on predicted safety and the dreamer mode.

**Formula**:
$$p_{safe} = \sigma(MLP(Pool(F)))$$
$$F_{gated} = F \cdot (1 + (1 - p_{safe}) \cdot \mathbb{I}_{safety\_mode})$$

Where:
-   $F$ are the raw driving features.
-   $p_{safe}$ is the predicted probability that the situation is safe.
-   $\mathbb{I}_{safety\_mode}$ is 1 if in `<SAFETY>` mode, 0 if in `<INSTRUCTION_FOLLOWING>` mode.

**Behavior**:
-   If $p_{safe} \approx 1$ (Safe): $F_{gated} \approx F$ (no change).
-   If $p_{safe} \approx 0$ (Unsafe) AND Safety Mode: $F_{gated} \approx 2F$ (boosted features trigger safe path).
-   If Instruction Mode: $F_{gated} = F$ (no modulation regardless of safety).

### 2.4 Mechanism 3: Consistency Loss (Trajectory Regularization)

We add a loss term that explicitly penalizes the model if it outputs the instruction path when it should output the safe path.

$$L_{consist} = \max(0, 1.0 - ||\hat{W} - W_{instruction}||)$$

Where:
- $\hat{W}$ is the predicted trajectory.
- $W_{instruction}$ is the trajectory corresponding to the unsafe instruction.
- This loss is only applied when the situation is **UNSAFE** AND in **`<SAFETY>` mode**.

**Rationale**: It explicitly says "If unsafe, NOT ONLY should you be close to the safe path, but you MUST NOT look like the instruction path."

---

## 3. Implementation Details

### 3.1 New Modules in `DrivingAdaptor`

```python
# Contrastive Head: Projects features to embedding space
self.contrastive_head = nn.Sequential(
    nn.Linear(hidden_size, 256),  # hidden_size = 896 for InternVL2-1B
    nn.ReLU(),
    nn.Linear(256, 128),
)
# Note: L2 normalization applied via F.normalize() in forward pass

# Safety Gate: Predicts safety probability
self.safety_gate = nn.Sequential(
    nn.Linear(hidden_size, 64),
    nn.ReLU(),
    nn.Linear(64, 1),
    nn.Sigmoid()
)

# Contrastive Loss
self.supcon_loss = SupConLoss(temperature=0.07)
```

### 3.2 Data Pipeline Changes

Added to `DatasetOutput` and `DrivingLabel`:
- `safe_to_execute: bool` - Whether the instruction is safe
- `is_safety_mode: bool` - Whether in `<SAFETY>` or `<INSTRUCTION_FOLLOWING>` mode
- `instruction_path: Tensor` - The trajectory for the (possibly unsafe) instruction

### 3.3 Differential Learning Rates

Different learning rates for different model components:

| Component | LR Multiplier | Rationale |
|-----------|---------------|-----------|
| New heads (contrastive, gate) | 3.0× | Randomly initialized, needs faster learning |
| Pretrained heads (route, speed) | 1.0× | Already trained |
| Language model (LoRA) | 0.5× | LoRA provides regularization |
| Vision encoder | Frozen | Preserve perception capabilities |

---

## 4. Hyperparameters

### 4.1 Final Configuration

| Parameter | Value | Rationale |
|-----------|-------|-----------|
| **Base Learning Rate** | 3e-5 | Same as original training |
| **Batch Size** | 16 | Provides enough negatives for contrastive learning |
| **Epochs** | 5 | Multiple passes for new heads to converge |
| **SupCon Temperature** | 0.07 | Default from SupCon paper |

### 4.2 Loss Weights

| Loss | Weight | Rationale |
|------|--------|-----------|
| `route_loss` | 1.0 | Primary trajectory task |
| `speed_wps_loss` | 1.0 | Primary speed task |
| `language_loss` | 1.0 | Primary language task |
| `gate_loss` | 1.0 | Critical for safety prediction |
| `contrastive_loss` | 0.3 | Auxiliary - feature separation |
| `consistency_loss` | 0.3 | Auxiliary - repel unsafe paths |

### 4.3 Model Configuration

| Parameter | Value |
|-----------|-------|
| Vision Model | InternVL2-1B (frozen) |
| Language Model | InternVL2-1B with LoRA |
| LoRA rank (r) | 32 |
| LoRA alpha | 64 |
| LoRA dropout | 0.1 |

### 4.4 Training Data

| Setting | Value |
|---------|-------|
| Dataset | Dreamer only (driving/QA disabled) |
| Dreamer folder | `dreamer` (all modes) |
| Mode filter | None (includes crash, lane_change, faster, slower, stop, target_speed) |
| Safety flag | Enabled (50% `<SAFETY>`, 50% `<INSTRUCTION_FOLLOWING>`) |
| Train batches | 10,000 per epoch |

---

## 5. Evaluation Strategy

### 5.1 Level 1: Feature Quality (Contrastive Success)
-   **Metric**: Linear Probe Accuracy.
-   **Method**: Freeze the model, train a logistic regression on `raw_features` to predict `safe_to_execute`.
-   **Check**: t-SNE visualization of `raw_features` showing separation.

### 5.2 Level 2: Mechanism Health (Gate Success)
-   **Metric**: Gate Accuracy.
-   **Method**: Check `safety_prob` predictions against ground truth.
-   **Sanity Check**: Does $p_{safe}$ correlate with `safe_to_execute`?

### 5.3 Level 3: Outcome Quality (Trajectory Success)
-   **Metric**: Crash Mode Trajectory Error (ADE/FDE).
-   **Method**: Calculate distance between predicted path and "safe default path" for UNSAFE samples.
-   **Goal**: Lower error than baseline (Baseline often outputs instruction path).

---

## 6. Risk Mitigation

| Risk | Mitigation |
|------|------------|
| **Gate Collapse**: Gate always outputs 0.5 or 1.0 | Strong supervision (`gate_loss` weight = 1.0). |
| **Feature Degradation**: Contrastive loss distorts regression features | Use projection head; apply contrastive on `raw_features`, regression on `gated_features`. |
| **Conflicting Gradients**: Regression wants features X, Contrastive wants Y | Lower contrastive weight (0.3) to not dominate. |
| **New heads underfitting** | Higher LR (3×) for new heads. |
| **Pretrained weights overfitting** | Lower LR (0.5×) for LoRA; freeze vision encoder. |

---

## 7. Files Modified

| File | Changes |
|------|---------|
| `simlingo_training/utils/custom_types.py` | Added `safe_to_execute`, `is_safety_mode`, `instruction_path` to `DatasetOutput` and `DrivingLabel` |
| `simlingo_training/dataloader/dataset_dreamer.py` | Populate new fields from dreamer data |
| `simlingo_training/dataloader/datamodule.py` | Collate new fields into batches |
| `simlingo_training/models/losses.py` | New file with `SupConLoss` implementation |
| `simlingo_training/models/adaptors/adaptors.py` | Added `ContrastiveHead`, `SafetyGate`, gating logic, new losses |
| `simlingo_training/models/driving.py` | Differential LR optimizer, loss weight support |
| `simlingo_training/config.py` | Added `use_contrastive`, `use_consistency`, `loss_weights` |
| `simlingo_training/config/experiment/contrastive.yaml` | New experiment config |

---

## 8. Commands

### Train
```bash
python simlingo_training/train.py experiment=contrastive \
    checkpoint=/path/to/outputs/simlingo/checkpoints/epoch=013.ckpt
```

### Evaluate
```bash
python simlingo_training/eval.py
# (Set load_path in eval.py to point to trained checkpoint)
```

---
