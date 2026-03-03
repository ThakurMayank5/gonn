[![MIT License](https://img.shields.io/badge/License-MIT-green.svg)](https://choosealicense.com/licenses/mit/) ![GitHub contributors](https://img.shields.io/github/contributors/ThakurMayank5/gonn) ![GitHub Release Date](https://img.shields.io/github/release-date/ThakurMayank5/gonn) ![GitHub language count](https://img.shields.io/github/languages/count/ThakurMayank5/gonn) ![GitHub Downloads (all assets, all releases)](https://img.shields.io/github/downloads/ThakurMayank5/gonn/total) ![GitHub Release](https://img.shields.io/github/v/release/ThakurMayank5/gonn) ![GitHub Tag](https://img.shields.io/github/v/tag/ThakurMayank5/gonn)

# gonn — Neural Networks in Pure Go

A fully-featured feedforward neural network framework built from scratch in Go. No ML libraries, no external dependencies — just pure Go.

---

## Features

- **Fully configurable architecture** — any depth, any width, per-layer settings
- **7 optimizers** — SGD, Momentum, EMA Momentum, Nesterov, RMSProp, Adam, AdamW
- **4 activation functions** — ReLU, Sigmoid, Tanh, Softmax
- **2 loss functions** — Mean Squared Error, Categorical Cross-Entropy (auto-selected)
- **4 weight initializers** — Xavier Uniform/Normal, Kaiming Uniform/Normal (per-layer)
- **Dropout regularization** — inverted dropout with per-layer configurable rates
- **Early stopping** — halts training on validation plateau, restores best weights
- **ReduceLROnPlateau** — automatically reduces learning rate when validation loss stalls
- **Weight decay** — L2 regularization (SGD/Momentum/RMSProp) and decoupled decay (AdamW)
- **Mini-batch training** — with per-epoch dataset shuffling and progress bar
- **Weight save/load** — binary serialization via `encoding/gob`
- **CSV data loader** — with header support, label one-hot encoding, and feature scaling
- **Dataset splitting** — with or without shuffling
- **Per-epoch validation** — loss reporting on a held-out set

---

## Installation

```bash
go get github.com/ThakurMayank5/gonn
```

---

## Quick Start

```go
import (
    activ "github.com/ThakurMayank5/gonn/activation"

    "github.com/ThakurMayank5/gonn/dataloader"
    "github.com/ThakurMayank5/gonn/dataset"

    nn    "github.com/ThakurMayank5/gonn/neuralnetwork"
)
```

### 1. Define a Model

```go
model := nn.Model{
    NeuralNetwork: nn.NeuralNetwork{
        InputLayer: nn.InputLayer{Neurons: 784},
        Layers: []nn.Layer{
            {
                Neurons:            256,
                ActivationFunction: activ.ReLU,
                Initialization:     nn.KaimingNormalInitializer,
                DropoutRate:        0.3,
            },
            {
                Neurons:            128,
                ActivationFunction: activ.ReLU,
                Initialization:     nn.KaimingNormalInitializer,
                DropoutRate:        0.3,
            },
            {
                Neurons:            64,
                ActivationFunction: activ.ReLU,
                Initialization:     nn.KaimingNormalInitializer,
                DropoutRate:        0.2,
            },
        },
        OutputLayer: nn.OutputLayer{
            Neurons:            10,
            ActivationFunction: activ.Softmax,
            Initialization:     nn.KaimingNormalInitializer,
        },
    },
    TrainingConfig: nn.TrainingConfig{
        Epochs:       30,
        LearningRate: 0.001,
        Optimizer:    nn.ADAMW,
        Beta1:        0.9,
        Beta2:        0.999,
        Epsilon:      1e-8,
        WeightDecay:  0.0001,
        LossFunction: "categorical_crossentropy",
        BatchSize:    64,

        // LR Scheduler
        ReduceOnPlateau: true,
        LRFactor:        0.5,
        LRPatience:      3,
        MinLR:           1e-6,

        // Early Stopping
        EarlyStopping:         true,
        EarlyStoppingPatience: 5,
    },
}
```

### 2. Initialize Weights

```go
err := model.InitializeWeights()
if err != nil {
    log.Fatal(err)
}
```

### 3. Load Data

```go
cfg := dataset.CSVConfig{
    HasHeader:      true,
    InputColumns:   pixelCols,      // []int of column indices
    HasLabelColumn: true,
    LabelColumn:    0,
    NumClasses:     10,
    Delimiter:      ',',
    Scaling:        dataset.MinMaxNormalize,
}

train, _ := dataloader.FromCSV("train.csv", cfg)
test, _  := dataloader.FromCSV("test.csv", cfg)
```

### 4. Train

```go
err = model.Fit(train, test)
```

`Fit` prints per-epoch progress bars, validation loss, LR reductions, and early stopping messages.

```bash

Starting training for 30 epochs with batch size 64 (938 batches per epoch)
Epoch 1/30
Progress: 938/938 (100.00%)[####################]
Validation Loss: 0.4259
Epoch 2/30
Progress: 938/938 (100.00%)[####################]
Validation Loss: 0.3834
Epoch 3/30
Progress: 938/938 (100.00%)[####################]
Validation Loss: 0.3790
Epoch 4/30
Progress: 938/938 (100.00%)[####################]
Validation Loss: 0.3517

```

### 5. Save & Load Weights

```go
model.SaveWeights("model.weights")

// Later — load into a fresh model with matching architecture
freshModel.LoadWeights("model.weights")
```

### 6. Evaluate & Predict

```go
accuracy, _ := model.Evaluate(test)   // prints loss & accuracy

output, _ := model.NeuralNetwork.Predict(inputVector)
// output is []float64 of length OutputLayer.Neurons
```

### 7. Print Architecture Summary

```go
model.NeuralNetwork.Summary()
```

---

## Optimizers

| Optimizer    | Constant          | Weight Decay    | Description                                          |
| ------------ | ----------------- | --------------- | ---------------------------------------------------- |
| SGD          | `nn.SGD`          | L2 on gradients | Vanilla stochastic gradient descent                  |
| Momentum     | `nn.MOMENTUM`     | L2 on gradients | Classical momentum (`v = β·v + grad`)                |
| EMA Momentum | `nn.EMA_MOMENTUM` | L2 on gradients | Exponential moving average momentum                  |
| Nesterov     | `nn.NESTEROV`     | L2 on gradients | Nesterov accelerated gradient (look-ahead)           |
| RMSProp      | `nn.RMSPROP`      | Decoupled       | Adaptive per-parameter LR via squared gradient cache |
| Adam         | `nn.ADAM`         | None            | Adaptive moment estimation                           |
| AdamW        | `nn.ADAMW`        | Decoupled       | Adam with decoupled weight decay (biases excluded)   |

**Default hyperparameters** (used when not explicitly set):

| Parameter | Default | Applies to                                |
| --------- | ------- | ----------------------------------------- |
| `Beta`    | 0.9     | Momentum, EMA Momentum, Nesterov, RMSProp |
| `Beta1`   | 0.9     | Adam, AdamW                               |
| `Beta2`   | 0.999   | Adam, AdamW                               |
| `Epsilon` | 1e-8    | Adam, AdamW                               |

---

## Activation Functions

| Constant        | Function        | Derivative                | Best for                      |
| --------------- | --------------- | ------------------------- | ----------------------------- |
| `activ.ReLU`    | `max(0, x)`     | `x > 0 → 1, else 0`       | Hidden layers                 |
| `activ.Sigmoid` | `1 / (1 + e⁻ˣ)` | `σ(1 − σ)`                | Binary output / hidden layers |
| `activ.Tanh`    | `tanh(x)`       | `1 − tanh²(x)`            | Hidden layers (zero-centered) |
| `activ.Softmax` | `eˣⁱ / Σeˣʲ`    | `pred − target` (with CE) | Multi-class output layer      |

---

## Weight Initializers

| Constant                       | Distribution | Formula                                | Recommended for |
| ------------------------------ | ------------ | -------------------------------------- | --------------- |
| `nn.XavierUniformInitializer`  | Uniform      | `U(−√(6/(fᵢₙ+fₒᵤₜ)), √(6/(fᵢₙ+fₒᵤₜ)))` | Sigmoid / Tanh  |
| `nn.XavierNormalInitializer`   | Normal       | `N(0, √(2/(fᵢₙ+fₒᵤₜ)))`                | Sigmoid / Tanh  |
| `nn.KaimingUniformInitializer` | Uniform      | `U(−√(6/fᵢₙ), √(6/fᵢₙ))`               | ReLU            |
| `nn.KaimingNormalInitializer`  | Normal       | `N(0, √(2/fᵢₙ))`                       | ReLU            |

Each layer can specify its own initializer. Biases are zero-initialized. Defaults: hidden layers → Xavier Normal, output layer → Kaiming Normal.

---

## Dropout

- Set `DropoutRate` (0.0–0.99) on any hidden `Layer`.
- Uses **inverted dropout**: active neurons are scaled by `1/(1−p)` during training, so inference requires no rescaling.
- Per-sample, per-layer binary masks (`DropoutMasks[batch][layer][neuron]`).
- Dropped neurons have their gradients zeroed during backpropagation.
- Dropout is **never applied** to the output layer or during inference (`Predict`).

```go
nn.Layer{
    Neurons:            128,
    ActivationFunction: activ.ReLU,
    Initialization:     nn.KaimingNormalInitializer,
    DropoutRate:        0.3, // drop 30% of neurons during training
}
```

---

## Early Stopping

Automatically halts training when validation loss stops improving, and **restores the best weights**.

| Parameter               | Type   | Description                                        |
| ----------------------- | ------ | -------------------------------------------------- |
| `EarlyStopping`         | `bool` | Enable early stopping                              |
| `EarlyStoppingPatience` | `int`  | Epochs to wait without improvement before stopping |

- On improvement: counter resets, best weights are deep-copied.
- On trigger: weights are restored to the best checkpoint, training ends.

```go
TrainingConfig: nn.TrainingConfig{
    EarlyStopping:         true,
    EarlyStoppingPatience: 5,
}
```

---

## ReduceLROnPlateau

Automatically reduces the learning rate when validation loss plateaus.

| Parameter         | Type      | Description                                     |
| ----------------- | --------- | ----------------------------------------------- |
| `ReduceOnPlateau` | `bool`    | Enable LR reduction                             |
| `LRFactor`        | `float64` | Multiplicative factor (e.g., 0.5 halves the LR) |
| `LRPatience`      | `int`     | Epochs to wait before reducing                  |
| `MinLR`           | `float64` | Floor for the learning rate                     |

- On improvement: patience counter resets.
- On plateau: `new_lr = lr × factor` (clamped to `MinLR`), counter resets.

```go
TrainingConfig: nn.TrainingConfig{
    ReduceOnPlateau: true,
    LRFactor:        0.5,
    LRPatience:      3,
    MinLR:           1e-6,
}
```

---

## Data Loading

### CSV Configuration

```go
cfg := dataset.CSVConfig{
    HasHeader:      true,
    InputColumns:   []int{1, 2, 3, 4},   // feature column indices
    HasLabelColumn: true,                  // enable one-hot encoding
    LabelColumn:    0,                     // column with class labels
    NumClasses:     10,                    // 0 = auto-detect
    Delimiter:      ',',
    Scaling:        dataset.MinMaxNormalize,
}

data, err := dataloader.FromCSV("data.csv", cfg)
```

### Scaling Methods

| Constant                    | Formula                   | Range         |
| --------------------------- | ------------------------- | ------------- |
| `dataset.NoScaling`         | No transformation         | Original      |
| `dataset.MinMaxNormalize`   | `(x − min) / (max − min)` | [0, 1]        |
| `dataset.ZScoreStandardize` | `(x − μ) / σ`             | mean=0, std=1 |

### Dataset Splitting

```go
train, test, err := dataset.SplitWithShuffle(data, 0.8)    // 80/20 shuffled split
train, test, err := dataset.SplitWithoutShuffle(data, 0.8)  // sequential split
```

---

## TrainingConfig Reference

| Field                   | Type           | Default | Description                                                                  |
| ----------------------- | -------------- | ------- | ---------------------------------------------------------------------------- |
| `Epochs`                | `int`          | —       | Number of training epochs                                                    |
| `LearningRate`          | `float64`      | —       | Initial learning rate                                                        |
| `Optimizer`             | `Optimizer`    | `SGD`   | Optimization algorithm                                                       |
| `LossFunction`          | `LossFunction` | Auto    | `"categorical_crossentropy"` or `"mse"` (auto-selected by output activation) |
| `BatchSize`             | `int`          | —       | Mini-batch size                                                              |
| `Beta`                  | `float64`      | 0.9     | Momentum factor (Momentum/EMA/Nesterov/RMSProp)                              |
| `Beta1`                 | `float64`      | 0.9     | First moment decay (Adam/AdamW)                                              |
| `Beta2`                 | `float64`      | 0.999   | Second moment decay (Adam/AdamW)                                             |
| `Epsilon`               | `float64`      | 1e-8    | Numerical stability (Adam/AdamW)                                             |
| `WeightDecay`           | `float64`      | 0       | L2 regularization / decoupled weight decay                                   |
| `ReduceOnPlateau`       | `bool`         | false   | Enable LR reduction on plateau                                               |
| `LRFactor`              | `float64`      | —       | LR multiplicative reduction factor                                           |
| `LRPatience`            | `int`          | —       | Epochs before LR reduction                                                   |
| `MinLR`                 | `float64`      | —       | Minimum learning rate floor                                                  |
| `EarlyStopping`         | `bool`         | false   | Enable early stopping                                                        |
| `EarlyStoppingPatience` | `int`          | —       | Epochs before early stop triggers                                            |

---

## Project Structure

```
gonn/
├── main.go                          # Fashion-MNIST example
├── go.mod
├── LICENSE
├── README.md
│
├── activation/
│   └── activations.go               # ReLU, Sigmoid, Tanh, Softmax
│
├── losses/
│   └── compute.go                   # MSE, Categorical Cross-Entropy
│
├── vectors/
│   └── vectors.go                   # Dot product and vector utilities
│
├── dataset/
│   ├── csv.go                       # CSVConfig, ScalingMethod constants
│   ├── dataset.go                   # Dataset struct
│   └── split.go                     # Train/test splitting (with/without shuffle)
│
├── dataloader/
│   └── loader.go                    # FromCSV loader with scaling & one-hot encoding
│
├── neuralnetwork/
│   ├── model.go                     # Core types: Model, Layer, NeuralNetwork, configs
│   ├── initializers.go              # Xavier & Kaiming weight initialization
│   ├── training.go                  # Fit loop, early stopping, LR scheduling
│   ├── batch.go                     # Mini-batch forward pass with dropout
│   ├── backpropogation.go           # Backpropagation & gradient computation
│   ├── optimizers.go                # All optimizer implementations
│   ├── predict.go                   # Single-sample inference
│   ├── evaluation.go                # Evaluate (loss + accuracy on dataset)
│   ├── validation.go                # ForwardPassBatch (loss computation)
│   ├── weights.go                   # Save/Load weights (gob encoding)
│   └── shuffler.go                  # Dataset shuffling utilities
│
├── gpuprocessing/
│   └── vectors.go                   # GPU vector ops (experimental)
│
└── cuda/
    └── dotproduct.cu                # CUDA kernel for dot product (experimental)
```

---

## Example: Fashion-MNIST Classification

The included `main.go` trains a network on the Fashion-MNIST dataset (60,000 training + 10,000 test samples, 28×28 grayscale images, 10 clothing categories).

| Layer  | Neurons | Activation | Initializer    | Dropout |
| ------ | ------- | ---------- | -------------- | ------- |
| Input  | 784     | —          | —              | —       |
| Hidden | 256     | ReLU       | Kaiming Normal | 0.3     |
| Hidden | 128     | ReLU       | Kaiming Normal | 0.3     |
| Hidden | 64      | ReLU       | Kaiming Normal | 0.2     |
| Output | 10      | Softmax    | Kaiming Normal | —       |

**Training config:** AdamW (β₁=0.9, β₂=0.999, ε=1e-8), weight decay 0.0001, batch size 64, LR 0.001 with ReduceOnPlateau (factor 0.5, patience 3, min 1e-6), early stopping (patience 5).

```bash
go run .
```

---

## License

MIT
