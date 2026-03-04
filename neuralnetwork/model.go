package neuralnetwork

import (
	"fmt"

	activation "github.com/ThakurMayank5/gonn/activation"
)

// Optimizer represents the optimization algorithm
type Optimizer string

// LossFunction represents the loss function
type LossFunction string

// Initialization represents weight initialization strategy
type Initialization string

const (
	SGD          Optimizer = "sgd"
	MOMENTUM     Optimizer = "momentum"
	EMA_MOMENTUM Optimizer = "ema_momentum"
	NESTEROV     Optimizer = "nesterov"
	// ADAGRAD      Optimizer = "adagrad"
	RMSPROP Optimizer = "rmsprop"
	ADAM    Optimizer = "adam"
	ADAMW   Optimizer = "adamw"
	// AMSGRAD      Optimizer = "amsgrad"
	// LAMB         Optimizer = "lamb"
)

const (
	XavierUniformInitializer  Initialization = "xavier_uniform"
	XavierNormalInitializer   Initialization = "xavier_normal"
	KaimingUniformInitializer Initialization = "kaiming_uniform"
	KaimingNormalInitializer  Initialization = "kaiming_normal"
)

// InputLayer represents the input layer configuration
type InputLayer struct {
	Neurons            int
	ActivationFunction activation.ActivationFunction
}

// OutputLayer represents the output layer configuration
type OutputLayer struct {
	Neurons            int
	ActivationFunction activation.ActivationFunction
	Initialization     Initialization
}

// Optimizer	Behavior
// SGD	 L2 to gradient
// Momentum	 L2 to gradient
// RMSProp	 decoupled
// Adam	No WeightDecay
// AdamW	Decoupled shrink

// TrainingConfig represents training hyperparameters
type TrainingConfig struct {
	Epochs          int
	LearningRate    float64
	Optimizer       Optimizer
	LossFunction    LossFunction
	BatchSize       int
	ValidationSplit float64
	Beta            float64 // For momentum and Adam optimizers

	// For Adam and AdamW
	Beta1   float64
	Beta2   float64
	Epsilon float64

	WeightDecay float64 // For AdamW and L2 regularization

	// LR Scheduler Parameters for ReduceLROnPlateau
	ReduceOnPlateau bool
	LRFactor        float64
	LRPatience      int
	MinLR           float64

	// Early Stopping Parameters
	EarlyStopping         bool
	EarlyStoppingPatience int
}

// ModelWeightsAndBiases stores the model parameters
type ModelWeightsAndBiases struct {
	Weights [][][]float64
	Biases  [][]float64
}

// Layer represents a hidden layer
type Layer struct {
	Neurons            int
	ActivationFunction activation.ActivationFunction
	Initialization     Initialization
	DropoutRate        float64
}

// computed gradients
type GradientBuffer struct {
	GradW [][][]float64
	GradB [][]float64
}

type OptimizerState struct {
	VelocitiesW [][][]float64 // For momentum-based optimizers
	VelocitiesB [][]float64

	CacheW [][][]float64 // For adaptive optimizers
	CacheB [][]float64

	Timestep int // For Adam optimizer
}

type TrainingState struct {
	BestValLoss          float64
	LRPatienceCounter    int
	EarlyStoppingCounter int

	BestWeights [][][]float64
	BestBiases  [][]float64
}

type InferenceState struct {
	InferenceBuffer     [][]float64
	ActivationFunctions []func(float64) float64
	// stored row major wise
	FlatWeights [][]float64 // contiguous weights per layer [neurons*inputSize], flat weights are cache friendly
	Biases      [][]float64 // cached reference to biases
	InputSizes  []int       // input dimension per layer
}

// NeuralNetwork represents the neural network architecture
type NeuralNetwork struct {
	InputLayer       InputLayer
	Layers           []Layer
	OutputLayer      OutputLayer
	WeightsAndBiases ModelWeightsAndBiases
	OptimizerState   *OptimizerState

	// Mask[batch_size][layer][neuron] = true if neuron is active, false if dropped out
	DropoutMasks [][][]bool // For storing dropout masks during training

	TrainingState *TrainingState
}

// Model represents the complete model with network and training config and inference state
type Model struct {
	NeuralNetwork  NeuralNetwork
	TrainingConfig TrainingConfig
	// Stores Buffers for inference to avoid repeated allocations
	InferenceState *InferenceState

	inferenceMode bool // If true, model is in inference mode
}

func (model *Model) initializeInferenceBuffer() {

	if model.InferenceState == nil {
		model.InferenceState = &InferenceState{}
	}

	weights := model.NeuralNetwork.WeightsAndBiases.Weights
	biases := model.NeuralNetwork.WeightsAndBiases.Biases
	numLayers := len(biases)

	inf := model.InferenceState
	inf.InferenceBuffer = make([][]float64, numLayers)
	inf.ActivationFunctions = make([]func(float64) float64, numLayers)
	inf.FlatWeights = make([][]float64, numLayers)
	inf.Biases = biases
	inf.InputSizes = make([]int, numLayers)

	for i := range numLayers {

		if i == numLayers-1 {
			inf.InferenceBuffer[i] = make([]float64, model.NeuralNetwork.OutputLayer.Neurons)

			// returns nil for Softmax
			inf.ActivationFunctions[i] = activation.GetActivationFunction(model.NeuralNetwork.OutputLayer.ActivationFunction)
		} else {
			inf.InferenceBuffer[i] = make([]float64, model.NeuralNetwork.Layers[i].Neurons)

			// returns nil for Softmax
			inf.ActivationFunctions[i] = activation.GetActivationFunction(model.NeuralNetwork.Layers[i].ActivationFunction)
		}

		// Flatten weights into a single contiguous array per layer for cache-friendly access during inference
		neurons := len(weights[i])
		if neurons > 0 {
			inputSize := len(weights[i][0])
			inf.InputSizes[i] = inputSize
			flat := make([]float64, neurons*inputSize)
			for j := range neurons {
				copy(flat[j*inputSize:(j+1)*inputSize], weights[i][j])
			}
			inf.FlatWeights[i] = flat
		}
	}

}

func (model *Model) SetInferenceMode(enabled bool) {
	model.inferenceMode = enabled

	if enabled {
		model.initializeInferenceBuffer()
	} else if model.InferenceState != nil {
		model.InferenceState.InferenceBuffer = nil
		model.InferenceState.FlatWeights = nil
	}

}

// AddLayer adds a hidden layer to the neural network
func (nn *NeuralNetwork) AddLayer(layer Layer) {
	nn.Layers = append(nn.Layers, layer)
}

// SetOutputLayer sets the output layer configuration
func (nn *NeuralNetwork) SetOutputLayer(layer OutputLayer) {
	nn.OutputLayer = layer
}

// SetInputLayer sets the input layer configuration
func (nn *NeuralNetwork) SetInputLayer(layer InputLayer) {
	nn.InputLayer = layer
}

// Summary prints the neural network architecture
func (nn *NeuralNetwork) Summary() {

	TotalLayers := len(nn.Layers) + 2 // Input and Output layers

	fmt.Println("Neural Network Summary:")
	fmt.Println("Total Layers:", TotalLayers)
	fmt.Println("Input Layer Neurons:", nn.InputLayer.Neurons)
	fmt.Println("Input Layer Activation Function:", nn.InputLayer.ActivationFunction)

	for i, layer := range nn.Layers {
		fmt.Println("Layer", i+1, "Neurons:", layer.Neurons)
		fmt.Println("Layer", i+1, "Activation Function:", layer.ActivationFunction)
	}

	fmt.Println("Output Layer Neurons:", nn.OutputLayer.Neurons)
	fmt.Println("Output Layer Activation Function:", nn.OutputLayer.ActivationFunction)

}

// InitializeWeights initializes the model weights and biases
func (model *Model) InitializeWeights() error {

	totalLayers := len(model.NeuralNetwork.Layers) + 2 // Input and Output layers

	totalTrainableLayers := totalLayers - 1 // Exclude input layer

	biases := make([][]float64, totalTrainableLayers)

	for i := range biases {
		if i == totalTrainableLayers-1 {

			biases[i] = make([]float64, model.NeuralNetwork.OutputLayer.Neurons)
			continue
		}
		biases[i] = make([]float64, model.NeuralNetwork.Layers[i].Neurons)
	}

	// Bias initialization
	for i := range biases {
		for j := range biases[i] {
			biases[i][j] = 0.0
		}
	}

	model.NeuralNetwork.WeightsAndBiases.Biases = biases

	// Allocate weight tensors
	allocateWeights(&model.NeuralNetwork)

	// Initialize each hidden layer with its own strategy
	for i, layer := range model.NeuralNetwork.Layers {
		init := layer.Initialization
		if init == "" {
			init = XavierNormalInitializer // default
		}
		initLayerWeights(&model.NeuralNetwork, i, init)
	}

	// Initialize output layer
	outputInit := model.NeuralNetwork.OutputLayer.Initialization
	if outputInit == "" {
		outputInit = KaimingNormalInitializer // default
	}
	initLayerWeights(&model.NeuralNetwork, len(model.NeuralNetwork.Layers), outputInit)

	return nil
}
