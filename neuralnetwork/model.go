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

// TrainingConfig represents training hyperparameters
type TrainingConfig struct {
	Epochs          int
	LearningRate    float64
	Optimizer       Optimizer
	LossFunction    LossFunction
	BatchSize       int
	ValidationSplit float64
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
}

// NeuralNetwork represents the neural network architecture
type NeuralNetwork struct {
	InputLayer       InputLayer
	Layers           []Layer
	OutputLayer      OutputLayer
	WeightsAndBiases ModelWeightsAndBiases
}

// Model represents the complete model with network and training config
type Model struct {
	NeuralNetwork  NeuralNetwork
	TrainingConfig TrainingConfig
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
