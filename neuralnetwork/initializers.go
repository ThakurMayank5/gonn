package neuralnetwork

import (
	"math"
	"math/rand"
)

// fanIn returns the number of inputs into layer i (i.e. the size of the previous layer).
func fanIn(nn *NeuralNetwork, layerIndex int) int {
	if layerIndex == 0 {
		return nn.InputLayer.Neurons
	}
	return nn.Layers[layerIndex-1].Neurons
}

// fanOut returns the number of neurons in layer i (size of the current layer).
func fanOut(nn *NeuralNetwork, layerIndex int) int {
	if layerIndex == len(nn.Layers) {
		return nn.OutputLayer.Neurons
	}
	return nn.Layers[layerIndex].Neurons
}

// allocateWeights allocates the top-level Weights slice for all trainable layers.
func allocateWeights(nn *NeuralNetwork) {
	nn.WeightsAndBiases.Weights = make([][][]float64, len(nn.Layers)+1)
	for i := range nn.WeightsAndBiases.Weights {
		nn.WeightsAndBiases.Weights[i] = make([][]float64, fanOut(nn, i))
	}
}

// initLayerWeights fills nn.WeightsAndBiases.Weights[layerIndex] using the
// given Initialization strategy.
func initLayerWeights(nn *NeuralNetwork, layerIndex int, init Initialization) {
	fi := fanIn(nn, layerIndex)
	fo := fanOut(nn, layerIndex)

	for j := 0; j < fo; j++ {
		weights := make([]float64, fi)

		switch init {

		// Kaiming Normal: w ~ N(0, sqrt(2/fan_in))
		case KaimingNormalInitializer:
			std := math.Sqrt(2.0 / float64(fi))
			for k := range weights {
				weights[k] = rand.NormFloat64() * std
			}

		// Kaiming Uniform: w ~ U(-limit, limit), limit = sqrt(6/fan_in)
		case KaimingUniformInitializer:
			limit := math.Sqrt(6.0 / float64(fi))
			for k := range weights {
				weights[k] = (rand.Float64()*2 - 1) * limit
			}

		// Xavier Normal: w ~ N(0, sqrt(2/(fan_in+fan_out)))
		case XavierNormalInitializer:
			std := math.Sqrt(2.0 / float64(fi+fo))
			for k := range weights {
				weights[k] = rand.NormFloat64() * std
			}

		// Xavier Uniform: w ~ U(-limit, limit), limit = sqrt(6/(fan_in+fan_out))
		case XavierUniformInitializer:
			limit := math.Sqrt(6.0 / float64(fi+fo))
			for k := range weights {
				weights[k] = (rand.Float64()*2 - 1) * limit
			}

		default:
			// Default to Xavier Normal if an unknown initializer is specified
			std := math.Sqrt(2.0 / float64(fi+fo))
			for k := range weights {
				weights[k] = rand.NormFloat64() * std
			}
		}

		nn.WeightsAndBiases.Weights[layerIndex][j] = weights
	}
}
