package neuralnetwork

import (
	"fmt"
	"math"

	"github.com/ThakurMayank5/gonn/activation"
)

// This uses mini batch gradient descent
func (model *Model) BackpropagateBatch(batchInputs [][]float64, batchTargets [][]float64) error {

	oldWeights := model.NeuralNetwork.WeightsAndBiases.Weights

	newWeights := make([][][]float64, len(oldWeights))

	for l := range oldWeights {
		newWeights[l] = make([][]float64, len(oldWeights[l]))

		for j := range oldWeights[l] {
			newWeights[l][j] = make([]float64, len(oldWeights[l][j]))
			copy(newWeights[l][j], oldWeights[l][j])
		}
	}

	oldBiases := model.NeuralNetwork.WeightsAndBiases.Biases

	newBiases := make([][]float64, len(oldBiases))

	for l := range oldBiases {
		newBiases[l] = make([]float64, len(oldBiases[l]))
		copy(newBiases[l], oldBiases[l])
	}

	batch_size := len(batchInputs)

	deltas := make([][]float64, batch_size)

	// Initialize memory for deltas of output layers
	for i := 0; i < batch_size; i++ {

		deltas[i] = make([]float64, model.NeuralNetwork.OutputLayer.Neurons)
	}

	z, a, predictions, err := model.PredictBatch(batchInputs, batchTargets)

	if err != nil {
		return err
	}

	// Output Layer Backpropagation

	for i := 0; i < batch_size; i++ {
		for j := 0; j < model.NeuralNetwork.OutputLayer.Neurons; j++ {

			if model.NeuralNetwork.OutputLayer.ActivationFunction != activation.Softmax {
				activationDerivativeFunc := getActivationDerivative(model.NeuralNetwork.OutputLayer.ActivationFunction)

				if activationDerivativeFunc == nil {
					return fmt.Errorf("unsupported activation function for backpropagation: %s", model.NeuralNetwork.OutputLayer.ActivationFunction)
				}

				// For non-softmax activations, we need to multiply by the derivative of the activation function
				deltas[i][j] = (predictions[i][j] - batchTargets[i][j]) * activationDerivativeFunc(z[i][len(model.NeuralNetwork.WeightsAndBiases.Biases)-1][j])

				continue
			}

			// For softmax with cross-entropy, the delta is simply (pred - target)
			deltas[i][j] = predictions[i][j] - batchTargets[i][j]
		}
	}

	lastLayer := len(model.NeuralNetwork.WeightsAndBiases.Weights) - 1
	prevLayer := lastLayer - 1

	outputNeurons := model.NeuralNetwork.OutputLayer.Neurons
	prevNeurons := model.NeuralNetwork.Layers[len(model.NeuralNetwork.Layers)-1].Neurons

	gradW := make([][]float64, outputNeurons)
	gradB := make([]float64, outputNeurons)

	for j := 0; j < outputNeurons; j++ {
		gradW[j] = make([]float64, prevNeurons)
	}

	for i := 0; i < batch_size; i++ {

		for j := 0; j < outputNeurons; j++ {

			for k := 0; k < prevNeurons; k++ {

				gradW[j][k] += deltas[i][j] * a[i][prevLayer][k]
			}

			gradB[j] += deltas[i][j]
		}
	}

	scale := 1.0 / float64(batch_size)

	for j := 0; j < outputNeurons; j++ {
		for k := 0; k < prevNeurons; k++ {
			gradW[j][k] *= scale
		}
		gradB[j] *= scale
	}

	for j := 0; j < outputNeurons; j++ {

		for k := 0; k < prevNeurons; k++ {

			newWeights[lastLayer][j][k] -=
				model.TrainingConfig.LearningRate * gradW[j][k]
		}

		newBiases[lastLayer][j] -=
			model.TrainingConfig.LearningRate * gradB[j]
	}

	// Backprop For Hidden Layers

	for l := (len(model.NeuralNetwork.Layers)) - 1; l >= 0; l-- {

		currentLayerNeurons := model.NeuralNetwork.Layers[l].Neurons

		nextLayerNeurons := 0

		if l == len(model.NeuralNetwork.Layers)-1 {
			nextLayerNeurons = model.NeuralNetwork.OutputLayer.Neurons
		} else {
			nextLayerNeurons = model.NeuralNetwork.Layers[l+1].Neurons
		}

		// Reallocating memory for deltas of hidden layers

		// for each batch
		newDeltas := make([][]float64, batch_size)

		// for each neuron in the current layer
		for i := 0; i < batch_size; i++ {
			newDeltas[i] = make([]float64, currentLayerNeurons)
		}

		// Calculating new deltas

		// each batch
		for i := 0; i < batch_size; i++ {

			// each neuron in the current layer
			for j := 0; j < currentLayerNeurons; j++ {

				var nextWeights [][]float64

				if l == len(model.NeuralNetwork.Layers)-1 {
					nextWeights = model.NeuralNetwork.WeightsAndBiases.Weights[lastLayer]
				} else {
					nextWeights = model.NeuralNetwork.WeightsAndBiases.Weights[l+1]
				}

				// each neuron in the next layer
				for k := 0; k < nextLayerNeurons; k++ {
					newDeltas[i][j] += deltas[i][k] * nextWeights[k][j]
				}

				// perform activation derivative multiplication for hidden layers

				activationDerivativeFunc := getActivationDerivative(model.NeuralNetwork.Layers[l].ActivationFunction)

				if activationDerivativeFunc == nil {
					return fmt.Errorf("unsupported activation function for backpropagation: %s", model.NeuralNetwork.Layers[l].ActivationFunction)
				}

				newDeltas[i][j] *= activationDerivativeFunc(z[i][l][j])

			}

		}

		// Update deltas for the next iteration
		deltas = newDeltas

		// Compute gradients and update weights and biases for layer l

		gradWHidden := make([][]float64, currentLayerNeurons)
		gradBHidden := make([]float64, currentLayerNeurons)

		prevNeurons := 0

		if l == 0 {
			prevNeurons = model.NeuralNetwork.InputLayer.Neurons
		} else {

			prevNeurons = model.NeuralNetwork.Layers[l-1].Neurons
		}

		for j := 0; j < currentLayerNeurons; j++ {
			gradWHidden[j] = make([]float64, prevNeurons)
		}

		// each sample from batch
		for i := 0; i < batch_size; i++ {

			// each neuron in current
			for j := 0; j < currentLayerNeurons; j++ {

				for k := 0; k < prevNeurons; k++ {

					if l != 0 {

						// grad for weight connecting neuron k in previous layer to neuron j in current layer += delta of neuron j * activation of neuron k in previous layer
						gradWHidden[j][k] += deltas[i][j] * a[i][l-1][k]
					} else {
						// grad for weight connecting input k to neuron j in first hidden layer += delta of neuron j * input k
						gradWHidden[j][k] += deltas[i][j] * batchInputs[i][k]
					}

				}

				// sum of all gradients of samples in batch
				gradBHidden[j] += deltas[i][j]

			}

		}

		for j := 0; j < currentLayerNeurons; j++ {

			gradBHidden[j] /= float64(batch_size)

			for k := 0; k < prevNeurons; k++ {

				gradWHidden[j][k] /= float64(batch_size)
			}

		}

		// Update weights and biases for layer l

		for j := 0; j < currentLayerNeurons; j++ {

			for k := 0; k < prevNeurons; k++ {

				newWeights[l][j][k] -=
					model.TrainingConfig.LearningRate * gradWHidden[j][k]
			}

			newBiases[l][j] -=
				model.TrainingConfig.LearningRate * gradBHidden[j]
		}

	}

	// replace the model weights and biases with the new values
	model.NeuralNetwork.WeightsAndBiases.Weights = newWeights
	model.NeuralNetwork.WeightsAndBiases.Biases = newBiases

	return nil

}

// getActivationDerivative returns the derivative function for an activation
func getActivationDerivative(activationFunc activation.ActivationFunction) func(float64) float64 {
	switch activationFunc {
	case activation.ReLU:
		return func(z float64) float64 {
			if z > 0 {
				return 1.0
			}
			return 0.0
		}
	case activation.Sigmoid:
		return func(z float64) float64 {
			// sigmoid'(z) = sigmoid(z) * (1 - sigmoid(z))
			s := 1.0 / (1.0 + math.Exp(-z))
			return s * (1.0 - s)
		}
	case activation.Tanh:
		return func(z float64) float64 {
			// tanh'(z) = 1 - tanh(z)^2
			t := math.Tanh(z)
			return 1.0 - (t * t)
		}
	default:
		return nil
	}
}
