package neuralnetwork

import (
	"fmt"

	"github.com/ThakurMayank5/gonn/activation"
	vectors "github.com/ThakurMayank5/gonn/vectors"
)

func (nn *NeuralNetwork) Predict(input []float64) ([]float64, error) {

	weights := nn.WeightsAndBiases.Weights
	biases := nn.WeightsAndBiases.Biases

	x := input

	// Iterate through each layer
	for i := 0; i < len(weights); i++ {
		{

			// Default to ReLU for hidden layers
			activationFunction := activation.ReLU

			if i == len(weights)-1 {
				activationFunction = nn.OutputLayer.ActivationFunction
			} else {
				activationFunction = nn.Layers[i].ActivationFunction
			}

			newX := make([]float64, len(biases[i])) // Initialized elements as 0

			// Iterate through each neuron in the current layer
			for j := range biases[i] {
				currWeights := weights[i][j]
				dotProduct, err := vectors.DotProduct(x, currWeights)

				if err != nil {
					return nil, fmt.Errorf("error computing dot product: %v", err)
				}

				if activationFunction != activation.Softmax {
					activationFunctionToUse := activation.GetActivationFunction(activationFunction)
					newX[j] = activationFunctionToUse(dotProduct + biases[i][j])
				} else {
					newX[j] = dotProduct + biases[i][j] // Store pre-activation for softmax
				}

			}

			// Apply softmax if needed
			if i == len(weights)-1 && activationFunction == activation.Softmax {
				newX = activation.SoftmaxFunc(newX)
			}

			x = newX

		}
	}
	return x, nil
}
