package neuralnetwork

import (
	"fmt"

	"github.com/ThakurMayank5/gonn/activation"
	"github.com/ThakurMayank5/gonn/vectors"
)

// z is pre activation values, a is post activation values, predictions is the final output
func (model *Model) PredictBatch(batchInputs [][]float64, batchTargets [][]float64) (z [][][]float64, a [][][]float64, predictions [][]float64, err error) {

	batchSize := len(batchInputs)

	predictions = make([][]float64, batchSize)

	// Make each z[i] same size as bias size

	z = make([][][]float64, batchSize)

	a = make([][][]float64, batchSize)

	for i := range z {
		z[i] = make([][]float64, len(model.NeuralNetwork.WeightsAndBiases.Biases))
		a[i] = make([][]float64, len(model.NeuralNetwork.WeightsAndBiases.Biases))
	}

	for i := range z {
		for j := range z[i] {
			z[i][j] = make([]float64, len(model.NeuralNetwork.WeightsAndBiases.Biases[j]))
			a[i][j] = make([]float64, len(model.NeuralNetwork.WeightsAndBiases.Biases[j]))
		}
	}

	weights := model.NeuralNetwork.WeightsAndBiases.Weights
	biases := model.NeuralNetwork.WeightsAndBiases.Biases

	// iteration over mini-batch
	for batch := range batchInputs {

		x := batchInputs[batch]

		// Iterate through each layer
		for i := 0; i < len(weights); i++ {
			{

				// Default to ReLU for hidden layers
				activationFunction := activation.ReLU

				if i == len(weights)-1 {
					activationFunction = model.NeuralNetwork.OutputLayer.ActivationFunction
				} else {
					activationFunction = model.NeuralNetwork.Layers[i].ActivationFunction
				}

				newX := make([]float64, len(biases[i])) // Initialized elements as 0

				// Iterate through each neuron in the current layer
				for j := range biases[i] {
					currWeights := weights[i][j]
					dotProduct, err := vectors.DotProduct(x, currWeights)

					if err != nil {
						return nil, nil, nil, fmt.Errorf("error computing dot product: %v", err)
					}

					if activationFunction != activation.Softmax {
						activationFunctionToUse := activation.GetActivationFunction(activationFunction)

						newX[j] = activationFunctionToUse(dotProduct + biases[i][j])
					} else {
						newX[j] = dotProduct + biases[i][j] // Store pre-activation for softmax
					}

					// Store pre-activation values for backpropagation
					z[batch][i][j] = dotProduct + biases[i][j]

					// Store post-activation values for backpropagation
					a[batch][i][j] = newX[j]

				}

				// Apply softmax if needed
				if i == len(weights)-1 && activationFunction == activation.Softmax {
					newX = activation.SoftmaxFunc(newX)
				}

				x = newX

			}
		}

		predictions[batch] = x

	}

	return z, a, predictions, nil
}
