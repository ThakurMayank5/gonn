package neuralnetwork

import "fmt"

// SGD: param -= lr * grad
func (model *Model) ApplyGradients(grads *GradientBuffer) error {
	lr := model.TrainingConfig.LearningRate
	weights := model.NeuralNetwork.WeightsAndBiases.Weights
	biases := model.NeuralNetwork.WeightsAndBiases.Biases

	if len(grads.GradW) != len(weights) || len(grads.GradB) != len(biases) {
		return fmt.Errorf("gradient dimensions do not match model parameters")
	}

	if len(grads.GradW) == 0 || len(grads.GradB) == 0 {
		return fmt.Errorf("no gradients to apply")
	}

	if model.TrainingConfig.Optimizer == "sgd" {

		for l := range grads.GradW {
			for j := range grads.GradW[l] {
				for k := range grads.GradW[l][j] {
					weights[l][j][k] -= lr * grads.GradW[l][j][k]
				}
				biases[l][j] -= lr * grads.GradB[l][j]
			}
		}
	} else {
		return fmt.Errorf("unsupported optimizer: %s", model.TrainingConfig.Optimizer)
	}
	return nil
}
