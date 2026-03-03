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

	switch model.TrainingConfig.Optimizer {
	// Vanilla SGD
	case SGD:
		for l := range grads.GradW {
			for j := range grads.GradW[l] {
				for k := range grads.GradW[l][j] {
					weights[l][j][k] -= lr * grads.GradW[l][j][k]
				}
				biases[l][j] -= lr * grads.GradB[l][j]
			}
		}

	case MOMENTUM:
		beta := model.TrainingConfig.Beta
		for l := range grads.GradW {
			for j := range grads.GradW[l] {

				for k := range grads.GradW[l][j] {
					// Update velocity
					model.NeuralNetwork.OptimizerState.VelocitiesW[l][j][k] = beta*model.NeuralNetwork.OptimizerState.VelocitiesW[l][j][k] + grads.GradW[l][j][k]
				}
				model.NeuralNetwork.OptimizerState.VelocitiesB[l][j] = beta*model.NeuralNetwork.OptimizerState.VelocitiesB[l][j] + grads.GradB[l][j]
			}
		}
		// Update parameters using velocity
		for l := range grads.GradW {
			for j := range grads.GradW[l] {
				for k := range grads.GradW[l][j] {
					weights[l][j][k] -= lr * model.NeuralNetwork.OptimizerState.VelocitiesW[l][j][k]
				}
				biases[l][j] -= lr * model.NeuralNetwork.OptimizerState.VelocitiesB[l][j]
			}
		}

	case EMA_MOMENTUM:
		beta := model.TrainingConfig.Beta
		for l := range grads.GradW {
			for j := range grads.GradW[l] {

				for k := range grads.GradW[l][j] {
					// Update velocity
					model.NeuralNetwork.OptimizerState.VelocitiesW[l][j][k] = beta*model.NeuralNetwork.OptimizerState.VelocitiesW[l][j][k] + (1-beta)*grads.GradW[l][j][k]
				}
				model.NeuralNetwork.OptimizerState.VelocitiesB[l][j] = beta*model.NeuralNetwork.OptimizerState.VelocitiesB[l][j] + (1-beta)*grads.GradB[l][j]
			}
		}
		// Update parameters using velocity
		for l := range grads.GradW {
			for j := range grads.GradW[l] {
				for k := range grads.GradW[l][j] {
					weights[l][j][k] -= lr * model.NeuralNetwork.OptimizerState.VelocitiesW[l][j][k]
				}
				biases[l][j] -= lr * model.NeuralNetwork.OptimizerState.VelocitiesB[l][j]
			}
		}

	default:
		return fmt.Errorf("unsupported optimizer: %s", model.TrainingConfig.Optimizer)
	}

	return nil
}
