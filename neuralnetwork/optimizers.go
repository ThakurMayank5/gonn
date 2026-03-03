package neuralnetwork

import (
	"fmt"
	"math"
)

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

		// v = beta * v + grad
		// param -= lr * v

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

		// theta = beta*theta + (1-beta)*grad
		// param -= lr * theta

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
	case NESTEROV:

		// v_prev = v
		// v = beta * v - lr * grad
		// param += -beta * v_prev + (1 + beta) * v

		beta := model.TrainingConfig.Beta
		for l := range grads.GradW {
			for j := range grads.GradW[l] {

				var v_prev_b float64
				for k := range grads.GradW[l][j] {
					// Store previous velocity
					v_prev_w := model.NeuralNetwork.OptimizerState.VelocitiesW[l][j][k]
					// Update velocity
					model.NeuralNetwork.OptimizerState.VelocitiesW[l][j][k] = beta*model.NeuralNetwork.OptimizerState.VelocitiesW[l][j][k] - lr*grads.GradW[l][j][k]
					// Update parameters using Nesterov formula
					weights[l][j][k] += -beta*v_prev_w + (1+beta)*model.NeuralNetwork.OptimizerState.VelocitiesW[l][j][k]
				}
				v_prev_b = model.NeuralNetwork.OptimizerState.VelocitiesB[l][j]
				model.NeuralNetwork.OptimizerState.VelocitiesB[l][j] = beta*model.NeuralNetwork.OptimizerState.VelocitiesB[l][j] - lr*grads.GradB[l][j]
				biases[l][j] += -beta*v_prev_b + (1+beta)*model.NeuralNetwork.OptimizerState.VelocitiesB[l][j]
			}
		}

	case RMSPROP:

		// cache = beta*cache + (1-beta)*grad^2
		// param -= lr * grad / (sqrt(cache) + epsilon)

		squaredGradW := model.NeuralNetwork.OptimizerState.CacheW
		squaredGradB := model.NeuralNetwork.OptimizerState.CacheB

		beta := model.TrainingConfig.Beta
		epsilon := 1e-8

		for l := range grads.GradW {
			for j := range grads.GradW[l] {
				for k := range grads.GradW[l][j] {
					squaredGradW[l][j][k] = beta*squaredGradW[l][j][k] + (1-beta)*grads.GradW[l][j][k]*grads.GradW[l][j][k]
					weights[l][j][k] -= lr * grads.GradW[l][j][k] / (math.Sqrt(squaredGradW[l][j][k]) + epsilon)
				}
				squaredGradB[l][j] = beta*squaredGradB[l][j] + (1-beta)*grads.GradB[l][j]*grads.GradB[l][j]
				biases[l][j] -= lr * grads.GradB[l][j] / (math.Sqrt(squaredGradB[l][j]) + epsilon)
			}
		}

	default:
		return fmt.Errorf("unsupported optimizer: %s", model.TrainingConfig.Optimizer)
	}

	return nil
}
