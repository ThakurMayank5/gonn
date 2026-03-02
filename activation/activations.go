package activation

import (
	"math"
)

// ActivationFunction represents the type of activation function
type ActivationFunction string

const (
	ReLU    ActivationFunction = "relu"
	Sigmoid ActivationFunction = "sigmoid"
	Tanh    ActivationFunction = "tanh"
	Softmax ActivationFunction = "softmax"
)

func GetActivationFunction(name ActivationFunction) func(float64) float64 {
	switch name {
	case ReLU:
		return reluFunc
	case Sigmoid:
		return sigmoidFunc
	case Tanh:
		return tanhFunc
	case Softmax:
		// Softmax is applied to entire vector, not element-wise
		return nil
	default:
		return nil
	}
}

func reluFunc(x float64) float64 {
	if x > 0 {
		return x
	}
	return 0
}

func sigmoidFunc(x float64) float64 {
	return 1 / (1 + math.Exp(-x))
}

func tanhFunc(x float64) float64 {
	return math.Tanh(x)
}

// SoftmaxFunc applies softmax to a vector
func SoftmaxFunc(x []float64) []float64 {
	result := make([]float64, len(x))
	max := x[0]
	for i := 1; i < len(x); i++ {
		if x[i] > max {
			max = x[i]
		}
	}

	// Subtract max for numerical stability
	sum := 0.0
	for i := 0; i < len(x); i++ {
		result[i] = math.Exp(x[i] - max)
		sum += result[i]
	}

	// Normalize
	for i := 0; i < len(result); i++ {
		result[i] /= sum
	}
	return result
}
