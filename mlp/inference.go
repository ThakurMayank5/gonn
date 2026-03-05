package mlp

import (
	"math"
)

// Predict performs inference on the input data and returns the output of the neural network.
// This is optmized for performance, using flat weight arrays and multiple accumulators in the dot product to maximize CPU utilization.
// To use this function, the model must be in inference mode, which initializes the necessary buffers and precomputes the flat weight arrays for efficient access.
func (model *Model) Predict(input []float64) ([]float64, error) {

	infState := model.InferenceState
	x := input
	numLayers := len(infState.InputSizes)

	for i := range numLayers {

		// Lodaing layer data
		newX := infState.InferenceBuffer[i]
		fw := infState.FlatWeights[i]
		bi := infState.Biases[i]
		actFn := infState.ActivationFunctions[i]
		inputSize := infState.InputSizes[i]
		neurons := len(bi)

		// Computing layer output
		for j := range neurons {

			// offset of j-th neuron's weights in the flat weight array
			offset := j * inputSize

			// slice   [low :       high :                  max]
			// fw      [offset:     offset+inputSize:       offset+inputSize]
			// this returns a slice of fw corresponding to the weights for the j-th neuron in the current layer

			newX[j] = dot8(x, fw[offset:offset+inputSize:offset+inputSize]) + bi[j]

			// Applying activation functions (non vectorized) other than Softmax (vectorized)
			if actFn != nil {
				newX[j] = actFn(newX[j])
			}
		}

		// Applying Softmax activation function in-place if applicable in the output layer
		if actFn == nil && i == numLayers-1 {
			softmaxInPlace(newX)
		}

		x = newX
	}
	return x, nil
}

// softmaxInPlace applies softmax directly to x
// No additional memory allocation
func softmaxInPlace(x []float64) {
	max := x[0]
	for i := 1; i < len(x); i++ {
		if x[i] > max {
			max = x[i]
		}
	}

	sum := 0.0
	for i := range x {
		x[i] = math.Exp(x[i] - max)
		sum += x[i]
	}

	for i := range x {
		x[i] /= sum
	}
}

// This is a performane trick to speed up the dot product by using 8 independent accumulators.
// dot8 computes a dot product using 8 independent accumulators.
// This breaks the serial dependency chain on a single sum variable,
// allowing the CPU to utilize multiple floating-point units in parallel.
func dot8(a, b []float64) float64 {
	n := len(a)

	// 8 independent accumulators for parallelism
	s0, s1, s2, s3 := 0.0, 0.0, 0.0, 0.0
	s4, s5, s6, s7 := 0.0, 0.0, 0.0, 0.0

	end := n &^ 7 // closest multiple of 8

	k := 0

	// For all multiples of using 8 parallel accumulators
	for ; k < end; k += 8 {
		s0 += a[k] * b[k]
		s1 += a[k+1] * b[k+1]
		s2 += a[k+2] * b[k+2]
		s3 += a[k+3] * b[k+3]
		s4 += a[k+4] * b[k+4]
		s5 += a[k+5] * b[k+5]
		s6 += a[k+6] * b[k+6]
		s7 += a[k+7] * b[k+7]
	}

	// Remaining elements after multiple of 8
	for ; k < n; k++ {
		s0 += a[k] * b[k]
	}
	return (s0 + s1 + s2 + s3) + (s4 + s5 + s6 + s7)
}
