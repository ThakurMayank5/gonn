package gonn

import "github.com/ThakurMayank5/gonn/tensor"

type ReLULayer struct{}

func (r *ReLULayer) Forward(input tensor.Tensor) tensor.Tensor {

	outputData := make([]float64, len(input.Data))

	for i, val := range input.Data {
		if val > 0 {
			outputData[i] = val
		} else {
			outputData[i] = 0
		}
	}

	return tensor.Tensor{
		Data:  outputData,
		Shape: input.Shape,
	}

}

func (r *ReLULayer)Init(){}