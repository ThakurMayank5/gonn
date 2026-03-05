package gonn

import (
	"math"

	"github.com/ThakurMayank5/gonn/tensor"
)

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

func (r *ReLULayer) Init() {}

type SoftMaxLayer struct{}

func (s *SoftMaxLayer) Forward(input tensor.Tensor) tensor.Tensor {

	outputData := make([]float64, len(input.Data))

	var sumExp float64

	for _, val := range input.Data {
		sumExp += math.Exp(val)
	}

	for i, val := range input.Data {
		outputData[i] = math.Exp(val) / sumExp
	}

	return tensor.Tensor{
		Data:  outputData,
		Shape: input.Shape,
	}
}

func (s *SoftMaxLayer) Init() {}
