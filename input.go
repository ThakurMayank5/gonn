package gonn

import (
	"github.com/ThakurMayank5/gonn/tensor"
)

type InputLayer struct {

	// channel, height, width for 2D input and length for 1D input
	shape []int
}

func (i *InputLayer) Forward(input tensor.Tensor) tensor.Tensor {
	return input
}

func (i *InputLayer) Init() {

}

func Input(shape []int) Layer {
	return &InputLayer{shape: shape}
}
