package gonn

import "github.com/ThakurMayank5/gonn/tensor"

type FlattenLayer struct {
	InputShape []int
}

// TODO: REVISE THIS LATER
func (f *FlattenLayer) Forward(input tensor.Tensor) tensor.Tensor {

	// Create a new tensor with the flattened shape
	return tensor.Tensor{
		Data:  input.Data,
		Shape: []int{len(input.Data)},
	}
}

func (f *FlattenLayer) Init() {}
