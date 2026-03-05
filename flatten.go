package gonn

import "github.com/ThakurMayank5/gonn/tensor"

type FlattenLayer struct{}


// TODO: REVISE THIS LATER
func (f *FlattenLayer) Forward(input tensor.Tensor) tensor.Tensor {

	// Calculate the total number of elements in the input tensor

	totalElements := 1
	for _, dim := range input.Shape {
		totalElements *= dim
	}

	// Create a new tensor with the flattened shape
	return tensor.Tensor{
		Data:  input.Data,
		Shape: []int{totalElements},
	}
}

func (f *FlattenLayer) Init() {}

