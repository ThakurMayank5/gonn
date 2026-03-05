package gonn

import "github.com/ThakurMayank5/gonn/tensor"

func (model *Model) Predict(input tensor.Tensor) tensor.Tensor {

	output := input

	for _, layer := range model.Layers {
		output = layer.Forward(output)
	}

	return output

}
