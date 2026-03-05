package gonn

import "github.com/ThakurMayank5/gonn/tensor"

type Conv2DLayer struct {
	Filters    int
	KernelSize int
	Stride     int
	Padding    int

	Initializer Initializer

	// shape of weights is (filters, inputChannels*kernelSize*kernelSize)
	Weights tensor.Tensor

	// shape of biases is (kernel size,1)
	Biases tensor.Tensor
}

func (c *Conv2DLayer) Forward(input tensor.Tensor) tensor.Tensor {
	// TODO: Implement forward pass for Conv2D layer
	return tensor.Tensor{}
}

func (c *Conv2DLayer) Init() {

}

func WithStride(stride int) LayerOption {
	return func(c any) {
		switch layer := c.(type) {
		case *Conv2DLayer:
			layer.Stride = stride
		case *MaxPool2DLayer:
			layer.Stride = stride
		case *AvgPool2DLayer:
			layer.Stride = stride
		}
	}
}

func WithPadding(padding int) LayerOption {
	return func(c any) {
		switch layer := c.(type) {
		case *Conv2DLayer:
			layer.Padding = padding

		case *MaxPool2DLayer:
			layer.Padding = padding
		case *AvgPool2DLayer:
			layer.Padding = padding
		}
	}
}

func Conv2D(filters, kernelSize int, options ...LayerOption) Layer {
	c := &Conv2DLayer{
		Filters:    filters,
		KernelSize: kernelSize,
		Stride:     1,
		Padding:    0,
	}
	for _, option := range options {
		option(c)
	}
	return c
}
