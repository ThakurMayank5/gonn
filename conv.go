package gonn

import (
	"github.com/ThakurMayank5/gonn/tensor"
)

type Conv2DLayer struct {
	Filters    int
	KernelSize int
	Stride     int
	Padding    int

	Initializer Initializer

	// shape of weights is (filters, inputChannels,kernelSize,kernelSize)
	Weights tensor.Tensor

	// shape of biases is (kernel size)
	Biases tensor.Tensor

	// stored during model build
	InputChannels int
}

func (conv *Conv2DLayer) Forward(input tensor.Tensor) tensor.Tensor {

	// H_out = (H_in + 2*padding - K_h) / stride + 1
	// W_out = (W_in + 2*padding - K_w) / stride + 1

	cIn := input.Shape[0]
	HIn := input.Shape[1]
	WIn := input.Shape[2]

	cOut := conv.Filters

	hOut := (input.Shape[1]+2*conv.Padding-conv.KernelSize)/conv.Stride + 1
	wOut := (input.Shape[2]+2*conv.Padding-conv.KernelSize)/conv.Stride + 1

	outputData := make([]float64, cOut*hOut*wOut)

	for f := 0; f < cOut; f++ {
		for h := 0; h < hOut; h++ {
			for w := 0; w < wOut; w++ {

				var sum float64
				for c := 0; c < cIn; c++ {
					for kh := 0; kh < conv.KernelSize; kh++ {
						for kw := 0; kw < conv.KernelSize; kw++ {
							hIn := h*conv.Stride + kh - conv.Padding
							wIn := w*conv.Stride + kw - conv.Padding
							if hIn >= 0 && hIn < HIn && wIn >= 0 && wIn < WIn {
								inputVal := input.Data[c*HIn*WIn+hIn*WIn+wIn]
								weightVal := conv.Weights.Data[f*cIn*conv.KernelSize*conv.KernelSize+c*conv.KernelSize*conv.KernelSize+kh*conv.KernelSize+kw]
								sum += inputVal * weightVal
							}
						}
					}
				}
				sum += conv.Biases.Data[f]
				outputData[f*hOut*wOut+h*wOut+w] = sum
			}
		}
	}

	return tensor.Tensor{
		Data:  outputData,
		Shape: []int{cOut, hOut, wOut},
	}
}

/*

Input shape: (inputChannels, inputHeight, inputWidth)

weights shape: (filters or outChannels, inputChannels*kernelSize*kernelSize)

bias shape: (filters, 1)

Output shape: (filters, outputHeight, outputWidth)

out channels => filters

in channels => inputChannels

outputHeight = (inputHeight - kernelSize + 2*padding) / stride + 1

outputWidth = (inputWidth - kernelSize + 2*padding) / stride + 1

*/

// shape of weights is (filters, inputChannels*kernelSize*kernelSize)

func (c *Conv2DLayer) Init() {

	// fanIn => inputChannels * kernelSize * kernelSize

	// fanOut => filters * kernelSize * kernelSize

	fanIn := c.InputChannels * c.KernelSize * c.KernelSize
	fanOut := c.Filters * c.KernelSize * c.KernelSize

	weightSize := c.Filters * c.InputChannels * c.KernelSize * c.KernelSize

	switch c.Initializer {
	case KaimingNormalInitializer:
		c.Weights.Data = KaimingNormal(fanIn, fanOut, weightSize)
	case KaimingUniformInitializer:
		c.Weights.Data = KaimingUniform(fanIn, fanOut, weightSize)
	case XavierNormalInitializer:
		c.Weights.Data = XavierNormal(fanIn, fanOut, weightSize)
	case XavierUniformInitializer:
		c.Weights.Data = XavierUniform(fanIn, fanOut, weightSize)
	default:
		c.Weights.Data = KaimingNormal(fanIn, fanOut, weightSize)
	}

	// Initialize biases to zeros
	c.Biases.Data = make([]float64, c.Filters)

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
