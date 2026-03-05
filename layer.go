package gonn

import "github.com/ThakurMayank5/gonn/tensor"

type LayerOption func(any)

func WithInitializer(initializer Initializer) LayerOption {
	return func(l any) {
		switch layer := l.(type) {

		case *DenseLayer:
			layer.Initializer = initializer

		case *Conv2DLayer:
			layer.Initializer = initializer
		}
	}
}

type Layer interface {
	Forward(x tensor.Tensor) tensor.Tensor
	Init()
}

func ReLU() Layer {
	return &ReLULayer{}
}

func SoftMax() Layer {
	return &SoftMaxLayer{}
}

func Flatten() Layer {
	return &FlattenLayer{}
}

func WithDropout(rate float64) LayerOption {
	return func(d any) {
		switch layer := d.(type) {
		case *DenseLayer:
			layer.DropoutRate = rate
		}
	}
}

func Dense(neuons int, options ...LayerOption) Layer {
	d := &DenseLayer{
		Neurons: neuons,
	}
	for _, option := range options {
		option(d)
	}
	return d
}
