package gonn

import (
	"github.com/ThakurMayank5/gonn/tensor"
)

type MaxPool2DLayer struct {
	PoolSize int
	Stride   int
	Padding  int
}

// TODO
func (p *MaxPool2DLayer) Forward(input tensor.Tensor) tensor.Tensor {

	return tensor.Tensor{}
}

func (p *MaxPool2DLayer) Init() {

}

// Stride and padding options are applied in conv.go file in switch statements

func MaxPool2D(poolSize int, options ...LayerOption) Layer {
	p := &MaxPool2DLayer{
		PoolSize: poolSize,
		Stride:   poolSize, // default stride is equal to pool size
	}
	for _, option := range options {
		option(p)
	}
	return p
}

type AvgPool2DLayer struct {
	PoolSize int
	Stride   int
	Padding  int
}

// TODO
func (p *AvgPool2DLayer) Forward(input tensor.Tensor) tensor.Tensor {

	return tensor.Tensor{}
}

func (p *AvgPool2DLayer) Init() {

}

// Stride and padding options are applied in conv.go file in switch statements

func AvgPool2D(poolSize int, options ...LayerOption) Layer {
	p := &AvgPool2DLayer{
		PoolSize: poolSize,
		Stride:   poolSize, // default stride is equal to pool size
	}
	for _, option := range options {
		option(p)
	}
	return p
}
