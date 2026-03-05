package gonn

import (
	"math"

	"github.com/ThakurMayank5/gonn/tensor"
)

type MaxPool2DLayer struct {
	PoolSize int
	Stride   int
	Padding  int
}

// TODO
func (p *MaxPool2DLayer) Forward(input tensor.Tensor) tensor.Tensor {

	cOut := input.Shape[0]
	hIn := input.Shape[1]
	wIn := input.Shape[2]

	hOut := (hIn+2*p.Padding-p.PoolSize)/p.Stride + 1
	wOut := (wIn+2*p.Padding-p.PoolSize)/p.Stride + 1

	outputData := make([]float64, cOut*hOut*wOut)

	for c := 0; c < cOut; c++ {
		for h := 0; h < hOut; h++ {
			for w := 0; w < wOut; w++ {

				maxVal := math.Inf(-1) // Initialize to negative infinity
				for kh := 0; kh < p.PoolSize; kh++ {
					for kw := 0; kw < p.PoolSize; kw++ {
						hIn := h*p.Stride + kh - p.Padding
						wIn := w*p.Stride + kw - p.Padding
						if hIn >= 0 && hIn < input.Shape[1] && wIn >= 0 && wIn < input.Shape[2] {
							inputVal := input.Data[c*input.Shape[1]*input.Shape[2]+hIn*input.Shape[2]+wIn]
							if inputVal > maxVal {
								maxVal = inputVal
							}

						}
					}
				}
				outputData[c*hOut*wOut+h*wOut+w] = maxVal
			}
		}
	}

	return tensor.Tensor{
		Data:  outputData,
		Shape: []int{cOut, hOut, wOut},
	}
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

	cOut := input.Shape[0]
	hIn := input.Shape[1]
	wIn := input.Shape[2]

	hOut := (hIn+2*p.Padding-p.PoolSize)/p.Stride + 1
	wOut := (wIn+2*p.Padding-p.PoolSize)/p.Stride + 1

	outputData := make([]float64, cOut*hOut*wOut)

	for c := 0; c < cOut; c++ {
		for h := 0; h < hOut; h++ {
			for w := 0; w < wOut; w++ {
				var sumVal float64
				var count int
				for kh := 0; kh < p.PoolSize; kh++ {
					for kw := 0; kw < p.PoolSize; kw++ {
						hIn := h*p.Stride + kh - p.Padding
						wIn := w*p.Stride + kw - p.Padding
						if hIn >= 0 && hIn < input.Shape[1] && wIn >= 0 && wIn < input.Shape[2] {
							inputVal := input.Data[c*input.Shape[1]*input.Shape[2]+hIn*input.Shape[2]+wIn]
							sumVal += inputVal
							count++
						}
					}
				}
				outputData[c*hOut*wOut+h*wOut+w] = sumVal / float64(count)
			}
		}
	}

	return tensor.Tensor{
		Data:  outputData,
		Shape: []int{cOut, hOut, wOut},
	}
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
