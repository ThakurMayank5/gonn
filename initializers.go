package gonn

import (
	"math"
	"math/rand"
)

type Initializer string

const (
	KaimingNormalInitializer  Initializer = "kaiming_normal"
	KaimingUniformInitializer Initializer = "kaiming_uniform"
	XavierNormalInitializer   Initializer = "xavier_normal"
	XavierUniformInitializer  Initializer = "xavier_uniform"
)

func KaimingNormal(fanIn int, fanOut int, weightSize int) []float64 {
	std := math.Sqrt(2.0 / float64(fanIn))
	weights := make([]float64, weightSize)
	for i := 0; i < weightSize; i++ {
		weights[i] = rand.NormFloat64() * std
	}
	return weights
}

func KaimingUniform(fanIn int, fanOut int, weightSize int) []float64 {
	limit := math.Sqrt(6.0 / float64(fanIn))
	weights := make([]float64, weightSize)
	for i := 0; i < weightSize; i++ {
		weights[i] = (rand.Float64()*2 - 1) * limit
	}
	return weights
}

func XavierNormal(fanIn, fanOut int, weightSize int) []float64 {
	std := math.Sqrt(2.0 / float64(fanIn+fanOut))
	weights := make([]float64, weightSize)
	for i := 0; i < weightSize; i++ {
		weights[i] = rand.NormFloat64() * std
	}
	return weights
}

func XavierUniform(fanIn, fanOut int, weightSize int) []float64 {
	limit := math.Sqrt(6.0 / float64(fanIn+fanOut))
	weights := make([]float64, weightSize)
	for i := 0; i < weightSize; i++ {
		weights[i] = (rand.Float64()*2 - 1) * limit
	}
	return weights
}
