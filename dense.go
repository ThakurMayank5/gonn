package gonn

import (
	"fmt"

	"github.com/ThakurMayank5/gonn/tensor"
)

type DenseLayer struct {
	Neurons int

	Initializer Initializer

	// shape of weights is (output_size, input_size)
	Weights tensor.Tensor

	// shape of bias is (output_size)
	Bias tensor.Tensor

	DropoutRate float64
}

func (d *DenseLayer) Init() {

	fanIn := d.Weights.Shape[1]
	fanOut := d.Weights.Shape[0]

	fmt.Println("Fan In:", fanIn)
	fmt.Println("Fan Out:", fanOut)

	switch d.Initializer {
	case KaimingNormalInitializer:
		d.Weights.Data = KaimingNormal(fanIn, fanOut, d.Weights.Shape[0]*d.Weights.Shape[1])
	case KaimingUniformInitializer:
		d.Weights.Data = KaimingUniform(fanIn, fanOut, d.Weights.Shape[0]*d.Weights.Shape[1])
	case XavierNormalInitializer:
		d.Weights.Data = XavierNormal(fanIn, fanOut, d.Weights.Shape[0]*d.Weights.Shape[1])
	case XavierUniformInitializer:
		d.Weights.Data = XavierUniform(fanIn, fanOut, d.Weights.Shape[0]*d.Weights.Shape[1])
	default:
		d.Weights.Data = KaimingNormal(fanIn, fanOut, d.Weights.Shape[0]*d.Weights.Shape[1])
	}

	// Initialize bias to zeros
	d.Bias.Data = make([]float64, fanOut)

}

/*

x    	*   w
[			[	[]
	.			[]
	.			[]
	.			.
	.			.
	.			.
]			]


*/

func (d *DenseLayer) Forward(x tensor.Tensor) tensor.Tensor {

	z := make([]float64, d.Neurons)

	for i := 0; i < d.Neurons; i++ {

		zx := 0.0
		for j := 0; j < x.Shape[0]; j++ {

			zx += d.Weights.Data[i*x.Shape[0]+j] * x.Data[j]

		}
		z[i] = zx + d.Bias.Data[i]

	}

	fmt.Printf(" Output: %v\n", z)
	fmt.Printf(" Output Shape: %v\n", []int{d.Neurons, 1})

	return tensor.Tensor{
		Data:  z,
		Shape: []int{d.Neurons, 1},
	}
}
