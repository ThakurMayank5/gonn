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

	// shape of bias is (output_size, 1)
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
		d.Weights.Data = KaimingNormal(fanIn, fanOut)
	case KaimingUniformInitializer:
		d.Weights.Data = KaimingUniform(fanIn, fanOut)
	case XavierNormalInitializer:
		d.Weights.Data = XavierNormal(fanIn, fanOut)
	case XavierUniformInitializer:
		d.Weights.Data = XavierUniform(fanIn, fanOut)
	default:
		d.Weights.Data = KaimingNormal(fanIn, fanOut)
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

	fmt.Println("Dense Layer Forward Pass:")
	fmt.Printf(" Input: %v\n", x.Data)
	fmt.Printf(" Weights: %v\n", d.Weights.Data)
	fmt.Printf(" Bias: %v\n", d.Bias.Data)

	z := make([]float64, d.Neurons)

	for i := 0; i < d.Neurons; i++ {

		zx := 0.0
		for j := 0; j < d.Weights.Shape[1]; j++ {

			zx += d.Weights.Data[i*d.Weights.Shape[1]+j] * x.Data[j]

		}
		z[i] = zx + d.Bias.Data[i]

	}

	return tensor.Tensor{
		Data:  z,
		Shape: []int{d.Neurons, 1},
	}
}
