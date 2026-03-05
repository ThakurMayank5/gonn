package gonn

import "fmt"

type Model struct {
	Layers []Layer
}

func (model *Model) Build(inputShape []int) {

	// Set shape of weights and biases tensor of layers
	for _, layer := range model.Layers {

		switch layer := layer.(type) {

		case *FlattenLayer:

			layer.InputShape = inputShape

			outputSize := 1
			for _, dim := range inputShape {
				outputSize *= dim
			}

			inputShape = []int{outputSize}

		case *Conv2DLayer:

			fmt.Println("The input shape is ", inputShape)

			layer.InputChannels = inputShape[0]

			fmt.Printf("Building Conv2D Layer with %d filters and kernel size %d\n", layer.Filters, layer.KernelSize)

			layer.Weights.Shape = []int{layer.Filters, layer.InputChannels, layer.KernelSize, layer.KernelSize}
			layer.Biases.Shape = []int{layer.Filters}

			h := inputShape[1]
			w := inputShape[2]

			hOut := (h+2*layer.Padding-layer.KernelSize)/layer.Stride + 1
			wOut := (w+2*layer.Padding-layer.KernelSize)/layer.Stride + 1

			inputShape = []int{layer.Filters, hOut, wOut}

		case *DenseLayer:

			fmt.Println("The input shape is ", inputShape)

			fmt.Printf("Building Dense Layer with %d neurons\n", layer.Neurons)

			denseLayer := layer

			denseLayer.Weights.Shape = []int{denseLayer.Neurons, inputShape[0]}
			denseLayer.Bias.Shape = []int{denseLayer.Neurons}

			fmt.Printf(" Weights shape: %v, Bias shape: %v\n", denseLayer.Weights.Shape, denseLayer.Bias.Shape)

			inputShape = []int{denseLayer.Neurons}

			fmt.Println(" Dense Layer built successfully")

		default:
			fmt.Printf("No Parameter Layer found : %T\n", layer)

		}

	}

	// Initialize weights for layers
	for _, layer := range model.Layers {
		layer.Init()
	}
}

func (model *Model) Summary() {

	fmt.Println("Model Summary:")
	for i, layer := range model.Layers {
		fmt.Printf(" Layer %d: %T\n", i+1, layer)
	}

}
