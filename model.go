package gonn

import "fmt"

type Model struct {
	Layers []Layer
}

func (model *Model) Build(inputShape []int) {

	// Set shape of weights and biases tensor of layers
	for _, layer := range model.Layers {

		switch layer := layer.(type) {

		case *DenseLayer:

			fmt.Println("The input shape is ", inputShape)

			fmt.Printf("Building Dense Layer with %d neurons\n", layer.Neurons)

			denseLayer := layer

			denseLayer.Weights.Shape = []int{denseLayer.Neurons, inputShape[0]}
			denseLayer.Bias.Shape = []int{denseLayer.Neurons, 1}

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
