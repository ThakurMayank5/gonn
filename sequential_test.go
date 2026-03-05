package gonn

import (
	"fmt"
	"testing"

	"github.com/ThakurMayank5/gonn/tensor"
)

func TestSequentialMLPModel(t *testing.T) {

	// Create a simple sequential model
	model := Sequential(
		Dense(4, WithInitializer(XavierNormalInitializer), WithDropout(0.5)),
		ReLU(),
		Dense(2, WithInitializer(XavierNormalInitializer)),
		SoftMax(),
	)

	fmt.Println("Created a Sequential Model")

	model.Build([]int{1})

	// model.Summary()

	output := model.Predict(tensor.Tensor{
		Data:  []float64{1.0},
		Shape: []int{1},
	})

	fmt.Printf("Model output: %v\n", output.Data)

}

func TestSequentialCNNModel(t *testing.T) {

	fmt.Println("Starting Conculation Neural Network Test")

	model := Sequential(
		Input([]int{3, 28, 28}),
		Conv2D(2, 3, WithStride(1), WithPadding(1), WithInitializer(XavierUniformInitializer)),
		ReLU(),
		MaxPool2D(2, WithStride(2)),
		Conv2D(4, 3, WithStride(1), WithPadding(1), WithInitializer(XavierUniformInitializer)),
		ReLU(),
		Conv2D(2, 3, WithStride(1), WithPadding(1), WithInitializer(XavierUniformInitializer)),
		ReLU(),
		Flatten(),
		Dense(10, WithInitializer(XavierUniformInitializer)),
		SoftMax(),
	)

	model.Build([]int{3, 28, 28})

	model.Summary()

	model.Predict(
		tensor.Tensor{
			Data:  make([]float64, 3*28*28), // Example input data (3 channels, 28x28 image)
			Shape: []int{3, 28, 28},
		},
	)

}
