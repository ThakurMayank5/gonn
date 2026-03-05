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
		ReLU(),
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

	model := Sequential(
		Conv2D(2, 3, WithStride(1), WithPadding(1), WithInitializer(XavierUniformInitializer)),
		ReLU(),
		MaxPool2D(2, WithStride(2)),
		Conv2D(4, 3, WithStride(1), WithPadding(1), WithInitializer(XavierUniformInitializer)),
		ReLU(),
		Conv2D(2, 3, WithStride(1), WithPadding(1), WithInitializer(XavierUniformInitializer)),
		ReLU(),
		Flatten(),
		Dense(10, WithInitializer(XavierUniformInitializer)),
		ReLU(),
	)

	model.Summary()

}
