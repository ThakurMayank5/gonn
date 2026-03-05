package benchmark

import (
	"math/rand"
	"testing"

	"github.com/ThakurMayank5/gonn/activation"
	nn "github.com/ThakurMayank5/gonn/mlp"
)

func setupModel() *nn.Model {
	model := &nn.Model{}

	model.NeuralNetwork.SetInputLayer(nn.InputLayer{
		Neurons:            784,
		ActivationFunction: activation.ReLU,
	})

	model.NeuralNetwork.AddLayer(nn.Layer{
		Neurons:            128,
		ActivationFunction: activation.ReLU,
		Initialization:     nn.XavierNormalInitializer,
	})

	model.NeuralNetwork.AddLayer(nn.Layer{
		Neurons:            64,
		ActivationFunction: activation.ReLU,
		Initialization:     nn.XavierNormalInitializer,
	})

	model.NeuralNetwork.SetOutputLayer(nn.OutputLayer{
		Neurons:            10,
		ActivationFunction: activation.Softmax,
		Initialization:     nn.KaimingNormalInitializer,
	})

	model.TrainingConfig = nn.TrainingConfig{
		LearningRate: 0.01,
		Optimizer:    nn.SGD,
		BatchSize:    32,
	}

	model.InitializeWeights()

	return model
}

func randomBatch(batchSize, inputSize, outputSize int) ([][]float64, [][]float64) {
	inputs := make([][]float64, batchSize)
	targets := make([][]float64, batchSize)
	for i := 0; i < batchSize; i++ {
		inputs[i] = make([]float64, inputSize)
		for j := range inputs[i] {
			inputs[i][j] = rand.Float64()*2 - 1
		}
		targets[i] = make([]float64, outputSize)
		targets[i][rand.Intn(outputSize)] = 1.0
	}
	return inputs, targets
}

func BenchmarkDenseForward(b *testing.B) {

	model := setupModel()

	model.SetInferenceMode(true)

	input := make([]float64, 784)
	for i := range input {
		input[i] = rand.Float64()*2 - 1
	}

	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		model.Predict(input)
	}
}

func BenchmarkBackpropagation(b *testing.B) {
	model := setupModel()
	inputs, targets := randomBatch(32, 784, 10)

	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		model.BackpropagateBatch(inputs, targets)
	}
}

func BenchmarkTrainingStep(b *testing.B) {
	model := setupModel()
	inputs, targets := randomBatch(32, 784, 10)

	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		grads, _ := model.BackpropagateBatch(inputs, targets)
		model.ApplyGradients(grads)
	}
}
