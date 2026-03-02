package neuralnetwork

import (
	"fmt"

	"github.com/ThakurMayank5/gonn/dataset"
	"github.com/ThakurMayank5/gonn/losses"
)

func (model *Model) Evaluate(dataset dataset.Dataset) (float64, error) {

	// Dataset validation
	if len(dataset.Inputs) == 0 || len(dataset.Outputs) == 0 {
		return 0.0, fmt.Errorf("dataset is empty")
	}

	if len(dataset.Inputs) != len(dataset.Outputs) {
		return 0.0, fmt.Errorf("number of inputs and outputs must be the same")
	}

	if len(dataset.Inputs[0]) != model.NeuralNetwork.InputLayer.Neurons {
		return 0.0, fmt.Errorf("input data does not match the number of neurons in the input layer")
	}

	correctPredictions := 0.0
	totalLoss := 0.0

	for i := range dataset.Inputs {
		input := dataset.Inputs[i]
		output, err := model.NeuralNetwork.Predict(input)
		if err != nil {
			fmt.Printf("Error predicting output for input %v: %v\n", input, err)
			return 0.0, err
		}

		// Compute loss (use Cross-Entropy for Softmax, MSE for others)
		var loss float64
		if model.NeuralNetwork.OutputLayer.ActivationFunction == "softmax" {
			loss, err = losses.CategoricalCrossEntropy(output, dataset.Outputs[i])
		} else {
			loss, err = losses.MeanSquaredError(dataset.Outputs[i], output)
		}
		if err != nil {
			fmt.Printf("Error computing loss for input %v: %v\n", input, err)
			return 0.0, err
		}

		totalLoss += loss

		// For multi-class classification: compare argmax of predictions with argmax of targets
		// Find predicted class (argmax)
		maxPredValue := output[0]
		predictedClass := 0
		for j := 1; j < len(output); j++ {
			if output[j] > maxPredValue {
				maxPredValue = output[j]
				predictedClass = j
			}
		}

		// Find target class (argmax of one-hot)
		maxTargetValue := dataset.Outputs[i][0]
		targetClass := 0
		for j := 1; j < len(dataset.Outputs[i]); j++ {
			if dataset.Outputs[i][j] > maxTargetValue {
				maxTargetValue = dataset.Outputs[i][j]
				targetClass = j
			}
		}

		// Check if predicted class matches target class
		if predictedClass == targetClass {
			correctPredictions++
		}
	}

	// Calculate accuracy as percentage (0-100)
	accuracyPercentage := (correctPredictions / float64(len(dataset.Inputs))) * 100.0
	avgLoss := totalLoss / float64(len(dataset.Inputs))

	fmt.Printf("  Loss: %.4f | Accuracy: %.2f%%\n", avgLoss, accuracyPercentage)

	return accuracyPercentage, nil
}
