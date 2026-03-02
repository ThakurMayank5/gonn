package neuralnetwork

import "github.com/ThakurMayank5/gonn/losses"

func (model *Model) ForwardPassBatch(batchInputs [][]float64, batchTargets [][]float64) (float64, error) {

	batchLoss := 0.0

	// iteration over mini-batch
	for i := range batchInputs {
		input := batchInputs[i]
		target := batchTargets[i]

		output, err := model.NeuralNetwork.Predict(input)
		if err != nil {
			return 0, err
		}

		loss := 0.0

		// Compute loss (use Cross-Entropy for Softmax, MSE for others)
		if model.NeuralNetwork.OutputLayer.ActivationFunction == "softmax" {
			loss, err = losses.CategoricalCrossEntropy(output, target)
		} else {
			loss, err = losses.MeanSquaredError(target, output)
		}
		if err != nil {
			return 0, err
		}

		batchLoss += loss

	}

	loss := batchLoss / float64(len(batchInputs))

	return loss, nil
}
