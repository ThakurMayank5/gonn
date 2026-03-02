package neuralnetwork

import (
	"fmt"
	"github.com/ThakurMayank5/gonn/dataset"
	"math/rand"
	"time"
)

func (model *Model) Fit(training dataset.Dataset, validation dataset.Dataset) error {

	// initialize a random seed for further use
	rand.Seed(time.Now().UnixNano())

	// Dataset validation

	if len(training.Inputs) == 0 || len(training.Outputs) == 0 {
		return fmt.Errorf("training dataset is empty")
	}

	if len(training.Inputs) != len(training.Outputs) {
		return fmt.Errorf("number of inputs and outputs must be the same")
	}

	if len(training.Inputs[0]) != model.NeuralNetwork.InputLayer.Neurons {
		return fmt.Errorf("input data does not match the number of neurons in the input layer")
	}

	total_samples := len(training.Inputs)

	epochs := model.TrainingConfig.Epochs
	batchSize := model.TrainingConfig.BatchSize

	batchesPerEpoch := (total_samples + batchSize - 1) / batchSize // Ceiling division

	fmt.Printf("Starting training for %d epochs with batch size %d (%d batches per epoch)\n", epochs, batchSize, batchesPerEpoch)

	for epoch := 1; epoch <= epochs; epoch++ {

		fmt.Printf("Epoch %d/%d\n", epoch, epochs)

		// Shuffle the training data at the beginning of each epoch
		shuffledIndices := rand.Perm(total_samples)

		for batch := 0; batch < batchesPerEpoch; batch++ {

			start := batch * batchSize
			end := start + batchSize
			if end > total_samples {
				end = total_samples
			}

			// Create mini-batch
			batchInputs := make([][]float64, end-start)
			batchTargets := make([][]float64, end-start)

			for i, idx := range shuffledIndices[start:end] {
				batchInputs[i] = training.Inputs[idx]
				batchTargets[i] = training.Outputs[idx]
			}

			// Backward Propagation with weight/bias updates for the entire batch
			err := model.BackpropagateBatch(batchInputs, batchTargets)
			if err != nil {
				return err
			}

			// Show progress bar
			ShowProgress(batch+1, batchesPerEpoch)

		}
		// Validation per epoch

		validationLoss, err := model.ForwardPassBatch(validation.Inputs, validation.Outputs)
		if err != nil {
			return err
		}
		fmt.Printf("\nValidation Loss: %.4f\n", validationLoss)

	}

	return nil

}

func ShowProgress(done, total int) {
	percent := float64(done) / float64(total) * 100
	fmt.Printf("\rProgress: %d/%d (%.2f%%)[", done, total, percent)

	number := 20 // Total number of characters in the progress bar

	filled := int(percent / 100 * float64(number))
	for i := 0; i < number; i++ {
		if i < filled {
			fmt.Print("#")
		} else {
			fmt.Print("-")
		}
	}

	fmt.Print("]")

}
