package neuralnetwork

import (
	"fmt"
	"math"
	"math/rand"
	"time"

	"github.com/ThakurMayank5/gonn/dataset"
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

	// LR scheduler initialization
	if model.TrainingConfig.ReduceOnPlateau {
		model.NeuralNetwork.TrainingState = &TrainingState{
			BestValLoss:       math.Inf(1), // Initialize to positive infinity
			LRPatienceCounter: 0,
		}
	}

	// Dropouts validation

	for i, layer := range model.NeuralNetwork.Layers {
		if layer.DropoutRate < 0 || layer.DropoutRate >= 1 {
			return fmt.Errorf("invalid dropout rate for layer %d: must be in [0, 1)", i+1)
		}
	}

	// set optimizers and other training parameters
	if model.TrainingConfig.Optimizer == "" {
		model.TrainingConfig.Optimizer = SGD
	}

	switch model.TrainingConfig.Optimizer {
	case SGD:
	// No additional setup needed for vanilla SGD
	case MOMENTUM, EMA_MOMENTUM, NESTEROV:
		// Initialize velocity buffers for momentum, exponential moving average momentum optimizers and nesterov

		VelocityW := make([][][]float64, len(model.NeuralNetwork.WeightsAndBiases.Weights))
		VelocityB := make([][]float64, len(model.NeuralNetwork.WeightsAndBiases.Biases))

		for l := range model.NeuralNetwork.WeightsAndBiases.Weights {
			VelocityW[l] = make([][]float64, len(model.NeuralNetwork.WeightsAndBiases.Weights[l]))
			for j := range model.NeuralNetwork.WeightsAndBiases.Weights[l] {
				VelocityW[l][j] = make([]float64, len(model.NeuralNetwork.WeightsAndBiases.Weights[l][j]))
			}
			VelocityB[l] = make([]float64, len(model.NeuralNetwork.WeightsAndBiases.Biases[l]))
		}

		// initial velocities are zero

		model.NeuralNetwork.OptimizerState = &OptimizerState{
			VelocitiesW: VelocityW,
			VelocitiesB: VelocityB,
		}

		if model.TrainingConfig.Beta == 0 {
			model.TrainingConfig.Beta = 0.9 // Default momentum factor
		}

	case RMSPROP:
		// Initialize cache for RMSProp
		CacheW := make([][][]float64, len(model.NeuralNetwork.WeightsAndBiases.Weights))
		CacheB := make([][]float64, len(model.NeuralNetwork.WeightsAndBiases.Biases))

		for l := range model.NeuralNetwork.WeightsAndBiases.Weights {
			CacheW[l] = make([][]float64, len(model.NeuralNetwork.WeightsAndBiases.Weights[l]))
			for j := range model.NeuralNetwork.WeightsAndBiases.Weights[l] {
				CacheW[l][j] = make([]float64, len(model.NeuralNetwork.WeightsAndBiases.Weights[l][j]))
			}
			CacheB[l] = make([]float64, len(model.NeuralNetwork.WeightsAndBiases.Biases[l]))
		}

		model.NeuralNetwork.OptimizerState = &OptimizerState{
			CacheW: CacheW,
			CacheB: CacheB,
		}

		if model.TrainingConfig.Beta == 0 {
			model.TrainingConfig.Beta = 0.9 // Default momentum factor
		}

	case ADAM, ADAMW:

		if model.TrainingConfig.WeightDecay == 0 && model.TrainingConfig.Optimizer == ADAMW {
			return fmt.Errorf("weight decay must be set for AdamW optimizer")
		}

		// Initialize velocity and cache for Adam
		VelocityW := make([][][]float64, len(model.NeuralNetwork.WeightsAndBiases.Weights))
		VelocityB := make([][]float64, len(model.NeuralNetwork.WeightsAndBiases.Biases))
		CacheW := make([][][]float64, len(model.NeuralNetwork.WeightsAndBiases.Weights))
		CacheB := make([][]float64, len(model.NeuralNetwork.WeightsAndBiases.Biases))

		for l := range model.NeuralNetwork.WeightsAndBiases.Weights {
			VelocityW[l] = make([][]float64, len(model.NeuralNetwork.WeightsAndBiases.Weights[l]))
			CacheW[l] = make([][]float64, len(model.NeuralNetwork.WeightsAndBiases.Weights[l]))
			for j := range model.NeuralNetwork.WeightsAndBiases.Weights[l] {
				VelocityW[l][j] = make([]float64, len(model.NeuralNetwork.WeightsAndBiases.Weights[l][j]))
				CacheW[l][j] = make([]float64, len(model.NeuralNetwork.WeightsAndBiases.Weights[l][j]))
			}
			VelocityB[l] = make([]float64, len(model.NeuralNetwork.WeightsAndBiases.Biases[l]))
			CacheB[l] = make([]float64, len(model.NeuralNetwork.WeightsAndBiases.Biases[l]))
		}

		if model.TrainingConfig.Beta1 == 0 {
			model.TrainingConfig.Beta1 = 0.9 // Default beta1 for Adam
		}
		if model.TrainingConfig.Beta2 == 0 {
			model.TrainingConfig.Beta2 = 0.999 // Default beta2 for Adam
		}

		if model.TrainingConfig.Epsilon == 0 {
			model.TrainingConfig.Epsilon = 1e-8 // Default epsilon for Adam
		}

		model.NeuralNetwork.OptimizerState = &OptimizerState{
			VelocitiesW: VelocityW,
			VelocitiesB: VelocityB,
			CacheW:      CacheW,
			CacheB:      CacheB,
			Timestep:    0,
		}

	default:
		return fmt.Errorf("unsupported optimizer: %s", model.TrainingConfig.Optimizer)
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

			// Backward Propagation — compute gradients
			grads, err := model.BackpropagateBatch(batchInputs, batchTargets)
			if err != nil {
				return err
			}

			// Apply gradients
			err = model.ApplyGradients(grads)
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

		if model.TrainingConfig.ReduceOnPlateau {

			if validationLoss < model.NeuralNetwork.TrainingState.BestValLoss {

				model.NeuralNetwork.TrainingState.BestValLoss = validationLoss
				model.NeuralNetwork.TrainingState.LRPatienceCounter = 0

			} else {

				model.NeuralNetwork.TrainingState.LRPatienceCounter++

				if model.NeuralNetwork.TrainingState.LRPatienceCounter >= model.TrainingConfig.LRPatience {

					oldLR := model.TrainingConfig.LearningRate

					newLR := model.TrainingConfig.LearningRate *
						model.TrainingConfig.LRFactor

					if newLR < model.TrainingConfig.MinLR {
						newLR = model.TrainingConfig.MinLR
					}

					model.TrainingConfig.LearningRate = newLR

					model.NeuralNetwork.TrainingState.LRPatienceCounter = 0

					fmt.Printf("Learning rate reduced from %.6f to %.6f due to plateau in validation loss\n", oldLR, newLR)

				}
			}
		}

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
