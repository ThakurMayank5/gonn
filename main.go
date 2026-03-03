package main

import (
	"fmt"

	activ "github.com/ThakurMayank5/gonn/activation"
	"github.com/ThakurMayank5/gonn/dataloader"
	"github.com/ThakurMayank5/gonn/dataset"
	nn "github.com/ThakurMayank5/gonn/neuralnetwork"
)

// Fashion MNIST class labels (index matches one-hot position)
var classNames = []string{
	"T-shirt/top", // 0
	"Trouser",     // 1
	"Pullover",    // 2
	"Dress",       // 3
	"Coat",        // 4
	"Sandal",      // 5
	"Shirt",       // 6
	"Sneaker",     // 7
	"Bag",         // 8
	"Ankle boot",  // 9
}

func main() {

	// Fashion MNIST Clothing Classification
	//
	// CSV format (fashion-mnist_train.csv / fashion-mnist_test.csv):
	//   Col  0      : label  (int 0-9)
	//   Col  1-784  : pixel values (0-255)
	//
	// Architecture:
	//   Input  : 784   (28×28 pixels)
	//   Hidden : 256   → ReLU   (Kaiming Normal)
	//   Hidden : 128   → ReLU   (Kaiming Normal)
	//   Hidden :  64   → ReLU   (Kaiming Normal)
	//   Output :  10   → Softmax (Kaiming Normal)

	model := nn.Model{
		NeuralNetwork: nn.NeuralNetwork{
			InputLayer: nn.InputLayer{
				Neurons: 784,
			},
			Layers: []nn.Layer{
				{
					Neurons:            256,
					ActivationFunction: activ.ReLU,
					Initialization:     nn.KaimingNormalInitializer,
				},
				{
					Neurons:            128,
					ActivationFunction: activ.ReLU,
					Initialization:     nn.KaimingNormalInitializer,
				},
				{
					Neurons:            64,
					ActivationFunction: activ.ReLU,
					Initialization:     nn.KaimingNormalInitializer,
				},
			},
			OutputLayer: nn.OutputLayer{
				Neurons:            10,
				ActivationFunction: activ.Softmax,
				Initialization:     nn.KaimingNormalInitializer,
			},
		},
		TrainingConfig: nn.TrainingConfig{
			Epochs:       30,
			LearningRate: 0.0005,
			Optimizer:    nn.ADAM,
			LossFunction: "categorical_crossentropy",
			BatchSize:    128,
		},
	}

	model.NeuralNetwork.Summary()

	// --- Step 1: Initialize weights ---

	err := model.InitializeWeights()
	if err != nil {
		fmt.Println("Error initializing weights:", err)
		return
	}

	// Build pixel column slice (cols 1-784)
	pixelCols := make([]int, 784)
	for i := range pixelCols {
		pixelCols[i] = i + 1
	}

	cfg := dataset.CSVConfig{
		HasHeader:      true,
		InputColumns:   pixelCols,
		HasLabelColumn: true,
		LabelColumn:    0,
		NumClasses:     10,
		Delimiter:      ',',
		Scaling:        dataset.MinMaxNormalize,
	}

	// --- Step 2: Load training set ---

	fmt.Println("Loading training set...")
	train, err := dataloader.FromCSV("fashion-mnist_train.csv", cfg)
	if err != nil {
		fmt.Println("Error loading training set:", err)
		return
	}
	fmt.Printf("Training set: %d samples, %d features, %d classes\n",
		train.NumSamples, train.NumFeatures, train.NumOutputs)

	// --- Step 3: Load test set ---

	fmt.Println("Loading test set...")
	test, err := dataloader.FromCSV("fashion-mnist_test.csv", cfg)
	if err != nil {
		fmt.Println("Error loading test set:", err)
		return
	}
	fmt.Printf("Test set:     %d samples\n\n", test.NumSamples)

	// --- Step 4: Train ---

	err = model.Fit(train, test)
	if err != nil {
		fmt.Println("Training error:", err)
		return
	}

	fmt.Println("\nTraining completed successfully!")

	// --- Step 5: Save trained weights ---

	err = model.SaveWeights("fashion_mnist.weights")
	if err != nil {
		fmt.Println("Error saving weights:", err)
		return
	}

	fmt.Println("Weights saved to fashion_mnist.weights")

	// --- Step 6: Load weights into a fresh model ---

	fmt.Println("\nLoading weights into a fresh model...")

	loadedModel := nn.Model{
		NeuralNetwork: nn.NeuralNetwork{
			InputLayer: nn.InputLayer{
				Neurons: 784,
			},
			Layers: []nn.Layer{
				{Neurons: 256, ActivationFunction: activ.ReLU},
				{Neurons: 128, ActivationFunction: activ.ReLU},
				{Neurons: 64, ActivationFunction: activ.ReLU},
			},
			OutputLayer: nn.OutputLayer{
				Neurons:            10,
				ActivationFunction: activ.Softmax,
			},
		},
	}

	err = loadedModel.LoadWeights("fashion_mnist.weights")
	if err != nil {
		fmt.Println("Error loading weights:", err)
		return
	}

	fmt.Println("Weights loaded from fashion_mnist.weights")

	// --- Step 7: Evaluate on test set ---

	fmt.Printf("\n--- Evaluation on Test Set (%d samples) ---\n", test.NumSamples)
	_, err = loadedModel.Evaluate(test)
	if err != nil {
		fmt.Println("Evaluation error:", err)
		return
	}

	// --- Step 8: Sample predictions (first 2 of each class) ---

	fmt.Println("\n--- Sample Predictions from Test Set (first 2 per class) ---")
	classCounts := make(map[int]int)
	samplesPerClass := 2

	for i := 0; i < len(test.Inputs); i++ {
		actualIdx := 0
		for j := 1; j < len(test.Outputs[i]); j++ {
			if test.Outputs[i][j] > test.Outputs[i][actualIdx] {
				actualIdx = j
			}
		}

		if classCounts[actualIdx] >= samplesPerClass {
			continue
		}

		prediction, err := loadedModel.NeuralNetwork.Predict(test.Inputs[i])
		if err != nil {
			continue
		}

		predIdx := 0
		for j := 1; j < len(prediction); j++ {
			if prediction[j] > prediction[predIdx] {
				predIdx = j
			}
		}

		match := "✓"
		if predIdx != actualIdx {
			match = "✗"
		}

		fmt.Printf("Sample %4d: Predicted %-14s (%.4f) | Actual %-14s %s\n",
			i+1,
			classNames[predIdx], prediction[predIdx],
			classNames[actualIdx],
			match)

		classCounts[actualIdx]++

		if len(classCounts) == 10 {
			allDone := true
			for _, c := range classCounts {
				if c < samplesPerClass {
					allDone = false
					break
				}
			}
			if allDone {
				break
			}
		}
	}
}
