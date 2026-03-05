package tests

import (
	"fmt"
	"testing"

	"github.com/ThakurMayank5/gonn/onnx"
)

func TestONNXModelLoading(t *testing.T) {

	onnxModelPath := "../onnx-models/mlp.onnx"

	model, err := onnx.LoadONNXMLPModel(onnxModelPath)

	t.Logf("Loaded ONNX model from %s", onnxModelPath)

	if err != nil {
		t.Fatalf("Failed to load ONNX model: %v", err)

	}

	model.NeuralNetwork.Summary()

	model.SetInferenceMode(true)

	output, err := model.Predict([]float64{0.5, 0.2, 0.1, 0.4})
	if err != nil {
		t.Fatalf("Failed to predict: %v", err)
	}
	fmt.Printf("Prediction output: %v\n", output)
}
