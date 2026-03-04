package tests

import (
	"testing"

	"github.com/ThakurMayank5/gonn/onnx"
)

func TestONNXModelLoading(t *testing.T) {

	onnxModelPath := "../onnx-models/mlp.onnx"

	model, err := onnx.LoadONNXModel(onnxModelPath)

	t.Logf("Loaded ONNX model from %s", onnxModelPath)

	if err != nil {
		t.Fatalf("Failed to load ONNX model: %v", err)

	}

	// if model == nil {
	// 	t.Fatalf("Model is nil after loading ONNX model")
	// }

	model.NeuralNetwork.Summary()

}

