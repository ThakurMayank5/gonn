package onnx

import (
	"encoding/binary"
	"fmt"
	"math"
	"os"

	nn "github.com/ThakurMayank5/gonn/mlp"

	"google.golang.org/protobuf/proto"
)

func LoadONNXModel(filePath string) (*nn.Model, error) {

	data, err := os.ReadFile(filePath)
	if err != nil {
		return nil, fmt.Errorf("failed to read ONNX file: %v", err)
	}

	// Print the bytes in a readable hexadecimal format
	// for i, b := range data {
	// 	fmt.Printf("%02x ", b)
	// 	if (i+1)%16 == 0 {
	// 		fmt.Println()
	// 	}
	// }

	// Create a new ModelProto instance to hold the unmarshaled data
	modelONNX := &ModelProto{}

	// Unmarshal the ONNX model from the byte data
	err = proto.Unmarshal(data, modelONNX)
	if err != nil {
		return nil, fmt.Errorf("failed to unmarshal ONNX model: %v", err)
	}

	// fmt.Println(modelONNX)

	tensorMap := map[string]*TensorProto{}

	for _, t := range modelONNX.Graph.Initializer {
		tensorMap[*t.Name] = t
	}

	// model := &nn.Model{}

	denseEncountered := false

	layers := make([]nn.Layer, 0)

	layer := nn.Layer{}

	modelWeights := make([][][]float64, 0)

	modelBiases := make([][]float64, 0)

	inputLayer := nn.InputLayer{
		Neurons: 0,
	}

	for _, node := range modelONNX.Graph.Node {

		if !denseEncountered {
			layer = nn.Layer{}
		}

		switch *node.OpType {

		case "Gemm":

			w := tensorMap[node.Input[1]]
			b := tensorMap[node.Input[2]]

			weights := decodeTensorFloat32(w)
			bias := decodeTensorFloat32(b)

			weights2D := make([][]float64, 0)

			// Weights are stored in row-major format

			neurons := len(bias)

			index := 0

			fmt.Println("Neurons ", neurons)
			fmt.Println("Total Weights ", len(weights))

			size := len(weights) / neurons

			if inputLayer.Neurons == 0 {
				inputLayer.Neurons = size
			}

			fmt.Println("Size ", size)

			row := make([]float64, 0)

			for i := 0; i < neurons; i++ {

				row = make([]float64, 0)

				for j := 0; j < size; j++ {

					row = append(row, float64(weights[index]))
					index++
				}

				fmt.Println(row)

				weights2D = append(weights2D, row)

				row = make([]float64, 0)
			}

			modelWeights = append(modelWeights, weights2D)

			f64biases := make([]float64, len(bias))

			for i := range bias {
				f64biases[i] = float64(bias[i])
			}

			modelBiases = append(modelBiases, f64biases)

			layer.Neurons = neurons

			fmt.Printf("Gemm layer found with weights: %v and bias: %v\n", weights, bias)

			denseEncountered = true

		case "Relu":

			fmt.Println("ReLU activation")
			if denseEncountered {
				layer.ActivationFunction = "ReLU"
				denseEncountered = false
				layers = append(layers, layer)
			}

		case "Softmax":

			fmt.Println("Softmax activation")

			if denseEncountered {
				layer.ActivationFunction = "Softmax"
				denseEncountered = false
				layers = append(layers, layer)
			}

		default:
			fmt.Printf("Unsupported ONNX operator: %s\n", *node.OpType)
			continue

		}

		fmt.Println(layer.Neurons)
		fmt.Println(layer.ActivationFunction)
		fmt.Println(layer.Initialization)
		// fmt.Println(layer.Dropout)
	}

	fmt.Println(modelWeights)
	fmt.Println(modelBiases)

	// Convert last layer to output layer

	outputLayer := nn.OutputLayer{
		Neurons:            layers[len(layers)-1].Neurons,
		ActivationFunction: layers[len(layers)-1].ActivationFunction,
	}

	if len(layers) > 0 {
		layers = layers[:len(layers)-1]
	}

	model := &nn.Model{
		NeuralNetwork: nn.NeuralNetwork{
			Layers:      layers,
			OutputLayer: outputLayer,
			InputLayer:  inputLayer,
		},
	}

	model.NeuralNetwork.WeightsAndBiases.Weights = modelWeights
	model.NeuralNetwork.WeightsAndBiases.Biases = modelBiases

	return model, nil

}

func decodeTensorFloat32(t *TensorProto) []float32 {

	raw := t.RawData
	count := len(raw) / 4

	out := make([]float32, count)

	for i := 0; i < count; i++ {
		bits := binary.LittleEndian.Uint32(raw[i*4:])
		out[i] = math.Float32frombits(bits)
	}

	return out
}
