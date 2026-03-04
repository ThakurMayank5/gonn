package onnx

import (
	"fmt"
	"os"

	nn "github.com/ThakurMayank5/gonn/neuralnetwork"
)

func LoadONNXModel(filePath string) (*nn.Model, error) {

	data, err := os.ReadFile(filePath)
	if err != nil {
		return nil, fmt.Errorf("failed to read ONNX file: %v", err)
	}

	fmt.Println(data)

	// For debugging: print the raw bytes in hexadecimal format
	fmt.Printf("Raw ONNX file bytes (%d bytes):\n", len(data))

	// Print the bytes in a readable hexadecimal format
	for i, b := range data {
		fmt.Printf("%02x ", b)
		if (i+1)%16 == 0 {
			fmt.Println()
		}
	}

	return nil, nil

}
