package neuralnetwork

import (
	"encoding/gob"
	"os"
)

type ModelParameters struct {
	Weights [][][]float64
	Biases  [][]float64
}

func (model *Model) SaveWeights(path string) error {

	file, err := os.Create(path)
	if err != nil {
		return err
	}
	defer file.Close()

	encoder := gob.NewEncoder(file)

	params := ModelParameters{
		Weights: model.NeuralNetwork.WeightsAndBiases.Weights,
		Biases:  model.NeuralNetwork.WeightsAndBiases.Biases,
	}

	return encoder.Encode(params)
}

func (model *Model) LoadWeights(path string) error {

	file, err := os.Open(path)
	if err != nil {
		return err
	}
	defer file.Close()

	decoder := gob.NewDecoder(file)

	var params ModelParameters

	err = decoder.Decode(&params)
	if err != nil {
		return err
	}

	model.NeuralNetwork.WeightsAndBiases.Weights = params.Weights
	model.NeuralNetwork.WeightsAndBiases.Biases = params.Biases

	return nil
}
