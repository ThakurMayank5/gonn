package neuralnetwork

func FlattenLayer(input [][][]float64) []float64 {
	flattened := make([]float64, 0)

	for _, sample := range input {
		for _, neuron := range sample {
			flattened = append(flattened, neuron...)
		}
	}

	return flattened
}
