package tensor

type Tensor struct {
	Data  []float64
	Shape []int
}

func NewTensor(data []float64, shape []int) *Tensor {
	return &Tensor{
		Data:  data,
		Shape: shape,
	}
}

func NewTensorFrom2D(data [][]float64) *Tensor {

	flatData := make([]float64, 0)
	for _, row := range data {
		flatData = append(flatData, row...)
	}

	return &Tensor{
		Data:  flatData,
		Shape: []int{len(data), len(data[0])},
	}
}

func NewTensorFrom3D(data [][][]float64) *Tensor {

	flatData := make([]float64, 0)

	for _, sample := range data {
		for _, row := range sample {
			flatData = append(flatData, row...)
		}
	}

	return &Tensor{
		Data:  flatData,
		Shape: []int{len(data), len(data[0]), len(data[0][0])},
	}
}
