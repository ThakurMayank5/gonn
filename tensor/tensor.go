package tensor

import "fmt"

type Tensor struct {
	Data  []float64
	Shape []int
}

func (t *Tensor) CheckValidTensor() (bool, error) {

	expectedSize := 1
	for _, dim := range t.Shape {
		expectedSize *= dim
	}

	return len(t.Data) == expectedSize, fmt.Errorf("invalid tensor: data length %d does not match expected size %d for shape %v", len(t.Data), expectedSize, t.Shape)
}

func NewTensor(data []float64, shape []int) (*Tensor, error) {

	tensor := Tensor{
		Data:  data,
		Shape: shape,
	}

	isValid, err := tensor.CheckValidTensor()
	if isValid {
		return &tensor, nil
	}

	return nil, err

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
