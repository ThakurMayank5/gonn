package tensor

func shapesEqual(shape1, shape2 []int) bool {
	if len(shape1) != len(shape2) {
		return false
	}
	for i := range shape1 {
		if shape1[i] != shape2[i] {
			return false
		}
	}
	return true
}

func Add(a, b Tensor) Tensor {
	if !shapesEqual(a.Shape, b.Shape) {
		panic("tensors must have the same shape for addition")
	}

	result := make([]float64, len(a.Data))
	for i := range a.Data {

		result[i] = a.Data[i] + b.Data[i]

	}

	return Tensor{
		Data:  result,
		Shape: a.Shape,
	}
}

func Mul(a, b Tensor) Tensor {
	if !shapesEqual(a.Shape, b.Shape) {
		panic("tensors must have the same shape for multiplication")
	}

	result := make([]float64, len(a.Data))
	for i := range a.Data {
		result[i] = a.Data[i] * b.Data[i]
	}

	return Tensor{
		Data:  result,
		Shape: a.Shape,
	}
}
