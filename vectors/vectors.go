package vectors

import "fmt"

func DotProduct(a, b []float64) (float64, error) {
	if len(a) != len(b) {
		return 0, fmt.Errorf("vectors must be of the same length")
	}
	dotProduct := 0.0
	for i := range a {
		dotProduct += a[i] * b[i]
	}
	return dotProduct, nil
}
