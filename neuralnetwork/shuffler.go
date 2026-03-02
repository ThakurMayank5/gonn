package neuralnetwork

import (
	"math/rand"
)

func ShuffleDatasetIndices(n int) (indices []int) {

	indices = make([]int, n)
	for i := 0; i < n; i++ {
		indices[i] = i
	}

	rand.Shuffle(len(indices), func(i, j int) {
		indices[i], indices[j] = indices[j], indices[i]
	})

	return indices
}
