package gonn

func Sequential(layers ...Layer) *Model {

	model:= &Model{
		Layers: layers,
	}

	// Initialize weights for layers


	return model
}
