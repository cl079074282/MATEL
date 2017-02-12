#pragma once

namespace basicml {
	
	class ml_nn_layer;
	class ml_nn_layer_config;

	class ml_nn_layer_creator {
	public:

		virtual ~ml_nn_layer_creator() {}

		ml_nn_layer* create(ml_nn_layer_config* config) const;
	};

	extern ml_nn_layer_creator* ml_current_layer_creator;
}