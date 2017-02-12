#pragma once

#include "ml_nn_layer.h"

namespace basicml {
	class ml_neural_network_config;

	class ml_nn_layer_loader {
	public:

		virtual ~ml_nn_layer_loader() {}

		void set_neural_network_config(ml_neural_network_config& config) {
			m_config = &config;
		}

		virtual bool load_layer_config(const ml_file_node& node);

	protected:


		ml_neural_network_config* m_config;
	};

	extern ml_nn_layer_loader* ml_current_layer_config_loader;
}