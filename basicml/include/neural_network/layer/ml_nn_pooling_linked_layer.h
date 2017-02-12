#pragma once

#include "ml_nn_linked_layer.h"

namespace basicml {
	class ml_nn_pooling_linked_layer : public ml_nn_linked_layer {
	public:

		virtual ~ml_nn_pooling_linked_layer() {}

		void feedforward(const ml_nn_layer_learning_params& pars);
		void backpropagation(const ml_nn_layer_learning_params& pars);
		ml_nn_layer* clone() const;

		virtual void write_learned_param(ml_file_storage& fs) const {}
		virtual void read_learned_param(const ml_file_node& node) {}

		virtual ml_nn_pooling_linked_layer* to_pooling_linked_layer() {return this;}
		virtual const ml_nn_pooling_linked_layer* to_pooling_linked_layer() const {return this;}

	protected:

		virtual void inner_compute_default_setting();

		vector<Mat> m_ff_singal_caches;
		vector<Mat> m_input_max_masks;
		vector<Mat> m_bp_singal_caches;
	};
}

