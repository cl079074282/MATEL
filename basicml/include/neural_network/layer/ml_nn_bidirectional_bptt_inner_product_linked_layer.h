#pragma once

#include "ml_nn_layer.h"
#include "ml_nn_bptt_inner_product_linked_layer.h"

namespace basicml {
	class ml_nn_bidirectional_bptt_inner_product_linked_layer : public ml_nn_linked_layer {
	public:

		void set_hidden_layer_unit_number(int unit_number);

		void set_hidden_layer_activate_func_type(int activate_func_type, const vector<double>& activate_pars = vector<double>()) {
			m_forward_info.m_inner_activate_type = activate_func_type;
			m_background_info.m_inner_activate_type = activate_func_type;
			m_forward_info.m_inner_activate_params = activate_pars;
			m_background_info.m_inner_activate_params = activate_pars;
		}

		virtual void init_need_learn_params(int data_type);

		virtual void feedforward(const ml_nn_layer_learning_params& pars);
		virtual void backpropagation(const ml_nn_layer_learning_params& pars);

		virtual ml_nn_layer* clone() const;

	private:

		ml_nn_bptt_info m_forward_info;
		ml_nn_bptt_info m_background_info;

		Mat m_output_bias;
		Mat m_v_output_bias;
		Mat m_d_output_bias;
	};
}

