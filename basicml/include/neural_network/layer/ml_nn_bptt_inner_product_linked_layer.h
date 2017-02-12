#pragma once

#include "ml_nn_linked_layer.h"

namespace basicml {

	class ml_nn_bptt_info {
	public:

		Mat m_input_inner_weight;
		Mat m_inner_bias;
		Mat m_inner_inner_weight;
		Mat m_inner_output_weight;
		
		Mat m_v_input_inner_weight;
		Mat m_v_inner_bias;
		Mat m_v_inner_inner_weight;
		Mat m_v_inner_output_weight;

		Mat m_d_input_inner_weight;
		Mat m_d_inner_bias;
		Mat m_d_inner_inner_weight;
		Mat m_temp_d_inner_inner_weight;
		Mat m_d_inner_output_weight;

		Mat m_inner_activate;
		Mat m_inner_activate_derivative;

		Mat m_inner_delta;

		Mat m_ff_signal;
		Mat m_bp_signal;
		int m_inner_activate_type;
		vector<double> m_inner_activate_params;
	};

	class ml_nn_bptt_inner_product_linked_layer : public ml_nn_linked_layer {
	public:
		
		virtual ~ml_nn_bptt_inner_product_linked_layer() {}

		void set_hidden_layer_unit_number(int unit_number);

		void set_drop_out_ratio(double ratio) {
			m_drop_out_ratio = ratio;
		}

		void set_hidden_layer_activate_func_type(int activate_func_type, const vector<double>& activate_pars = vector<double>()) {
			m_info.m_inner_activate_type = activate_func_type;
			m_info.m_inner_activate_params = activate_pars;
		}

		virtual void init_need_learn_params(int data_type);

		virtual void feedforward(const ml_nn_layer_learning_params& pars);
		virtual void backpropagation(const ml_nn_layer_learning_params& pars);

		virtual ml_nn_layer* clone() const;

		/** Forward pass of bptt linked layer.

		@param forward it is true for normal bptt layer. When the bptt layer is bidirectional, the forward pass according to forward inner layer should
		set forward to true, otherwise (backward inner layer) is false.
		@note To implement the bidirectional bptt, here the info.m_ff_signal does not add the bias item.
		*/
		static void bptt_forward(
			ml_nn_bptt_info& info,
			const Mat& input, 
			const ml_nn_layer_learning_params& pars,
			bool forward);

		/**

		@note It does not update the learned parameters, it only calculate the back propagation signal and the gradient of learned parameters.
		*/
		static void bptt_backpropagation(
			ml_nn_bptt_info& info,
			const Mat& input_data,
			const Mat& out_delta,
			const ml_nn_layer_learning_params& pars,
			bool forward);

	private:

		ml_nn_bptt_info m_info;
		Mat m_output_bias;
		Mat m_v_output_bias;
		Mat m_d_output_bias;

		double m_drop_out_ratio;

		/*Mat m_input_inner_weight;
		Mat m_inner_bias;
		Mat m_inner_inner_weight;
		Mat m_inner_output_weight;
		Mat m_output_bias;

		Mat m_v_input_inner_weight;
		Mat m_v_inner_bias;
		Mat m_v_inner_inner_weight;
		Mat m_v_inner_output_weight;
		Mat m_v_output_bias;

		Mat m_d_input_inner_weight;
		Mat m_d_inner_bias;
		Mat m_d_inner_inner_weight;
		Mat m_d_inner_inner_weight_temp;
		Mat m_d_inner_output_weight;
		Mat m_d_output_bias;

		Mat m_inner_activates_cache;
		Mat m_inner_activate_derivative;
		Mat m_ff_signal_cache;
		Mat m_bp_signal_cache;

		Mat m_inner_delta_cache;
		int m_activate_func_type;
		vector<double> m_activate_params;*/
	};
}