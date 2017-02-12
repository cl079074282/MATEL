#include "stdafx.h"
#include "ml_nn_bidirectional_bptt_inner_product_linked_layer.h"
#include "ml_nn_data_layer.h"

void ml_nn_bidirectional_bptt_inner_product_linked_layer::feedforward(const ml_nn_layer_learning_params& pars) {
	const Mat& input_data = m_input->to_data_layer()->get_data(0);

	ml_nn_bptt_inner_product_linked_layer::bptt_forward(m_forward_info,
		input_data, 
		pars,
		true);

	ml_nn_bptt_inner_product_linked_layer::bptt_forward(m_background_info,
		input_data, 
		pars,
		false);

	m_forward_info.m_ff_signal += m_background_info.m_ff_signal;
	ml_mat_op::self_add_row(m_forward_info.m_ff_signal, m_output_bias);

	m_output->to_data_layer()->feedforward_singal(m_forward_info.m_ff_signal, pars);
}

void ml_nn_bidirectional_bptt_inner_product_linked_layer::backpropagation(const ml_nn_layer_learning_params& pars) {
	const Mat& prev_data = m_input->to_data_layer()->get_data(0);
	const Mat& next_delta = m_output->to_data_layer()->get_delta(0);

	ml_nn_bptt_inner_product_linked_layer::bptt_backpropagation(m_forward_info,
		prev_data,
		next_delta,
		pars,
		true);

	//ml_define::update_learned_param(m_forward_info.m_input_inner_weight, m_forward_info.m_v_input_inner_weight, m_forward_info.m_d_input_inner_weight, m_weight_update_setting.m_momentum, m_weight_update_setting.m_learning_rate);
	//ml_define::update_learned_param(m_forward_info.m_inner_inner_weight, m_forward_info.m_v_inner_inner_weight, m_forward_info.m_d_inner_inner_weight, m_weight_update_setting.m_momentum, m_weight_update_setting.m_learning_rate);
	//ml_define::update_learned_param(m_forward_info.m_inner_output_weight, m_forward_info.m_v_inner_output_weight, m_forward_info.m_d_inner_output_weight, m_weight_update_setting.m_momentum, m_weight_update_setting.m_learning_rate);
	//ml_define::update_learned_param(m_forward_info.m_inner_bias, m_forward_info.m_v_inner_bias, m_forward_info.m_d_inner_bias, m_bias_update_setting.m_momentum, m_bias_update_setting.m_learning_rate);
	//
	ml_nn_bptt_inner_product_linked_layer::bptt_backpropagation(m_background_info,
		prev_data,
		next_delta,
		pars,
		true);

	//ml_define::update_learned_param(m_background_info.m_input_inner_weight, m_background_info.m_v_input_inner_weight, m_background_info.m_d_input_inner_weight, m_weight_update_setting.m_momentum, m_weight_update_setting.m_learning_rate);
	//ml_define::update_learned_param(m_background_info.m_inner_inner_weight, m_background_info.m_v_inner_inner_weight, m_background_info.m_d_inner_inner_weight, m_weight_update_setting.m_momentum, m_weight_update_setting.m_learning_rate);
	//ml_define::update_learned_param(m_background_info.m_inner_output_weight, m_background_info.m_v_inner_output_weight, m_background_info.m_d_inner_output_weight, m_weight_update_setting.m_momentum, m_weight_update_setting.m_learning_rate);
	//ml_define::update_learned_param(m_background_info.m_inner_bias, m_background_info.m_v_inner_bias, m_background_info.m_d_inner_bias, m_bias_update_setting.m_momentum, m_bias_update_setting.m_learning_rate);

	ml_mat_op::mean_reduce(m_d_output_bias, next_delta, 0);
	//ml_define::update_learned_param(m_output_bias, m_v_output_bias, m_d_output_bias, m_bias_update_setting.m_momentum, m_bias_update_setting.m_learning_rate);

	m_forward_info.m_bp_signal += m_background_info.m_bp_signal;
	m_input->to_data_layer()->backprapogation_singal(m_forward_info.m_bp_signal, pars);
}

ml_nn_layer* ml_nn_bidirectional_bptt_inner_product_linked_layer::clone() const {
	ml_nn_bidirectional_bptt_inner_product_linked_layer* layer = new ml_nn_bidirectional_bptt_inner_product_linked_layer();

	return layer;
}