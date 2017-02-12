#include "stdafx.h"
#include "ml_nn_bptt_inner_product_linked_layer.h"
#include "ml_nn_data_layer.h"


void ml_nn_bptt_inner_product_linked_layer::init_need_learn_params(int data_type) {

}

void ml_nn_bptt_inner_product_linked_layer::feedforward(const ml_nn_layer_learning_params& pars) {
	const Mat& input_data = m_input->to_data_layer()->get_data(0);

	bptt_forward(m_info, input_data, pars, true);
	ml_mat_op::self_add_row(m_info.m_ff_signal, m_output_bias);

	m_output->to_data_layer()->feedforward_singal(m_info.m_ff_signal, pars);
}

void ml_nn_bptt_inner_product_linked_layer::backpropagation(const ml_nn_layer_learning_params& pars) {
	const Mat& prev_data = m_input->to_data_layer()->get_data(0);
	const Mat& next_delta = m_output->to_data_layer()->get_delta(0);
	
	bptt_backpropagation(m_info,
		prev_data,
		next_delta,
		pars,
		true);
	
	ml_mat_op::mean_reduce(m_d_output_bias, next_delta, 0);

	//ml_define::update_learned_param(m_info.m_input_inner_weight, m_info.m_v_input_inner_weight, m_info.m_d_input_inner_weight, m_weight_update_setting.m_momentum, m_weight_update_setting.m_learning_rate);
	//ml_define::update_learned_param(m_info.m_inner_inner_weight, m_info.m_v_inner_inner_weight, m_info.m_d_inner_inner_weight, m_weight_update_setting.m_momentum, m_weight_update_setting.m_learning_rate);
	//ml_define::update_learned_param(m_info.m_inner_output_weight, m_info.m_v_inner_output_weight, m_info.m_d_inner_output_weight, m_weight_update_setting.m_momentum, m_weight_update_setting.m_learning_rate);
	//ml_define::update_learned_param(m_info.m_inner_bias, m_info.m_v_inner_bias, m_info.m_d_inner_bias, m_bias_update_setting.m_momentum, m_bias_update_setting.m_learning_rate);
	//ml_define::update_learned_param(m_output_bias, m_v_output_bias, m_d_output_bias, m_bias_update_setting.m_momentum, m_bias_update_setting.m_learning_rate);

	m_input->to_data_layer()->backprapogation_singal(m_info.m_bp_signal, pars);
}

ml_nn_layer* ml_nn_bptt_inner_product_linked_layer::clone() const {
	ml_nn_bptt_inner_product_linked_layer* layer = new ml_nn_bptt_inner_product_linked_layer();
	layer->m_input = m_input;
	layer->m_output = m_output;

	return layer;
}


void ml_nn_bptt_inner_product_linked_layer::bptt_forward(
	ml_nn_bptt_info& info,
	const Mat& input, 
	const ml_nn_layer_learning_params& pars,
	bool forward) {
		ml_mat_op::cpu_mul(info.m_inner_activate, input, info.m_input_inner_weight, false, false, pars.m_number_of_calculation);
		ml_mat_op::self_add_row(info.m_inner_activate, info.m_inner_bias);

		for (int i = 0; i < (int)pars.m_seq_ranges.size(); ++i) {
			Mat temp_inner_ff_signal;

			//The first input of a sequence has no previous ff signal, so we direct compute the activate value. 
			int start_index = forward ? pars.m_seq_ranges[i].start : pars.m_seq_ranges[i].end - 1;
			int stop_index = forward ? pars.m_seq_ranges[i].end : pars.m_seq_ranges[i].start - 1;
			int step = forward ? 1 : -1;	//+ step indicates next while - step indicates previous

			Mat seq_first_activate = info.m_inner_activate.row(start_index);
			ml_define::activate(seq_first_activate, seq_first_activate, info.m_inner_activate_type, info.m_inner_activate_params);

			for (int row = start_index + step; row != stop_index; row += step) {
				//First we need to calculate the ff signal of inner to inner direction.
				Mat prev_activate = info.m_inner_activate.row(row - step);
				Mat cur_activate = info.m_inner_activate.row(row);

				ml_mat_op::cpu_mul(temp_inner_ff_signal, prev_activate, info.m_inner_inner_weight);

				//Add previous ff signal of inner to inner direction to the ff signal (cur_activate) of input to inner direction.
				cur_activate += temp_inner_ff_signal;
				ml_define::activate(cur_activate, cur_activate, info.m_inner_activate_type, info.m_inner_activate_params);
			}
		}

		ml_mat_op::cpu_mul(info.m_ff_signal, info.m_inner_activate, info.m_inner_output_weight, false, false, pars.m_number_of_calculation);
}

void ml_nn_bptt_inner_product_linked_layer::bptt_backpropagation(
	ml_nn_bptt_info& info,
	const Mat& input_data,
	const Mat& out_delta,
	const ml_nn_layer_learning_params& pars,
	bool forward) {		
		ml_mat_op::cpu_mul(info.m_inner_delta, out_delta, info.m_inner_output_weight, false, true);
		ml_define::activate_derivative(info.m_inner_activate_derivative, info.m_inner_activate, info.m_inner_activate_type, info.m_inner_activate_params);

		for (int i = 0; i < (int)pars.m_seq_ranges.size(); ++i) {
			Mat temp_inner_bp_singal;
			int start_index = forward ? pars.m_seq_ranges[i].end - 1 : pars.m_seq_ranges[i].start;
			int end_index = forward ? pars.m_seq_ranges[i].start - 1 : pars.m_seq_ranges[i].end;
			int step = forward ? -1 : 1;//+ step indicates previous while - step indicates next

			Mat back_inner_delta = info.m_inner_delta.row(start_index);
			multiply(back_inner_delta, info.m_inner_activate_derivative.row(start_index), back_inner_delta);		

			for (int row = start_index + step; row != end_index; row += step) {
				Mat next_delta = info.m_inner_delta.row(row - step);
				Mat cur_delta = info.m_inner_delta.row(row);

				ml_mat_op::cpu_mul(temp_inner_bp_singal, next_delta, info.m_inner_inner_weight, false, true);
				cur_delta += temp_inner_bp_singal;
				multiply(cur_delta, info.m_inner_activate_derivative.row(row), cur_delta);
			}
		}

		ml_mat_op::cpu_mul(info.m_bp_signal, info.m_inner_delta, info.m_input_inner_weight, false, true);

		//update weight and bias
		ml_mat_op::cpu_mul(info.m_d_inner_output_weight, info.m_inner_activate, out_delta, true, false);
		info.m_d_inner_output_weight /= info.m_inner_activate.size[0];

		ml_mat_op::cpu_mul(info.m_d_input_inner_weight, input_data, info.m_inner_delta, true, false);
		info.m_d_input_inner_weight /= input_data.size[0];
		ml_mat_op::mean_reduce(info.m_d_inner_bias, out_delta, 0);

		int number = 0;

		for (int i = 0; i < (int)pars.m_seq_ranges.size(); ++i) {
			int start_index = forward ? pars.m_seq_ranges[i].end - 1 : pars.m_seq_ranges[i].start;
			int end_index = forward ? pars.m_seq_ranges[i].start: pars.m_seq_ranges[i].end - 1;
			int step = forward ? -1 : 1;

			for (int row = start_index; row != end_index; row += step) {

				if (number == 0) {
					ml_mat_op::cpu_mul(info.m_d_inner_inner_weight, info.m_inner_activate.row(row + step), info.m_inner_delta.row(row), true, false);
				} else {
					ml_mat_op::cpu_mul(info.m_temp_d_inner_inner_weight, info.m_inner_activate.row(row + step), info.m_inner_delta.row(row), true, false);
					info.m_d_inner_inner_weight += info.m_temp_d_inner_inner_weight;
				}

				++number;
			}
		}

		BASICLOG_ASSERT2(number = input_data.rows - (int)pars.m_seq_ranges.size()); 
		info.m_d_inner_inner_weight /= number;
}