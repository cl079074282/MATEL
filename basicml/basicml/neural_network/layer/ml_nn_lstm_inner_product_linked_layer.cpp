#include "stdafx.h"
#include "ml_nn_lstm_inner_product_linked_layer.h"

void ml_nn_lstm_inner_product_linked_layer::feedforward(const ml_nn_layer_learning_params& pars) {

}

void ml_nn_lstm_inner_product_linked_layer::feedforward(ml_nn_lstm_info& info, const Mat& input_data, const vector<Range>& seq_ranges, bool forward) {
	ml_mat_op::cpu_mul(info.m_cin_activation_without_gate_cache, input_data, info.m_in_to_cin_weight);
	ml_mat_op::self_add_row(info.m_cin_activation_without_gate_cache, info.m_cin_bias);

	ml_mat_op::cpu_mul(info.m_igate.m_gate_activate_cache, input_data, info.m_igate.m_in_to_gate_weight);
	ml_mat_op::self_add_row(info.m_igate.m_gate_activate_cache, info.m_igate.m_gate_bias);

	ml_mat_op::cpu_mul(info.m_fgate.m_gate_activate_cache, input_data, info.m_fgate.m_in_to_gate_weight);
	ml_mat_op::self_add_row(info.m_fgate.m_gate_activate_cache, info.m_fgate.m_gate_bias);

	ml_mat_op::cpu_mul(info.m_ogate.m_gate_activate_cache, input_data, info.m_ogate.m_in_to_gate_weight);
	ml_mat_op::self_add_row(info.m_ogate.m_gate_activate_cache, info.m_ogate.m_gate_bias);

	if (!info.m_igate.m_enable_gate) {
		info.m_cin_activation_cache = info.m_cin_activation_without_gate_cache;
	}

	if (!info.m_ogate.m_enable_gate) {
		info.m_cout_activation_cache = info.m_cout_activation_without_gate_cache;
	}

	for (int i = 0; i < (int)seq_ranges.size(); ++i) {
		Mat temp_block;	//temp memory
		Mat temp_cell;	//temp memory

		//The first input of a sequence has no previous ff signal, so we direct compute the activate value. 
		int start_index = forward ? seq_ranges[i].start : seq_ranges[i].end - 1;
		int stop_index = forward ? seq_ranges[i].end : seq_ranges[i].start - 1;
		int step = forward ? 1 : 0;	//+ step indicates next while - step indicates previous

		for (int row = start_index; row != stop_index; row += step) {
			Mat prev_cout_activation = info.m_cout_activation_cache.row(row - step);
			Mat prev_cout_signal = info.m_cout_signal_cache.row(row - step);

			Mat cur_igate_activation;
			Mat cur_fgate_activation;
			Mat cur_ogate_activation;

			if (info.m_igate.m_enable_gate) {
				//calculate activation on input gate
				cur_igate_activation = info.m_igate.m_gate_activate_cache.row(row);
				calculate_nonfirst_gate_activation(cur_igate_activation, temp_block, prev_cout_activation, prev_cout_signal, info.m_igate, row == start_index, info.m_block_number, info.m_cell_pre_block_number);
			}
			
			if (info.m_fgate.m_enable_gate) {
				//calculate activation on forget gate
				cur_fgate_activation = info.m_fgate.m_gate_activate_cache.row(row);
				calculate_nonfirst_gate_activation(cur_fgate_activation, temp_block, prev_cout_activation, prev_cout_signal, info.m_fgate, row == start_index, info.m_block_number, info.m_cell_pre_block_number);
			}
			
			if (info.m_ogate.m_enable_gate) {
				//calculate activation on output gate
				cur_ogate_activation = info.m_ogate.m_gate_activate_cache.row(row);
				calculate_nonfirst_gate_activation(cur_ogate_activation, temp_block, prev_cout_activation, prev_cout_signal, info.m_ogate, row == start_index, info.m_block_number, info.m_cell_pre_block_number);
			}
			
			Mat cur_cout_signal = info.m_cout_signal_cache.row(row);
			Mat cur_cin_activation_without_gate = info.m_cin_activation_without_gate_cache.row(row);
			Mat cur_cin_activation = info.m_cin_activation_cache.row(row);

			//calculate activation on cell input
			ml_mat_op::cpu_mul(temp_cell, prev_cout_activation, info.m_cout_to_cin_weight);
			cur_cin_activation_without_gate += temp_cell;
			ml_define::activate(cur_cin_activation_without_gate, cur_cin_activation_without_gate, info.m_cin_activate_type, info.m_cin_activate_params);
			
			if (info.m_igate.m_enable_gate) {
				multiply_gate_activation(cur_cin_activation, cur_cin_activation_without_gate, cur_igate_activation, info.m_block_number, info.m_cell_pre_block_number);
			}
		
			//calculate signal of cell out
			prev_cout_signal.copyTo(cur_cout_signal);

			if (info.m_fgate.m_enable_gate) {
				multiply_gate_activation(cur_cout_signal, cur_cout_signal, cur_fgate_activation, info.m_block_number, info.m_cell_pre_block_number);
			}
			
			cur_cout_signal += cur_cin_activation;

			//calculate activation of cell output
			Mat cur_cout_without_gate_activation = info.m_cout_activation_without_gate_cache.row(row);
			Mat cur_cout_activation = info.m_cout_activation_cache.row(row);
			ml_define::activate(cur_cout_without_gate_activation, cur_cout_signal, info.m_cout_activate_type, info.m_cout_activate_params);

			if (info.m_ogate.m_enable_gate) {
				multiply_gate_activation(cur_cout_activation, cur_cout_without_gate_activation, cur_ogate_activation, info.m_block_number, info.m_cell_pre_block_number);
			}
		}
	}

	ml_mat_op::cpu_mul(info.m_out_activation_without_bias_cache, info.m_cout_activation_cache, info.m_cout_to_out_weight);
}

void ml_nn_lstm_inner_product_linked_layer::backpropagation(ml_nn_lstm_info& info, const Mat& input_data, const Mat& next_delta, const vector<Range>& seq_ranges, bool forward) {
	ml_mat_op::cpu_mul(info.m_cout_epsilon_cache, next_delta, info.m_cout_to_out_weight, false, true);
		
	int pooling_size[] = {1, info.m_cell_pre_block_number};

	for (int i = 0; i < (int)seq_ranges.size(); ++i) {
		Mat temp_block;
		Mat temp_cell;

		int start_index = forward ? seq_ranges[i].end - 1 : seq_ranges[i].start;
		int end_index = forward ? seq_ranges[i].start - 1 : seq_ranges[i].end;
		int step = forward ? -1 : 1;//+ step indicates previous while - step indicates next
	
		for (int row = start_index; row != end_index; row += step) {
			//calculate epsilon on cout
			int next_row = row - step;

			Mat& cur_cout_epsilon = info.m_cout_epsilon_cache.row(row);

			if (row != start_index) {
				Mat& next_cin_delta = info.m_cin_delta_cache.row(next_row);
				ml_mat_op::cpu_mul(temp_cell, next_cin_delta, info.m_cout_to_cin_weight, false, true);

				cur_cout_epsilon += temp_cell;

				if (info.m_ogate.m_enable_gate) {
					ml_mat_op::cpu_mul(temp_cell, info.m_ogate.m_gate_delta_cache.row(next_row), info.m_ogate.m_cout_to_gate_weight, false, true);
					cur_cout_epsilon += temp_cell;

					ml_mat_op::cpu_mul(temp_cell, info.m_fgate.m_gate_delta_cache.row(next_row), info.m_fgate.m_cout_to_gate_weight, false, true);
					cur_cout_epsilon += temp_cell;

					ml_mat_op::cpu_mul(temp_cell, info.m_ogate.m_gate_delta_cache.row(next_row), info.m_ogate.m_cout_to_gate_weight, false, true);
					cur_cout_epsilon += temp_cell;
				}
			}

			//calculate delta on cout
			Mat cur_cout_delta = info.m_cout_delta_cache.row(row);
			Mat cur_cout_without_gate_activation = info.m_cout_activation_without_gate_cache.row(row);
			ml_define::activate_derivative(temp_block, cur_cout_without_gate_activation, info.m_cout_activate_type, info.m_cout_activate_params);

			multiply(cur_cout_epsilon, temp_block, cur_cout_delta);

			if (info.m_ogate.m_enable_gate) {
				Mat back_ogate_activation = info.m_ogate.m_gate_activate_cache.row(start_index);
				multiply_gate_activation(cur_cout_delta, cur_cout_delta, back_ogate_activation, info.m_block_number, info.m_cell_pre_block_number);
			}

			//calculate delta on cell
			Mat cur_cell_delta = info.m_cell_delta_cache.row(row);
			cur_cout_delta.copyTo(cur_cell_delta);

			if (row != start_index) {
				multiply_gate_activation(temp_cell, info.m_cell_delta_cache.row(next_row), info.m_igate.m_gate_activate_cache.row(next_row), info.m_block_number, info.m_cell_pre_block_number);

				cur_cell_delta += temp_cell;
				
				if (info.m_igate.m_enable_gate && info.m_igate.m_enable_peep_connection) {
					gate_multiply_cell_gate_weight(temp_cell, info.m_igate.m_gate_delta_cache.row(next_row), info.m_igate.m_v_cout_to_gate_weight, info.m_block_number, info.m_cell_pre_block_number);
					cur_cell_delta += temp_cell;
				}

				if (info.m_fgate.m_enable_gate && info.m_fgate.m_enable_peep_connection) {
					gate_multiply_cell_gate_weight(temp_cell, info.m_fgate.m_gate_delta_cache.row(next_row), info.m_fgate.m_v_cout_to_gate_weight, info.m_block_number, info.m_cell_pre_block_number);
					cur_cell_delta += temp_cell;
				}

				if (info.m_ogate.m_enable_gate && info.m_ogate.m_enable_peep_connection) {
					gate_multiply_cell_gate_weight(temp_cell, info.m_ogate.m_gate_delta_cache.row(next_row), info.m_ogate.m_v_cout_to_gate_weight, info.m_block_number, info.m_cell_pre_block_number);
					cur_cell_delta += temp_cell;
				}
			}

			//calculate delta on output gate
			if (info.m_ogate.m_enable_gate) {
				Mat cur_ogate_delta = info.m_ogate.m_gate_delta_cache.row(row);
				multiply(cur_cout_epsilon, info.m_cout_activation_without_gate_cache.row(row), temp_cell);
				ml_mat_op::ml_pooling(temp_block, Mat(), temp_cell, ml_mat_op::pooling_type_kernel_sum, pooling_size, pooling_size);
				ml_define::activate_derivative(cur_ogate_delta, info.m_ogate.m_gate_activate_cache.row(row), info.m_ogate.m_gate_activate_type, info.m_ogate.m_gate_activate_params);

				multiply(cur_ogate_delta, temp_block, cur_ogate_delta);
			}

			//calculate delta on forget gate
			if (info.m_fgate.m_enable_gate) {
				Mat cur_fgate_delta = info.m_fgate.m_gate_delta_cache.row(row);
				if (row != end_index + step) {
					multiply(cur_cell_delta, info.m_cout_signal_cache.row(row + step), temp_cell);
					ml_mat_op::ml_pooling(temp_block, Mat(), temp_cell, ml_mat_op::pooling_type_kernel_sum, pooling_size, pooling_size);
					ml_define::activate_derivative(cur_fgate_delta, info.m_fgate.m_gate_activate_cache.row(row), info.m_ogate.m_gate_activate_type, info.m_ogate.m_gate_activate_params);

					multiply(cur_fgate_delta, temp_block, cur_fgate_delta);
				} else {
					cur_fgate_delta *= 0;
				}
			}

			//calculate delta on input gate
			if (info.m_igate.m_enable_gate) {
				Mat cur_igate_delta = info.m_igate.m_gate_delta_cache.row(row);
				multiply(cur_cell_delta, info.m_cin_activation_without_gate_cache.row(row), temp_cell);
				ml_mat_op::ml_pooling(temp_block, Mat(), temp_cell, ml_mat_op::pooling_type_kernel_sum, pooling_size, pooling_size);
				ml_define::activate_derivative(cur_igate_delta, info.m_igate.m_gate_activate_cache.row(row), info.m_igate.m_gate_activate_type, info.m_igate.m_gate_activate_params);

				multiply(cur_igate_delta, temp_block, cur_igate_delta);
			}

			//calculate delta on cell input
			Mat cur_cin_delta = info.m_cin_delta_cache.row(start_index);
			ml_define::activate_derivative(temp_block, info.m_cin_activation_without_gate_cache.row(row), info.m_cin_activate_type, info.m_cin_activate_params);
			multiply(temp_block, cur_cell_delta, cur_cin_delta);

			if (info.m_igate.m_enable_gate) {
				Mat cur_igate_activation = info.m_igate.m_gate_activate_cache.row(start_index);
				multiply_gate_activation(cur_cin_delta, cur_cin_delta, cur_igate_activation, info.m_block_number, info.m_cell_pre_block_number);
			}
		}
	}

	//Here we compute the gradient for learned params
	//calculate cout_to_out_weight
	ml_mat_op::cpu_mul(info.m_d_cout_to_out_weight, info.m_cout_delta_cache, next_delta, false, true);
	info.m_d_cout_to_out_weight /= next_delta.rows;


	//calculate input_to_cin_weight
	ml_mat_op::cpu_mul(info.m_d_in_to_cin_weight, input_data, info.m_cin_delta_cache, false, true);
	info.m_d_in_to_cin_weight /= next_delta.rows;
	ml_mat_op::mean_reduce(info.m_d_cin_bias, info.m_cin_delta_cache, 0);

	//calculate cout_to_cin_weight, cout_to_igate_weight, cout_to_fgate_weight, cout_to_ogate_weight, 
	//cell_to_igate_weight, cell_to_fgate_weight, cell_to_ogate_weight
	int number = 0;

	for (int i = 0; i < (int)seq_ranges.size(); ++i) {
		int start_index = forward ? seq_ranges[i].end - 1 : seq_ranges[i].start;
		int end_index = forward ? seq_ranges[i].start: seq_ranges[i].end - 1;
		int step = forward ? -1 : 1;

		for (int row = start_index; row != end_index; row += step) {
			int prev_row = row + step;

			if (number == 0) {
				ml_mat_op::cpu_mul(info.m_d_cout_to_cin_weight, info.m_cout_activation_cache.row(prev_row), info.m_cin_delta_cache.row(row), true, false);

				if (info.m_igate.m_enable_gate) {
					ml_mat_op::cpu_mul(info.m_igate.m_cout_to_gate_weight, info.m_cout_activation_cache.row(prev_row), info.m_igate.m_cout_to_gate_weight.row(row), true, false);

					if (info.m_igate.m_enable_peep_connection) {
						gate_multiply_cell(info.m_igate.m_cell_to_gate_weight, info.m_igate.m_gate_delta_cache.row(row), info.m_cout_signal_cache.row(prev_row), info.m_block_number, info.m_cell_pre_block_number);
					}
				}

				if (info.m_fgate.m_enable_gate) {
					ml_mat_op::cpu_mul(info.m_fgate.m_cout_to_gate_weight, info.m_cout_activation_cache.row(prev_row), info.m_fgate.m_cout_to_gate_weight.row(row), true, false);

					if (info.m_fgate.m_enable_peep_connection) {
						gate_multiply_cell(info.m_fgate.m_cell_to_gate_weight, info.m_fgate.m_gate_delta_cache.row(row), info.m_cout_signal_cache.row(prev_row), info.m_block_number, info.m_cell_pre_block_number);
					}
				}

				if (info.m_ogate.m_enable_gate) {
					ml_mat_op::cpu_mul(info.m_ogate.m_cout_to_gate_weight, info.m_cout_activation_cache.row(prev_row), info.m_ogate.m_cout_to_gate_weight.row(row), true, false);

					if (info.m_ogate.m_enable_peep_connection) {
						gate_multiply_cell(info.m_ogate.m_cell_to_gate_weight, info.m_ogate.m_gate_delta_cache.row(row), info.m_cout_signal_cache.row(prev_row), info.m_block_number, info.m_cell_pre_block_number);
					}
				}
			} else {
				ml_mat_op::cpu_mul(info.m_d_cout_to_cin_weight_temp, info.m_cout_activation_cache.row(prev_row), info.m_cin_delta_cache.row(row), true, false);
				info.m_d_cout_to_cin_weight += info.m_d_cout_to_cin_weight_temp;

				if (info.m_igate.m_enable_gate) {
					ml_mat_op::cpu_mul(info.m_d_cout_to_gate_weight_temp, info.m_cout_activation_cache.row(prev_row), info.m_igate.m_cout_to_gate_weight.row(row), true, false);
					info.m_igate.m_cout_to_gate_weight += info.m_d_cout_to_gate_weight_temp;

					if (info.m_igate.m_enable_peep_connection) {
						gate_multiply_cell(info.m_d_cell_to_gate_weight_temp, info.m_igate.m_gate_delta_cache.row(row), info.m_cout_signal_cache.row(prev_row), info.m_block_number, info.m_cell_pre_block_number);
						info.m_igate.m_cell_to_gate_weight += info.m_d_cell_to_gate_weight_temp;
					}
				}

				if (info.m_fgate.m_enable_gate) {
					ml_mat_op::cpu_mul(info.m_d_cout_to_gate_weight_temp, info.m_cout_activation_cache.row(prev_row), info.m_fgate.m_cout_to_gate_weight.row(row), true, false);
					info.m_fgate.m_cout_to_gate_weight += info.m_d_cout_to_gate_weight_temp;

					if (info.m_fgate.m_enable_peep_connection) {
						gate_multiply_cell(info.m_d_cell_to_gate_weight_temp, info.m_fgate.m_gate_delta_cache.row(row), info.m_cout_signal_cache.row(prev_row), info.m_block_number, info.m_cell_pre_block_number);
						info.m_fgate.m_cell_to_gate_weight += info.m_d_cell_to_gate_weight_temp;
					}
				}

				if (info.m_ogate.m_enable_gate) {
					ml_mat_op::cpu_mul(info.m_d_cout_to_gate_weight_temp, info.m_cout_activation_cache.row(prev_row), info.m_ogate.m_cout_to_gate_weight.row(row), true, false);
					info.m_ogate.m_cout_to_gate_weight += info.m_d_cout_to_gate_weight_temp;

					if (info.m_ogate.m_enable_peep_connection) {
						gate_multiply_cell(info.m_d_cell_to_gate_weight_temp, info.m_ogate.m_gate_delta_cache.row(row), info.m_cout_signal_cache.row(prev_row), info.m_block_number, info.m_cell_pre_block_number);
						info.m_ogate.m_cell_to_gate_weight += info.m_d_cell_to_gate_weight_temp;
					}
				}
			}

			++number;
		}
	}

	BASICLOG_ASSERT2(number = input_data.rows - (int)seq_ranges.size()); 
	info.m_d_cout_to_cin_weight /= number;
	
	if (info.m_igate.m_enable_gate) {
		info.m_igate.m_cout_to_gate_weight /= number;
		ml_mat_op::mean_reduce(info.m_igate.m_d_gate_bias, info.m_igate.m_gate_delta_cache, 0);

		if (info.m_igate.m_enable_peep_connection) {
			info.m_igate.m_cell_to_gate_weight /= number;
		}
	}

	if (info.m_fgate.m_enable_gate) {
		info.m_fgate.m_cout_to_gate_weight /= number;
		ml_mat_op::mean_reduce(info.m_fgate.m_d_gate_bias, info.m_fgate.m_gate_delta_cache, 0);

		if (info.m_fgate.m_enable_peep_connection) {
			info.m_fgate.m_cell_to_gate_weight /= number;
		}
	}

	if (info.m_ogate.m_enable_gate) {
		info.m_ogate.m_cout_to_gate_weight /= number;
		ml_mat_op::mean_reduce(info.m_ogate.m_d_gate_bias, info.m_ogate.m_gate_delta_cache, 0);

		if (info.m_ogate.m_enable_peep_connection) {
			info.m_ogate.m_cell_to_gate_weight /= number;
		}
	}
}

void ml_nn_lstm_inner_product_linked_layer::calculate_nonfirst_gate_activation(Mat& activation, Mat& temp_block, const Mat& prev_cout_activation, const Mat& prev_cout_signal, const ml_nn_lstm_gate_info& gate, bool first, int block_number, int cell_pre_block_number) {	
	if (!first) {
		ml_mat_op::cpu_mul(temp_block, prev_cout_activation, gate.m_cout_to_gate_weight);
		activation += temp_block;

		if (gate.m_enable_peep_connection) {
			cell_multiply_cell_gate_weight(temp_block, prev_cout_signal, gate.m_cell_to_gate_weight, block_number, cell_pre_block_number);
			activation += temp_block;
		}
	}
		
	ml_define::activate(activation, activation, gate.m_gate_activate_type, gate.m_gate_activate_params);
}

void ml_nn_lstm_inner_product_linked_layer::multiply_gate_activation(Mat& res, const Mat& cell, Mat& gate_activation, int block_number, int cell_pre_block_number) {
	if (cell_pre_block_number == 1) {
		multiply(cell, gate_activation, res);
	} else {
		res.create(cell.rows, cell.cols, cell.type());

		if (CV_64FC1 == cell.type()) {
			uchar* ptr_res_dim0 = res.data;
			uchar* ptr_gate_dim0 = gate_activation.data;
			uchar* ptr_cell_dim0 = cell.data;
			
			for (int row = 0; row < cell.rows; ++row) {
				double* ptr_res_dim1 = (double*)ptr_res_dim0;
				double* ptr_gate_dim1 = (double*)ptr_gate_dim0;
				double* ptr_cell_dim1 = (double*)ptr_cell_dim0;

				for (int iter_block = 0; iter_block < block_number; ++iter_block) {
					for (int iter_cell = 0; iter_cell < cell_pre_block_number; ++iter_cell) {
						*ptr_res_dim1 = *ptr_cell_dim1 * *ptr_gate_dim1;
						ptr_res_dim1 += res.step.p[1];
						ptr_cell_dim1 += cell.step.p[1];
					}

					ptr_gate_dim1 += gate_activation.step.p[1];
				}

				ptr_res_dim0 += res.step.p[0];
				ptr_gate_dim0 += gate_activation.step.p[0];
				ptr_cell_dim0 += cell.step.p[0];
			}
		} else if (CV_32FC1 == cell.type()) {
			uchar* ptr_res_dim0 = res.data;
			uchar* ptr_gate_dim0 = gate_activation.data;
			uchar* ptr_cell_dim0 = cell.data;

			for (int row = 0; row < cell.rows; ++row) {
				float* ptr_res_dim1 = (float*)ptr_res_dim0;
				float* ptr_gate_dim1 = (float*)ptr_gate_dim0;
				float* ptr_cell_dim1 = (float*)ptr_cell_dim0;

				for (int iter_block = 0; iter_block < block_number; ++iter_block) {
					for (int iter_cell = 0; iter_cell < cell_pre_block_number; ++iter_cell) {
						*ptr_res_dim1 = *ptr_cell_dim1 * *ptr_gate_dim1;
						ptr_res_dim1 += res.step.p[1];
						ptr_cell_dim1 += cell.step.p[1];
					}

					ptr_gate_dim1 += gate_activation.step.p[1];
				}
			}
		}
	}
}

void ml_nn_lstm_inner_product_linked_layer::cell_multiply_cell_gate_weight(Mat& res, const Mat& cell, const Mat& cell_gate_weight, int block_number, int cell_pre_block_number) {
	if (cell_pre_block_number == 1) {
		multiply(cell, cell_gate_weight, res);
	} else {
		res.create(1, block_number, cell.type());
		for (int iter_block = 0; iter_block < block_number; ++iter_block) {
			ml_mat_op::cpu_mul(res.col(iter_block), cell.colRange(iter_block * cell_pre_block_number, cell_pre_block_number), cell_gate_weight.row(iter_block), false, true);
		}
	}
}

void ml_nn_lstm_inner_product_linked_layer::gate_multiply_cell_gate_weight(Mat& res, const Mat& gate, const Mat& cell_gate_weight, int block_number, int cell_pre_block_number) {
	if (cell_pre_block_number == 1) {
		multiply(gate, cell_gate_weight, res);
	} else {
		res.create(gate.rows, block_number * cell_pre_block_number, gate.type());

		if (CV_64FC1 == gate.type()) {
			uchar* ptr_res_dim0 = res.data;
			uchar* ptr_cell_gate_weight_dim0 = cell_gate_weight.data;
			uchar* ptr_gate_dim0 = gate.data;

			for (int row = 0; row < gate.rows; ++row) {
				double* ptr_res_dim1 = (double*)ptr_res_dim0;
				double* ptr_gate_dim1 = (double*)ptr_gate_dim0;

				for (int col = 0; col < gate.cols; ++col) {
					double* ptr_cell_gate_weight_dim1 = (double*)ptr_cell_gate_weight_dim0;					
					for (int iter_cell = 0; iter_cell < cell_pre_block_number; ++iter_cell) {
					
						*ptr_res_dim1 = *ptr_gate_dim1 * *ptr_cell_gate_weight_dim1;
						ptr_res_dim1 += res.step.p[1];
						ptr_cell_gate_weight_dim1 += cell_gate_weight.step.p[1];
					}

					ptr_gate_dim1 += gate.step.p[1];
					ptr_cell_gate_weight_dim0 += cell_gate_weight.step.p[0];
				}

				ptr_res_dim0 += res.step.p[0];
				ptr_gate_dim0 += gate.step.p[0];
			}
		} else if (CV_32FC1 == gate.type()) {
			uchar* ptr_res_dim0 = res.data;
			uchar* ptr_cell_gate_weight_dim0 = cell_gate_weight.data;
			uchar* ptr_gate_dim0 = gate.data;

			for (int row = 0; row < gate.rows; ++row) {
				double* ptr_res_dim1 = (double*)ptr_res_dim0;
				double* ptr_gate_dim1 = (double*)ptr_gate_dim0;

				for (int col = 0; col < gate.cols; ++col) {
					double* ptr_cell_gate_weight_dim1 = (double*)ptr_cell_gate_weight_dim0;					
					for (int iter_cell = 0; iter_cell < cell_pre_block_number; ++iter_cell) {

						*ptr_res_dim1 = *ptr_gate_dim1 * *ptr_cell_gate_weight_dim1;
						ptr_res_dim1 += res.step.p[1];
						ptr_cell_gate_weight_dim1 += cell_gate_weight.step.p[1];
					}

					ptr_gate_dim1 += gate.step.p[1];
					ptr_cell_gate_weight_dim0 += cell_gate_weight.step.p[0];
				}

				ptr_res_dim0 += res.step.p[0];
				ptr_gate_dim0 += gate.step.p[0];
			}	
		}
	}
}

void ml_nn_lstm_inner_product_linked_layer::gate_multiply_cell(Mat& cell_to_gate_weight, const Mat& gate, const Mat& cell, int block_number, int cell_pre_block_number) {
	if (cell_pre_block_number == 1) {
		ml_mat_op::cpu_mul(cell_to_gate_weight, cell, gate, true, false);
	} else {
		cell_to_gate_weight.create(block_number, cell_pre_block_number, cell.type());
		ml_mat_op::set(cell_to_gate_weight, Scalar(0));

		if (CV_64FC1 == gate.type()) {
			uchar* ptr_cell_gate_weight_dim0 = cell_to_gate_weight.data;
			
			for (int iter_sample = 0; iter_sample < gate.rows; ++iter_sample) {
				for (int row = 0; row < cell_to_gate_weight.rows; ++row) {
					double* ptr_cell_gate_weight_dim1 = (double*)ptr_cell_gate_weight_dim0;
					for (int col = 0; col < cell_to_gate_weight.cols; ++col) {
						*ptr_cell_gate_weight_dim1 += cell.at<double>(iter_sample, row * cell_pre_block_number + col) * gate.at<double>(iter_sample, row);
					}

					ptr_cell_gate_weight_dim0 += cell_to_gate_weight.step.p[0];
				}
			}
		} else if (CV_32FC1 == gate.type()) {
			uchar* ptr_cell_gate_weight_dim0 = cell_to_gate_weight.data;

			for (int iter_sample = 0; iter_sample < gate.rows; ++iter_sample) {
				for (int row = 0; row < cell_to_gate_weight.rows; ++row) {
					float* ptr_cell_gate_weight_dim1 = (float*)ptr_cell_gate_weight_dim0;
					for (int col = 0; col < cell_to_gate_weight.cols; ++col) {
						*ptr_cell_gate_weight_dim1 += cell.at<float>(iter_sample, row * cell_pre_block_number + col) * gate.at<float>(iter_sample, row);
					}

					ptr_cell_gate_weight_dim0 += cell_to_gate_weight.step.p[0];
				}
			}
		}
	}
}