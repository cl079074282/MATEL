#include "stdafx.h"

#include "ml_nn_convolution_linked_layer.h"
#include "ml_nn_data_layer.h"
#include "ml_nn_data_layer_config.h"
#include "ml_nn_convolution_linked_layer_config.h"
#include "ml_random.h"
#include "ml_file_storage.h"

void ml_nn_convolution_linked_layer::feedforward(const ml_nn_layer_learning_params& pars) {
	ml_timer timer(m_config->get_unique_name() + L" feedforward", false);
	timer.begin();

	ml_nn_convolution_linked_layer_config* config = (ml_nn_convolution_linked_layer_config*)m_config;
	ml_nn_data_layer_config* input_config = (ml_nn_data_layer_config*)m_input->get_config();
	ml_nn_data_layer_config* output_config = (ml_nn_data_layer_config*)m_output->get_config();

	const vector<int>& next_image_size = m_output->get_config()->to_data_layer_config()->get_size();

	if (config->is_padded()) {
		m_prev_data_padded_caches.resize(input_config->get_channel());

		for (int iter_input = 0; iter_input < (int)m_prev_data_padded_caches.size(); ++iter_input) {
			ml_mat_op::pand_mat(m_prev_data_padded_caches[iter_input], m_input->to_data_layer()->get_data()[iter_input], &config->get_batch_tl_pad()[0], &config->get_batch_br_pad()[0], Scalar(0));
		}
	}

	const vector<Mat>& prev_data = config->is_padded() ? m_prev_data_padded_caches : m_input->get_data();
	int sample_count = prev_data.front().size.p[0];

	m_ff_singal_caches.resize(output_config->get_channel());

	m_next_batch_sizes.resize(next_image_size.size() + 1);
	m_next_batch_sizes[0] = sample_count;

	for (int i = 0; i < (int)next_image_size.size(); ++i) {
		m_next_batch_sizes[i + 1] = next_image_size[i];
	}

	if (config->is_big_kernel() && m_output->get_config()->to_data_layer_config()->get_channel_feature_size() > 256) {
		m_conv_results.resize(1);

		for (int iter_output = 0; iter_output < output_config->get_channel(); ++iter_output) {		
			m_ff_singal_caches[iter_output].create((int)m_next_batch_sizes.size(), &m_next_batch_sizes[0], prev_data.front().type());
			ml_mat_op::set(m_ff_singal_caches[iter_output], Scalar(0));
			
			for (int iter_input = 0; iter_input < input_config->get_channel(); ++iter_input) {
				if (m_kernels[iter_input][iter_output].empty()) {
					continue;
				}

				ml_mat_op::cpu_conv(m_conv_results[0], prev_data[iter_input], m_kernels[iter_input][iter_output], Conv_Boundary_Type_Valid, config->get_conv_type(), Conv_Compute_Type_FFT, -1, &config->get_batch_stride()[0]);
				m_ff_singal_caches[iter_output] += m_conv_results[0];
			}

			m_ff_singal_caches[iter_output] += m_biases[iter_output];
		}
	} else {

		m_conv_results.resize(output_config->get_channel());

#pragma omp parallel for
		for (int iter_output = 0; iter_output < output_config->get_channel(); ++iter_output) {		
			m_ff_singal_caches[iter_output].create((int)m_next_batch_sizes.size(), &m_next_batch_sizes[0], prev_data.front().type());
			ml_mat_op::set_zero(m_ff_singal_caches[iter_output]);

			for (int iter_input = 0; iter_input < input_config->get_channel(); ++iter_input) {
				if (m_kernels[iter_input][iter_output].empty()) {
					continue;
				}

				ml_mat_op::cpu_conv(m_conv_results[iter_output], prev_data[iter_input], m_kernels[iter_input][iter_output], Conv_Boundary_Type_Valid, config->get_conv_type(), Conv_Compute_Type_Direct, -1, &config->get_batch_stride()[0]);
				m_ff_singal_caches[iter_output] += m_conv_results[iter_output];
			}

			m_ff_singal_caches[iter_output] += m_biases[iter_output];
		}
	}

	timer.end();
	m_output->to_data_layer()->feedforward_singal(m_ff_singal_caches, pars);
}

void ml_nn_convolution_linked_layer::backpropagation(const ml_nn_layer_learning_params& pars) {
	ml_timer timer(m_config->get_unique_name() + L" backpropagation", false);
	timer.begin();

	bool bp_singal_need_calculate = !m_input->to_data_layer()->get_prev_linked_layers().empty();

	ml_nn_convolution_linked_layer_config* config = (ml_nn_convolution_linked_layer_config*)m_config;

	int next_layer_image_number = (int)m_kernels.front().size();
	int prev_layer_image_number = (int)m_kernels.size();
	const vector<int>& prev_image_size = m_input->get_config()->to_data_layer_config()->get_size();

	const vector<Mat>& prev_data = config->is_padded() ? m_prev_data_padded_caches : m_input->get_data();
	int sample_count = prev_data.front().size.p[0];

	vector<int> temp_sizes = prev_image_size;

	for (int i = 0; i < (int)temp_sizes.size(); ++i) {
		temp_sizes[i] += config->get_tl_pad()[i] + config->get_br_pad()[i];
	}

	int* prev_convolved_delta_image_size = prev_data[0].size.p;

	if (config->is_substride()) {
		m_next_delta_restore_by_stride_caches.resize(next_layer_image_number);
		for (int iter_out = 0; iter_out < next_layer_image_number; ++iter_out) {
			ml_mat_op::restore_mat_by_stride(m_next_delta_restore_by_stride_caches[iter_out], prev_convolved_delta_image_size, m_output->to_data_layer()->get_delta()[iter_out], &config->get_batch_stride()[0]);
		}
	}

	const vector<Mat>& next_delta = config->is_substride() ? m_next_delta_restore_by_stride_caches : m_output->get_delta();
	vector<Mat>& bp_singal_caches = config->is_padded() ? m_bp_singal_padded_caches : m_bp_singal_caches;

	if (bp_singal_need_calculate) {
		bp_singal_caches.resize(prev_layer_image_number);

		if (config->is_big_kernel() && m_input->get_config()->to_data_layer_config()->get_channel_feature_size() > 256) {
			m_bp_singal_conv_result.resize(1);

			for (int iter_input = 0; iter_input < prev_layer_image_number; ++iter_input) {
				bp_singal_caches[iter_input].create(3, prev_convolved_delta_image_size, next_delta.front().type());
				ml_mat_op::set_zero(bp_singal_caches[iter_input]);

				for (int iter_output = 0; iter_output < next_layer_image_number; ++iter_output) {		
					if (m_kernels[iter_input][iter_output].empty()) {
						continue;
					}

					Conv_Type conv_type;

					if (config->get_conv_type() == Conv_Type_Conv) {
						conv_type = Conv_Type_Corr;
					} else {
						conv_type = Conv_Type_Conv;
					}

					ml_mat_op::cpu_conv(m_bp_singal_conv_result[0], next_delta[iter_output], m_kernels[iter_input][iter_output], Conv_Boundary_Type_Full, conv_type, Conv_Compute_Type_FFT);

					bp_singal_caches[iter_input] += m_bp_singal_conv_result[0];	
				}
			}

		} else {
			m_bp_singal_conv_result.resize(prev_layer_image_number);

#pragma omp parallel for
			for (int iter_input = 0; iter_input < prev_layer_image_number; ++iter_input) {
				bp_singal_caches[iter_input].create(3, prev_convolved_delta_image_size, next_delta.front().type());
				ml_mat_op::set_zero(bp_singal_caches[iter_input]);

				for (int iter_output = 0; iter_output < next_layer_image_number; ++iter_output) {
					if (m_kernels[iter_input][iter_output].empty()) {
						continue;
					}

						Conv_Type conv_type;

						if (config->get_conv_type() == Conv_Type_Conv) {
							conv_type = Conv_Type_Corr;
						} else {
							conv_type = Conv_Type_Conv;
						}

						ml_mat_op::cpu_conv(m_bp_singal_conv_result[iter_input], next_delta[iter_output], m_kernels[iter_input][iter_output], Conv_Boundary_Type_Full, conv_type, Conv_Compute_Type_Direct, -1, NULL);
						bp_singal_caches[iter_input] += m_bp_singal_conv_result[iter_input];						
				}
			}
		}
	}

	m_back_kernel_conv_results.resize(next_layer_image_number);

	if (m_flip_flag_size != (int)config->get_batch_kernel_size().size()) {
		m_flip_flag_size = (int)config->get_batch_kernel_size().size();
		BASICML_SAFE_DELETE_ARRAY(m_flip_flags);

		m_flip_flags = new bool[m_flip_flag_size];
		for (int i = 0; i < m_flip_flag_size; ++i) {
			m_flip_flags[i] = true;
		}
	}

#pragma omp parallel for
	for (int iter_output = 0; iter_output < next_layer_image_number; ++iter_output) {
		for (int iter_input = 0; iter_input < prev_layer_image_number; ++iter_input) {
			if (m_kernels[iter_input][iter_output].empty()) {
				continue;
			}

			bool* dst_filp_flags;
			if (config->get_conv_type() == Conv_Type_Conv) {
				dst_filp_flags = m_flip_flags;
			} else {
				dst_filp_flags = NULL;
			}

			ml_mat_op::cpu_conv(m_back_kernel_conv_results[iter_output], prev_data[iter_input], next_delta[iter_output], Conv_Boundary_Type_Valid, Conv_Type_Corr, Conv_Compute_Type_Direct, -1, NULL, dst_filp_flags, NULL, NULL);

			Mat d_kernel = ml_mat_op::decrease_dim(m_back_kernel_conv_results[iter_output]);
			d_kernel /= sample_count;

			ml_define::add_penalty(d_kernel, m_kernels[iter_input][iter_output], config->get_weight_update_info().m_penalty_type, config->get_weight_update_info().m_penalty_alpha);

			ml_define::update_learned_param(m_kernels[iter_input][iter_output], m_v_kernels[iter_input][iter_output], d_kernel, config->get_weight_update_info().m_momentum, config->get_weight_update_info().m_learning_rate);
			
			if (0) {
				BASICLOG_INFO2(ml_string()<<config->get_unique_name()<<L"kernel_"<<iter_input<<L"_"<<iter_output<<L" kernel update");
				ml_define::statistic_weight_update_info(m_v_kernels[iter_input][iter_output]);

				BASICLOG_INFO2(L"current kernel");
				ml_define::statistic_weight_update_info(m_kernels[iter_input][iter_output]);
			}
		}

		double d_kernel_bias = cv::sum(next_delta[iter_output]).val[0] / sample_count;
		ml_define::add_penalty(d_kernel_bias, m_biases[iter_output], config->get_bias_update_info().m_penalty_type, config->get_bias_update_info().m_penalty_alpha);
		ml_define::update_learned_param(m_biases[iter_output], m_v_biases[iter_output], d_kernel_bias, config->get_bias_update_info().m_momentum, config->get_bias_update_info().m_learning_rate);
	}

	if (pars.m_iteration_index % config->get_weight_update_info().m_learning_rate_scale_intereation == 0) {
		config->scale_cur_weight_learning_rate();
	}

	if (pars.m_iteration_index % config->get_bias_update_info().m_learning_rate_scale_intereation == 0) {
		config->scale_cur_bias_learning_rate();
	}

	timer.end();

	if (bp_singal_need_calculate) {
		if (config->is_padded()) {
			m_bp_singal_caches.resize(bp_singal_caches.size());

			for (int iter_input = 0; iter_input < (int)m_bp_singal_caches.size(); ++iter_input) {
				ml_mat_op::unpand_mat(m_bp_singal_caches[iter_input], bp_singal_caches[iter_input], &config->get_batch_tl_pad()[0], &config->get_batch_br_pad()[0]);
			}
		}

		m_input->to_data_layer()->backprapogation_singal(m_bp_singal_caches, pars);	
	}
}

void ml_nn_convolution_linked_layer::init_need_learn_params(int data_type) {
	__super::init_need_learn_params(data_type);

	ml_nn_convolution_linked_layer_config* config = (ml_nn_convolution_linked_layer_config*)m_config;
	ml_nn_data_layer_config* input_config = (ml_nn_data_layer_config*)m_input->get_config();
	ml_nn_data_layer_config* output_config = (ml_nn_data_layer_config*)m_output->get_config();

	vector<Mat> tempKernels;
	tempKernels.resize(output_config->get_channel());
	m_kernels.resize(input_config->get_channel(), tempKernels);
	m_biases.resize(output_config->get_channel(), 0);

	m_v_kernels = m_kernels;
	m_v_biases = m_biases;
	
	int kernel_dims = (int)config->get_kernel_size().size();
	int input_group_size = (input_config->get_channel() - 1) / config->get_group() + 1;
	int output_group_size = (output_config->get_channel() - 1) / config->get_group() + 1;

	for (int iterOutput = 0; iterOutput < output_config->get_channel(); ++iterOutput) {
		int out_group_index = iterOutput / output_group_size;

		if (config->get_bias_update_info().m_learned_param_init_type == ml_Learned_Param_Init_Type_Gaussian) {
			m_biases[iterOutput] = ml_random::gaussian(0, config->get_bias_update_info().m_learned_param_init_params[0]);
		} else {
		
		}

		for (int iterInput = 0; iterInput < input_config->get_channel(); ++iterInput) {
			int input_group_index = iterInput / input_group_size;

			if (input_group_index == out_group_index) {
				if (config->get_weight_update_info().m_learned_param_init_type == ml_Learned_Param_Init_Type_Gaussian) {
					if (kernel_dims == 1) {
						ml_random::gaussian_iid(m_kernels[iterInput][iterOutput], 1, config->get_kernel_size()[0], data_type, 0, config->get_weight_update_info().m_learned_param_init_params[0]);
					} else if (kernel_dims == 2) {
						ml_random::gaussian_iid(m_kernels[iterInput][iterOutput], config->get_kernel_size()[0], config->get_kernel_size()[1], data_type, 0, config->get_weight_update_info().m_learned_param_init_params[0]);
					}

				} else {

				}

				m_v_kernels[iterInput][iterOutput].create(m_kernels[iterInput][iterOutput].dims, m_kernels[iterInput][iterOutput].size.p, data_type);
				ml_mat_op::set_zero(m_v_kernels[iterInput][iterOutput]);
			} else {
				m_kernels[iterInput][iterOutput] = Mat();
			}			
		}
	}
}

void ml_nn_convolution_linked_layer::inner_compute_default_setting() {
	ml_nn_data_layer_config* input_config = (ml_nn_data_layer_config*)m_input->get_config();
	ml_nn_data_layer_config* output_config = (ml_nn_data_layer_config*)m_output->get_config();
	ml_nn_convolution_linked_layer_config* config = (ml_nn_convolution_linked_layer_config*)m_config;

	config->compute_default_setting();

	vector<int> output_sizes = input_config->get_size();
	
	for (int i = 0; i < (int)output_sizes.size(); ++i) {
		output_sizes[i] = ml_mat_op::after_conv(output_sizes[i] + config->get_tl_pad()[i] + config->get_br_pad()[i], config->get_kernel_size()[i], Conv_Boundary_Type_Valid, config->get_stride()[i]);
	}

	output_config->set_size(output_sizes);
}

void ml_nn_convolution_linked_layer::write_learned_param(ml_file_storage& fs) const {
	ml_nn_convolution_linked_layer_config* config = (ml_nn_convolution_linked_layer_config*)m_config;
	ml_nn_data_layer_config* input_config = (ml_nn_data_layer_config*)m_input->get_config();
	ml_nn_data_layer_config* output_config = (ml_nn_data_layer_config*)m_output->get_config();

	fs<<L"{";

	fs<<L"prev_channel"<<(int)input_config->get_channel();
	fs<<L"next_channel"<<(int)output_config->get_channel();

	for (int i = 0; i < (int)m_kernels.size(); ++i) {
		for (int j = 0; j < (int)m_kernels[i].size(); ++j) {
			fs<<(ml_string()<<L"kernel_"<<i<<L"_"<<j)<<m_kernels[i][j];
			fs<<(ml_string()<<L"v_kernel_"<<i<<L"_"<<j)<<m_v_kernels[i][j];
		}
	}

	fs<<(ml_string()<<L"biases")<<m_biases;
	fs<<(ml_string()<<L"v_biases")<<m_v_biases;

	fs<<L"}";
}

void ml_nn_convolution_linked_layer::read_learned_param(const ml_file_node& node) {
	int prev_channel;
	int next_channel;

	if (!node[L"prev_channel"].empty()) {
		prev_channel = node[L"prev_channel"];
	} else {
		BASICLOG_ASSERT2(false);
	}

	if (!node[L"next_channel"].empty()) {
		next_channel	= node[L"next_channel"];
	} else {
		BASICLOG_ASSERT2(false);
	}

	vector<Mat> temp;
	temp.resize(next_channel);
	m_kernels.resize(prev_channel, temp);
	m_v_kernels.resize(prev_channel, temp);

	for (int i = 0; i < prev_channel; ++i) {
		for (int j = 0; j < next_channel; ++j) {
			m_kernels[i][j] = node[(ml_string()<<L"kernel_"<<i<<L"_"<<j)];
			m_v_kernels[i][j] = node[(ml_string()<<L"v_kernel_"<<i<<L"_"<<j)];
		}
	}

	m_biases = node[L"biases"];
	m_v_biases = node[L"v_biases"];
}

ml_nn_layer* ml_nn_convolution_linked_layer::clone() const {
	ml_nn_convolution_linked_layer* layer = new ml_nn_convolution_linked_layer();

	for (int i = 0; i < (int)m_kernels.size(); ++i) {
		for (int j = 0; j < (int)m_kernels[i].size(); ++j) {
			layer->m_kernels[i][j] = m_kernels[i][j].clone();
			layer->m_v_kernels[i][j] = m_v_kernels[i][j].clone();
		}
	}
	
	layer->m_biases = m_biases;
	layer->m_v_biases = m_v_biases;

	return layer;
}