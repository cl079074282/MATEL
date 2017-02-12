#include "stdafx.h"

#include "ml_nn_pooling_linked_layer.h"
#include "ml_nn_data_layer.h"
#include "ml_nn_pooling_linked_layer_config.h"
#include "ml_nn_data_layer_config.h"

void ml_nn_pooling_linked_layer::feedforward(const ml_nn_layer_learning_params& pars) {
	const vector<Mat>& prev_datas = m_input->to_data_layer()->get_data();
	int prev_image_number = m_input->get_config()->to_data_layer_config()->get_channel();

	int sample_count = prev_datas.front().size.p[0];

	m_ff_singal_caches.resize(prev_image_number);
	m_input_max_masks.resize(prev_image_number);

	ml_nn_pooling_linked_layer_config* config = (ml_nn_pooling_linked_layer_config*)m_config;

	for (int iter_output = 0; iter_output < prev_image_number; ++iter_output) {
		ml_mat_op::ml_pooling(m_ff_singal_caches[iter_output], m_input_max_masks[iter_output], prev_datas[iter_output], config->get_type(), &config->get_batch_kernel_size()[0], &config->get_batch_stride_size()[0]);
	}

	m_output->to_data_layer()->feedforward_singal(m_ff_singal_caches, pars);
}

void ml_nn_pooling_linked_layer::backpropagation(const ml_nn_layer_learning_params& pars) {
	const vector<Mat>& prev_datas = m_input->to_data_layer()->get_data();
	const vector<Mat>& next_deltas = m_output->to_data_layer()->get_delta();
	int prev_image_number = m_input->get_config()->to_data_layer_config()->get_channel();

	m_bp_singal_caches.resize(prev_image_number);
	ml_nn_pooling_linked_layer_config* config = (ml_nn_pooling_linked_layer_config*)m_config;

	for (int iter_image = 0; iter_image < prev_image_number; ++iter_image) {		
		//ml_mat_op::traceMat(next_deltas[iter_image]);
		ml_mat_op::ml_unpooling(m_bp_singal_caches[iter_image], prev_datas[iter_image].size.p, next_deltas[iter_image], m_input_max_masks[iter_image], config->get_type(), &config->get_batch_kernel_size()[0], &config->get_batch_stride_size()[0]);
		//ml_mat_op::traceMat(m_bp_singal_caches[iter_image]);
	}

	m_input->to_data_layer()->backprapogation_singal(m_bp_singal_caches, pars);
}

void ml_nn_pooling_linked_layer::inner_compute_default_setting() {
	ml_nn_data_layer_config* input_config = (ml_nn_data_layer_config*)m_input->get_config();
	ml_nn_data_layer_config* output_config = (ml_nn_data_layer_config*)m_output->get_config();
	ml_nn_pooling_linked_layer_config* config = (ml_nn_pooling_linked_layer_config*)m_config;

	config->compute_default_setting();

	vector<int> output_sizes = input_config->get_size();

	for (int i = 0; i < (int)output_sizes.size(); ++i) {
		output_sizes[i] = ml_mat_op::get_pooling_result_size(output_sizes[i], config->get_kernel_size()[i], config->get_stride().empty() ? config->get_kernel_size()[i] : config->get_kernel_size()[i]);
	}

	output_config->set_channel(input_config->get_channel());
	output_config->set_size(output_sizes);
}

ml_nn_layer* ml_nn_pooling_linked_layer::clone() const {
	return new ml_nn_pooling_linked_layer();
}