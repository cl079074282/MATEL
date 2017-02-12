#include "stdafx.h"

#include "ml_nn_maxout_linked_layer.h"
#include "ml_nn_data_layer.h"
#include "ml_nn_data_layer_config.h"
#include "ml_nn_maxout_linked_layer_config.h"

void ml_nn_maxout_linked_layer::feedforward(const ml_nn_layer_learning_params& pars) {
	ml_nn_data_layer_config* output_config = m_output->get_config()->to_data_layer_config();
	int next_image_number = output_config->get_channel();
	const vector<Mat>& prev_data = m_input->to_data_layer()->get_data();

	ml_mat_op::max_out(m_ff_singal_caches, m_next_input_max_masks, prev_data, m_config->to_maxout_linked_layer_config()->get_k());
	m_output->to_data_layer()->feedforward_singal(m_ff_singal_caches, pars);
}

void ml_nn_maxout_linked_layer::backpropagation(const ml_nn_layer_learning_params& pars) {
	ml_nn_data_layer_config* input_config = m_input->get_config()->to_data_layer_config();
	int prev_image_number = input_config->get_channel();

	const vector<Mat>& next_deltas = m_output->to_data_layer()->get_delta();

	ml_mat_op::restore_max_out(m_bp_singal_caches, prev_image_number, next_deltas, m_next_input_max_masks);
	m_input->to_data_layer()->backprapogation_singal(m_bp_singal_caches, pars);
}

void ml_nn_maxout_linked_layer::inner_compute_default_setting() {
	m_output->get_config()->to_data_layer_config()->set_channel(m_input->get_config()->to_data_layer_config()->get_channel());
	m_output->get_config()->to_data_layer_config()->set_size(m_input->get_config()->to_data_layer_config()->get_size());
}

ml_nn_layer* ml_nn_maxout_linked_layer::clone() const {
	return new ml_nn_maxout_linked_layer();
}