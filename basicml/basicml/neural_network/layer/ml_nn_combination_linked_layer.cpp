#include "stdafx.h"

#include "ml_nn_combination_linked_layer.h"
#include "ml_nn_data_layer_config.h"
#include "ml_nn_data_layer.h"

void ml_nn_combination_linked_layer::feedforward(const ml_nn_layer_learning_params& pars) {
	ml_nn_data_layer_config* input_config = (ml_nn_data_layer_config*)m_input->get_config();
	
	const vector<Mat>& prev_datas = m_input->to_data_layer()->get_data();
	const vector<int>& prev_image_size = input_config->get_size();
	int prev_image_number = input_config->get_channel();

	int sample_count = prev_datas.front().size.p[0];
	int data_type = prev_datas.front().type();

	int prev_channel_feature_size = input_config->get_channel_feature_size();
	m_ff_singal_cache.create(sample_count, prev_channel_feature_size * prev_image_number, data_type);

	uchar* ptr_next_input_cache_dim0 = m_ff_singal_cache.data;

	int image_byte = prev_channel_feature_size;

	if (data_type == CV_32FC1) {
		image_byte *= sizeof(float);
	} else {
		image_byte *= sizeof(double);
	}

	for (int iter_sample = 0; iter_sample < sample_count; ++iter_sample) {
		uchar* ptr_next_input_cache_dim1 = ptr_next_input_cache_dim0;

		for (int iter_input = 0; iter_input < prev_image_number; ++iter_input) {
			const uchar* imagePointer = prev_datas[iter_input].ptr<uchar>(iter_sample);

			memcpy(ptr_next_input_cache_dim1, imagePointer, image_byte);
			ptr_next_input_cache_dim1 += image_byte;
		}

		ptr_next_input_cache_dim0 += m_ff_singal_cache.step.p[0];
	}

	m_output->to_data_layer()->feedforward_singal(m_ff_singal_cache, pars);
}

void ml_nn_combination_linked_layer::backpropagation(const ml_nn_layer_learning_params& pars) {
	ml_nn_data_layer_config* input_config = (ml_nn_data_layer_config*)m_input->get_config();

	const Mat& next_delta = m_output->to_data_layer()->get_delta(0);

	const vector<int>& prev_image_size = input_config->get_size();
	int prev_image_number = input_config->get_channel();

	m_bp_singal_caches.resize(prev_image_number);

	int prev_channel_feature_size = input_config->get_channel_feature_size();

	if ((int)prev_image_size.size() == 2) {
		int dim_sizes[] = {next_delta.rows, prev_image_size[0], prev_image_size[1]};

		for (int iter_image = 0; iter_image < prev_image_number; ++iter_image) {
			Mat col_range = next_delta.colRange(iter_image * prev_channel_feature_size, (iter_image + 1) * prev_channel_feature_size);

			col_range.copyTo(m_bp_singal_caches[iter_image]);
			m_bp_singal_caches[iter_image] = m_bp_singal_caches[iter_image].reshape(1, 3, dim_sizes);
		}
	}
	
	m_input->to_data_layer()->backprapogation_singal(m_bp_singal_caches, pars);
}

void ml_nn_combination_linked_layer::inner_compute_default_setting() {
	m_output->get_config()->to_data_layer_config()->set_size(m_input->get_config()->to_data_layer_config()->get_feature_size());
}

ml_nn_layer* ml_nn_combination_linked_layer::clone() const {
	return new ml_nn_combination_linked_layer();
}