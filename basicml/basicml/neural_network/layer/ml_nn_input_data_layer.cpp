#include "stdafx.h"

#include "ml_nn_input_data_layer.h"
#include "ml_nn_linked_layer.h"

void ml_nn_input_data_layer::feedforward_by_input(const mt_mat& input, const ml_nn_layer_learning_params& pars) {
	mt_mat temp_input = input;

	basiclog_assert_message2(temp_input.channel() == m_channels, L"the channel of input feature does not match the image number of setting");
	basiclog_assert_message2(temp_input.size()[1] == feature_size(), L"the dim of input feature does not match the image size of setting");

	vector<mt_mat> splited_input_datas;
	temp_input.split(splited_input_datas);

	vector<i32> sizes;
	sizes.push_back(input.size()[0] * pars.m_sequence_length);
	sizes.insert(sizes.end(), m_data_sizes.begin(), m_data_sizes.end());

	for (int iter_channel = 0; iter_channel < (i32)splited_input_datas.size(); ++iter_channel) {
		splited_input_datas[iter_channel] = splited_input_datas[iter_channel].reshape(sizes);
	}

	feedforward_singal(splited_input_datas, pars);
}

ml_nn_layer* ml_nn_input_data_layer::clone() const {
	return new ml_nn_input_data_layer();
}