#include "stdafx.h"

#include "ml_nn_batch_norm_linked_layer.h"
#include "ml_nn_data_layer.h"
#include "ml_learning_param_updater.h"

void ml_nn_batch_norm_linked_layer::init_need_learn_params(int data_type) {
	__super::init_need_learn_params(data_type);

	m_gmmas.resize(m_input->channel());
	m_betas.resize(m_input->channel());
	m_total_train_means.resize(m_input->channel());
	m_total_train_variances.resize(m_output->channel());

	for (int i = 0; i < (int)m_gmmas.size(); ++i) {
		int cols = 1;
		if (m_batch_norm_type == Batch_Norm_Type_Unit) {
			cols = m_input->channel_feature_size();
		}

		m_gmmas[i] = mt_mat(1, cols, data_type, mt_scalar(1));
		m_betas[i] = mt_mat(1, cols, data_type, mt_scalar(0));
	}
}

void ml_nn_batch_norm_linked_layer::feedforward(const ml_nn_layer_learning_params& pars) {
	vector<mt_mat> prev_datas = m_input->to_data_layer()->get_activated_output();
	
	for (int iter_prev_channel = 0; iter_prev_channel < (int)prev_datas.size(); ++iter_prev_channel) {
		int new_dims[2];
		
		if (m_batch_norm_type == Batch_Norm_Type_Unit) {
			new_dims[0] = prev_datas[iter_prev_channel].size()[0];
			new_dims[1] = m_input->channel_feature_size();
		} else {
			new_dims[0] = prev_datas[iter_prev_channel].size()[0] * m_input->channel_feature_size();
			new_dims[1] = 1;
		}
		
		prev_datas[iter_prev_channel] = prev_datas[iter_prev_channel].reshape(2, new_dims);
	}

	vector<mt_mat> ff_signals;
	ff_signals.resize(prev_datas.size());

	for (int iter_prev_channel = 0; iter_prev_channel < (int)prev_datas.size(); ++iter_prev_channel) {

		mt_mat mean = prev_datas[iter_prev_channel].reduce(mt_mat::Reduce_Type_Mean, 0);
		mt_mat variance = prev_datas[iter_prev_channel].reduce(mt_mat::Reduce_Type_Variance, 0);

		if (pars.m_batch_norm_statistic) {
			if (pars.m_iteration_index == 0) {
				m_total_train_means[iter_prev_channel].set(mean);
				m_total_train_variances[iter_prev_channel].set(variance);
			} else if (pars.m_iteration_index + 1 == pars.m_total_iteration_number) {
				m_total_train_means[iter_prev_channel] /= pars.m_total_iteration_number;
				double unbias_term = (prev_datas[iter_prev_channel].size()[0] - 1) / prev_datas[iter_prev_channel].size()[0];
				m_total_train_variances[iter_prev_channel] /= (pars.m_total_iteration_number * unbias_term);
			} else {
				m_total_train_means[iter_prev_channel] += mean;
				m_total_train_variances[iter_prev_channel] += variance;
			}
		}

		prev_datas[iter_prev_channel] = prev_datas[iter_prev_channel] - mean.repeat(prev_datas[iter_prev_channel].size()[0], 0);
		variance = variance + m_regular_term;
		mt_mat standard_variance = variance.pow(0.5);
		mt_mat normed_data = prev_datas[iter_prev_channel] / standard_variance.repeat(prev_datas[iter_prev_channel].size()[0], 0);
		ff_signals[iter_prev_channel] = normed_data * m_gmmas[iter_prev_channel] + m_betas[iter_prev_channel];
	}
	
	for (int iter_prev_channel = 0; iter_prev_channel < (int)ff_signals.size(); ++iter_prev_channel) {
		ff_signals[iter_prev_channel] = ff_signals[iter_prev_channel].reshape(prev_datas[iter_prev_channel].dim(), prev_datas[iter_prev_channel].size());
	}

	m_output->to_data_layer()->feedforward_singal(ff_signals, pars);
}

void ml_nn_batch_norm_linked_layer::update_learning_param(const vector<mt_mat>& losses, const ml_nn_layer_learning_params& pars) {
	m_weight_updater->update(m_gmmas, losses.front().auto_derivative()->derivate(m_gmmas, losses), pars);
	m_bias_updater->update(m_betas, losses.front().auto_derivative()->derivate(m_betas, losses), pars);
}