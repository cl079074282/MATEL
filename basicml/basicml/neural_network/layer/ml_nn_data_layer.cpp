#include "stdafx.h"

#include "ml_nn_data_layer.h"
#include "ml_nn_linked_layer.h"
#include "ml_nn_data_layer_config.h"

void ml_nn_data_layer::feedforward_drop_drawn_singal(const vector<mt_mat>& drawn_singals, const ml_nn_layer_learning_params& pars) {
	basiclog_assert2(pars.m_inference_stage);

	if (m_drawn_input_singals.empty()) {
		m_drawn_input_singals = drawn_singals;
	} else {
		basiclog_assert2(m_drawn_input_singals.size() == drawn_singals.size());

		for (int i = 0; i < (int)drawn_singals.size(); ++i) {
			m_drawn_input_singals[i] += drawn_singals[i];
		}
	}

	++m_feedforward_count;

	try_activate(pars);
}

void ml_nn_data_layer::feedforward_singal(const mt_mat& input_singal, const ml_nn_layer_learning_params& pars) {
	vector<mt_mat> input_singals;
	input_singals.push_back(input_singal);

	feedforward_singal(input_singals, pars);
}

void ml_nn_data_layer::feedforward_singal(const vector<mt_mat>& singals, const ml_nn_layer_learning_params& pars) {
	basiclog_assert2((i32)singals.size() == channel());

	m_input_singals.resize(singals.size());

	for (i32 iter_channel = 0; iter_channel < (i32)singals.size(); ++iter_channel) {
		m_input_singals[iter_channel].push_back(singals[iter_channel]);
	}

	m_input_singals.push_back(singals);

	++m_feedforward_count;
	try_activate(pars);
}

void ml_nn_data_layer::try_activate(const ml_nn_layer_learning_params& pars) {
	if (m_feedforward_count >= (int)m_prev_linked_layers.size()) {

		m_activated_signals.resize(channel());

		if (pars.m_inference_stage) {
			// inference stage, we do not need to consider the auto derivative
			for (i32 iter_channel = 0; iter_channel < m_channels; ++iter_channel) {
				m_activated_signals[iter_channel] = mt_mat_helper::add(m_input_singals[iter_channel]);
			}

			if (m_drawn_input_singals.empty()) {
				for (i32 iter_channel = 0; iter_channel < m_channels; ++iter_channel) {
					m_activated_signals[iter_channel].self_activate(m_activate_type, m_activate_params);
				}
			} else {
				basiclog_assert2(m_channels == 1);
				basiclog_assert2((int)m_data_sizes.size() == 1);

				mt_mat temp_data = mt_mat(m_drawn_input_singals.front());
				temp_data.set(0);

				for (int i = 0; i < (int)m_drawn_input_singals.size(); ++i) {

					if (!m_input_singals.empty()) {
						m_drawn_input_singals[i] += m_activated_signals[0];
					}

					m_drawn_input_singals[i].self_activate(m_activate_type, m_activate_params);
					temp_data += m_drawn_input_singals[i];
				}

				m_activated_signals[0] = temp_data / (int)m_drawn_input_singals.size();
			}


		} else {
			basiclog_assert2(m_drawn_input_singals.empty());

			// inference stage, we need to consider the auto derivative
			vector<mt_mat> out_singals;
			out_singals.resize(channel());

			for (i32 iter_channel = 0; iter_channel < channel(); ++iter_channel) {
				out_singals[iter_channel] = mt_mat_helper::add(m_input_singals[iter_channel]);
				m_activated_signals[iter_channel] = out_singals[iter_channel].activate(m_activate_type, m_activate_params);
			}
		}

		for (int i = 0; i < (int)m_next_linked_layers.size(); ++i) {
			m_next_linked_layers[i]->feedforward(pars);
		}

		m_feedforward_count = 0;

		m_input_singals.clear();
		m_drawn_input_singals.clear();
	}
}

void ml_nn_data_layer::compute_default_setting() {
	for (int i = 0; i < (int)m_next_linked_layers.size(); ++i) {
		m_next_linked_layers[i]->compute_default_setting();
	}
}

ml_nn_layer* ml_nn_data_layer::clone() const {
	return new ml_nn_data_layer();
}