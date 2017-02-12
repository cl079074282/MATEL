#include "stdafx.h"

#include "ml_bsgd_learning_param_updater.h"

void ml_bsgd_learning_param_updater::update(mt_mat& learning_param, mt_mat& gradient, const ml_nn_layer_learning_params& pars) {
	mt_mat v_gradient;

	for (i32 i = 0; i < (i32)m_v_gradients.size(); ++i) {
		if (learning_param.is_same(m_learning_params[i])) {
			v_gradient = m_v_gradients[i];
		}
	}

	add_penalty(gradient, learning_param, m_penalty_type, m_penalty_alpha);

	v_gradient = v_gradient * m_momentum - gradient * m_learning_ratio;
	gradient += v_gradient;

	if (pars.m_iteration_index + 1 % m_auto_scaled_iteration_number == 0) {
		m_learning_ratio *= m_learning_ratio_scaled_ratio;
	} 
}

void ml_bsgd_learning_param_updater::init(vector<mt_mat>& learning_params) {
	m_learning_params = learning_params;

	m_v_gradients.resize(m_learning_params.size());

	for (i32 i = 0; i < (i32)m_v_gradients.size(); ++i) {
		m_v_gradients[i] = mt_mat(m_learning_params[i], mt_mat::Construct_Type_Create_As_Size);
	}
}

void ml_bsgd_learning_param_updater::on_copy_from_other() {
	m_v_gradients.resize(m_learning_params.size());

	for (i32 i = 0; i < (i32)m_v_gradients.size(); ++i) {
		m_v_gradients[i] = mt_mat(m_learning_params[i], mt_mat::Construct_Type_Create_As_Size);
	}
}

void ml_bsgd_learning_param_updater::write(sys_json_writer& writer, b8 write_learned_param) const {
	writer<<L"{";
	writer<<L"updater_type"<<class_name();

	writer<<L"auto_scaled_iteration_number"<<m_auto_scaled_iteration_number;
	writer<<L"learning_ratio"<<m_learning_ratio;
	writer<<L"learning_ratio_scaled_ratio"<<m_learning_ratio_scaled_ratio;
	writer<<L"momentum"<<m_momentum;
	writer<<L"penalty_type"<<ml_Penalty_Type_Descriptions[m_penalty_type];
	
	writer<<L"penalty_alpha"<<m_penalty_alpha;

	if (write_learned_param) {
		writer<<L"v_gradients"<<m_v_gradients;
	}

	writer<<L"}";
}

ml_bsgd_learning_param_updater* ml_bsgd_learning_param_updater::read(const sys_json_reader& reader) {
	if ((wstring)reader[L"updater_type"] != class_name()) {
		return NULL;
	}

	ml_bsgd_learning_param_updater* updater = new ml_bsgd_learning_param_updater();
	reader[L"auto_scaled_iteration_number"]>>updater->m_auto_scaled_iteration_number;
	reader[L"learning_ratio"]>>updater->m_learning_ratio;
	reader[L"learning_ratio_scaled_ratio"]>>updater->m_learning_ratio_scaled_ratio;
	reader[L"momentum"]>>updater->m_momentum;

	if (reader.has_key(L"penalty_type")) {
		wstring penalty_type_description;
		reader[L"penalty_type"]>>penalty_type_description;

		updater->m_penalty_type = (ml_Penalty_Type)ml_helper::find_in_text(ml_Penalty_Type_Descriptions, penalty_type_description, sys_true);
		reader[L"penalty_alpha"]>>updater->m_penalty_alpha;
	} 

	reader[L"v_gradients"]>>updater->m_v_gradients;

	return updater;
}