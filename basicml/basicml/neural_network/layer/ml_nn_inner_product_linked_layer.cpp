#include "stdafx.h"

#include "ml_nn_inner_product_linked_layer.h"
#include "ml_nn_data_layer.h"
#include "ml_learning_param_updater.h"



void ml_nn_inner_product_linked_layer::feedforward(const ml_nn_layer_learning_params& pars) {
	mt_mat input_signal = m_input->to_data_layer()->get_front_activated_output();

	if (pars.m_inference_stage && m_drop_type != Drop_Type_Null && m_inference_type == Inference_Type_Drawn) {
		basiclog_assert2(m_drawn_number > 0);
		mt_mat drop_means = input_signal.mul(m_weight);
		drop_means *= m_drop_ratio;

		mt_mat prev_input_square = input_signal.pow(2.0);
		mt_mat weight_square = m_weight.pow(2);

		mt_mat drop_standard_deviation = prev_input_square.mul(weight_square);

		drop_standard_deviation *= m_drop_ratio * (1 - m_drop_ratio);
		drop_standard_deviation.self_pow(0.5);

		vector<mt_mat> drop_samples;

		mt_random::gaussian_iid(drop_samples, m_drawn_number, drop_means, drop_standard_deviation);

		for (i32 iter_drawn_number = 0; iter_drawn_number < m_drawn_number; ++iter_drawn_number) {
			drop_samples[iter_drawn_number] += m_bias.repeat(input_signal.size()[0], 1);
		}

		m_output->to_data_layer()->feedforward_drop_drawn_singal(drop_samples, pars);

	} else {
		if (pars.m_inference_stage) {
			if (m_drop_type == Drop_Type_Null) {
				mt_mat ff_signal = input_signal.mul(m_weight) + m_bias.repeat(ff_signal.size()[0], 0);
				m_output->to_data_layer()->feedforward_singal(ff_signal, pars);
			} else {
				if (m_drop_type == Inference_By_Average) {
					mt_mat ff_signal = input_signal.mul(m_weight) + m_bias.repeat(ff_signal.size()[0], 0);
					ff_signal *= m_drop_ratio;
					m_output->to_data_layer()->feedforward_singal(ff_signal, pars);

				} else {
					basiclog_assert2(sys_false);
				}
			}
		} else {
			mt_mat ff_signal;

			if (m_drop_type == Drop_Type_Out) {
				mt_mat out_mask = mt_random::bernoulli_iid(input_signal.size()[0], m_weight.size()[1], m_weight.depth_channel(), 1, m_drop_ratio);
				input_signal = input_signal * out_mask;
				ff_signal = input_signal.mul(m_weight);

			} else if (m_drop_type == Drop_Type_Connect) {
				mt_mat weight_mask = mt_random::bernoulli_iid(m_weight.dim(), m_weight.size(), m_weight.depth_channel(), 1, m_drop_ratio);
				ff_signal = input_signal.mul(m_weight * weight_mask);
			} else {

				ff_signal = input_signal.mul(m_weight);
			}

			ff_signal = ff_signal + m_bias.repeat(ff_signal.size()[0], 0);

			m_output->to_data_layer()->feedforward_singal(ff_signal, pars);
		}
	}
}

void ml_nn_inner_product_linked_layer::update_learning_param(const vector<mt_mat>& losses, const ml_nn_layer_learning_params& pars) {
	m_weight_updater->update(m_weight, losses.front().auto_derivative()->derivate(m_weight, losses), pars);
	m_bias_updater->update(m_bias, losses.front().auto_derivative()->derivate(m_bias, losses), pars);
}

void ml_nn_inner_product_linked_layer::init_need_learn_params(int data_type) {
	__super::init_need_learn_params(data_type);

	basiclog_info2(sys_strcombine()<<L"layer: "<<name()<<L" init needed learning parameters");

	if (m_weight_updater->init_type() == ml_Learning_Param_Init_Type_Gaussian) {
		m_weight = mt_random::gaussian_iid(m_input->size()[0], m_output->size()[0], data_type, m_weight_updater->init_param()[0], m_weight_updater->init_param()[1]);		
	} else {
		basiclog_assert2(false);
	}

	if (m_bias_updater->init_type() == ml_Learning_Param_Init_Type_Gaussian) {
		m_bias = mt_random::gaussian_iid(1, m_output->size()[0], data_type, m_bias_updater->init_param()[0], m_bias_updater->init_param()[1]);
	}

	m_weight_updater->init(m_weight);
	m_bias_updater->init(m_bias);
}

void ml_nn_inner_product_linked_layer::write(sys_json_writer& writer, b8 save_learned_param) const {
	writer<<L"{";

	writer<<L"input"<<m_input->name();
	writer<<L"output"<<m_output->name();
	writer<<L"drop_type"<<ml_Drop_Type_Descriptions[m_drop_type];

	if (m_drop_type != Drop_Type_Null) {
		writer<<L"drop_type"<<ml_Drop_Type_Descriptions[m_drop_type];
		writer<<L"drop_ratio"<<m_drop_ratio;
		writer<<L"inference_type"<<ml_Inference_Type_Descriptions[m_inference_type];

		if (m_inference_type == Inference_Type_Drawn) {
			writer<<L"drawn_number"<<m_drawn_number;
		}
	}

	writer<<L"weight_updater";
	m_weight_updater->write(writer, save_learned_param);

	writer<<L"bias_updater";
	m_bias_updater->write(writer, save_learned_param);

	if (save_learned_param) {
		writer<<L"weight"<<m_weight;
		writer<<L"bias"<<m_bias;
	}

	writer<<L"}";
}

ml_nn_inner_product_linked_layer* ml_nn_inner_product_linked_layer::read(const sys_json_reader& reader) {
	if ((wstring)reader[L"layer_type"] != ml_nn_inner_product_linked_layer::layer_type()) {
		return NULL;
	}

	ml_nn_inner_product_linked_layer* layer = new ml_nn_inner_product_linked_layer();
	layer->set_name(reader.node_name());

	reader[L"input"]>>layer->m_input_name;
	basiclog_assert2(!layer->m_input_name.empty());

	reader[L"output"]>>layer->m_output_name;
	basiclog_assert2(!layer->m_output_name.empty());

	wstring drop_type;
	reader[L"drop_type"]>>drop_type;
	basiclog_assert2(!drop_type.empty());

	layer->m_drop_type = (Drop_Type)ml_helper::find_in_text(ml_Drop_Type_Descriptions, sizeof(ml_Drop_Type_Descriptions) / sizeof(wstring), drop_type);

	if (reader.has_key(L"weight_updater")) {
		layer->m_weight_updater = ml_learning_param_updater::read(reader[L"weight_updater"]);
	} else {
		layer->m_weight_updater = new ml_bsgd_learning_param_updater();
	}

	if (reader.has_key(L"bias_updater")) {
		layer->m_bias_updater = ml_learning_param_updater::read(reader[L"bias_updater"]);
	} else {
		layer->m_bias_updater = new ml_bsgd_learning_param_updater();
	}

	if (reader.has_key(L"weight")) {
		reader[L"weight"]>>layer->m_weight;

		if (layer->m_weight.is_empty()) {
			basiclog_assert2(L"weight matrix contains no element");
		}
	}

	if (reader.has_key(L"bias")) {
		reader[L"bias"]>>layer->m_bias;

		if (layer->m_bias.is_empty()) {
			basiclog_assert2(L"bias matrix contains no element");
		}
	} else {
		basiclog_assert2(L"can not load bias");
	}
}

void ml_nn_inner_product_linked_layer::copy_learned_param(const ml_nn_linked_layer* other) {	
	m_weight.set(other->to_inner_product_linked_layer()->m_weight);
	m_bias.set(other->to_inner_product_linked_layer()->m_bias);

	m_weight_updater->on_copy_from_other();
	m_bias_updater->on_copy_from_other();
}

void ml_nn_inner_product_linked_layer::copy_learned_param(const mt_mat& weight, const mt_mat& bias) {
	m_weight.set(weight);
	m_bias.set(bias);

	m_weight_updater->on_copy_from_other();
	m_bias_updater->on_copy_from_other();
}

ml_nn_layer* ml_nn_inner_product_linked_layer::clone() const {
	ml_nn_inner_product_linked_layer* layer = new ml_nn_inner_product_linked_layer();

	layer->m_weight = m_weight.clone();
	layer->m_bias = m_bias.clone();

	layer->m_input_name = m_input->name();
	layer->m_output_name = m_output->name();

	return layer;
}