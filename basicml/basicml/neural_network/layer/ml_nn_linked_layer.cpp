#include "stdafx.h"

#include "ml_nn_linked_layer.h"
#include "ml_nn_data_layer.h"
#include "ml_learning_param_updater.h"

ml_nn_linked_layer::ml_nn_linked_layer(const wstring& layer_name, ml_nn_data_layer* input_layer, ml_nn_data_layer* output_layer, ml_learning_param_updater* weight_updater /* = new ml_bsgd_learning_param_updater() */, ml_learning_param_updater* bias_updater /* = new ml_bsgd_learning_param_updater() */)
	: ml_nn_layer(layer_name)
	, m_input(NULL)
	, m_output(NULL) {
	

		m_weight_updater = weight_updater;
		m_bias_updater = bias_updater;

		if (m_weight_updater == NULL) {
			m_weight_updater = new ml_bsgd_learning_param_updater();
		}

		if (m_bias_updater == NULL) {
			m_bias_updater = new ml_bsgd_learning_param_updater();
		}

		set_input(input_layer);
		set_output(output_layer);
}

ml_nn_linked_layer::~ml_nn_linked_layer() {
	basicsys_delete(m_weight_updater);
	basicsys_delete(m_bias_updater);
}

const wstring& ml_nn_linked_layer::input_name() const  {
	if (m_input != NULL) {
		return m_input->name();
	}

	return m_input_name;
}

const wstring& ml_nn_linked_layer::output_name() const {
	if (m_output != NULL) {
		return m_output->name();
	}

	return m_output_name;
}

void ml_nn_linked_layer::compute_default_setting() {
	inner_compute_default_setting();

	if (NULL != m_output) {
		m_output->compute_default_setting();
	}
}

void ml_nn_linked_layer::set_input(ml_nn_data_layer* input) {
	if (m_input == input) {
		return;
	}

	//a linked layer can only has a input layer, hence we need to erase itself from the previous input data layer 
	if (NULL != m_input) {
		vector<ml_nn_linked_layer*>::iterator iter = find(m_input->m_next_linked_layers.begin(), m_input->m_next_linked_layers.end(), this);

		if (iter != m_input->m_next_linked_layers.end()) {
			m_input->m_next_linked_layers.erase(iter);
		}
	}

	m_input = input;
	m_input->m_next_linked_layers.push_back(this);
}

void ml_nn_linked_layer::set_output(ml_nn_data_layer* output) {
	if (m_output == output) {
		return;
	}

	if (NULL != m_output) {
		vector<ml_nn_linked_layer*>::iterator iter = find(m_output->m_prev_linked_layers.begin(), m_output->m_prev_linked_layers.end(), this);

		if (iter != m_output->m_prev_linked_layers.end()) {
			m_output->m_prev_linked_layers.erase(iter);
		}
	}

	m_output = output;
	m_output->m_prev_linked_layers.push_back(this);
}

void ml_nn_linked_layer::write(sys_json_writer& writer, b8 write_learned_param /* = sys_true */) const {

}

ml_nn_linked_layer* ml_nn_linked_layer::read(const sys_json_reader& reader) {
	return NULL;
}