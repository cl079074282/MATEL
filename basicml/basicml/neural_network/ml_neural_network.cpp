#include "stdafx.h"
#include "ml_neural_network.h"
#include "ml_nn_input_data_layer.h"
#include "ml_nn_linked_layer.h"
#include "ml_nn_output_data_layer.h"

ml_neural_network::~ml_neural_network() {
	for (int i = 0; i < (int)m_input_layers.size(); ++i) {
		delete m_input_layers[i];
	}

	for (int i = 0; i < (int)m_hidden_layers.size(); ++i) {
		delete m_hidden_layers[i];
	}

	for (int i = 0; i < (int)m_output_layers.size(); ++i) {
		delete m_output_layers[i];
	}

	for (int i = 0; i < (int)m_linked_layers.size(); ++i) {
		delete m_linked_layers[i];
	}
}

void ml_neural_network::write(const wstring& path, b8 text_type, b8 write_learned_param) const {
	if (text_type) {
		sys_string_file_buffer_writer writer(path);
		write(&writer, write_learned_param);
	} else {
		sys_byte_file_buffer_writer writer(path);
		write(&writer, write_learned_param);
	}
}

ml_neural_network* ml_neural_network::read(const wstring& path, b8 text_type) {
	if (text_type) {
		sys_string_file_buffer_reader reader(path);
		return read(&reader);
	} else {
		sys_byte_file_buffer_reader reader(path);
		return read(&reader);
	}
}

void ml_neural_network::write(sys_buffer_writer* buffer_writer, b8 write_learned_param) const {
	sys_json_writer json_writer(buffer_writer);

	json_writer<<L"ml_neural_network";

	write(json_writer, write_learned_param);
}

ml_neural_network* ml_neural_network::read(sys_buffer_reader* buffer_reader) {
	sys_json_reader json_reader(buffer_reader);

	if (!json_reader.has_key(L"ml_neural_network")) {
		return NULL;
	}

	return read(json_reader);
}

void ml_neural_network::write(sys_json_writer& writer, b8 write_learned_param) const {
	writer<<L"{";
	writer<<L"version"<<0;

	writer<<L"statistic_on_train";
	m_statistic_on_train.write(writer);
	writer<<L"statistic_on_validation";
	m_statistic_on_validation.write(writer);
	
	writer<<L"statistic_iteration_number"<<m_statistic_iteration_number;
	writer<<L"m_iteration_number"<<m_iteration_number;
	writer<<L"m_auto_save_iteration_number"<<m_auto_save_iteration_number;
	writer<<L"m_auto_save_dir_path"<<m_auto_save_dir_path;
	writer<<L"depth"<<mt_mat_helper::depth_str(m_depth);
	writer<<L"m_label_for_categories"<<m_label_for_categories;

	writer<<L"layers"<<L"{";

	ml_helper::write(writer, m_input_layers, m_hidden_layers, m_output_layers, m_linked_layers, write_learned_param);
	
	writer<<L"}";
}

ml_neural_network* ml_neural_network::read(sys_json_reader& reader) {
	ml_neural_network* nn = new ml_neural_network();

	nn->m_statistic_on_train.read(reader[L"statistic_on_train"]);
	nn->m_statistic_on_validation.read(reader[L"statistic_on_validation"]);

	reader[L"statistic_iteration_number"]>>nn->m_statistic_iteration_number;
	reader[L"iteration_number"]>>nn->m_statistic_iteration_number;
	reader[L"auto_save_iteration_number"]>>nn->m_auto_save_iteration_number;
	reader[L"auto_save_dir_path"]>>nn->m_auto_save_dir_path;
	nn->m_depth = mt_mat_helper::depth_i32(reader[L"depth"]);
	reader[L"label_for_categories"]>>nn->m_label_for_categories;

	ml_helper::read(nn->m_input_layers, nn->m_hidden_layers, nn->m_output_layers, nn->m_linked_layers, reader[L"layers"]);

	return nn;
}

void ml_neural_network::add_layer(ml_nn_layer* layer) {
	if (!check_layer_name(layer)) {
		return;
	}
	
	if (layer->to_input_data_layer() != NULL) {
		m_input_layers.push_back(layer);
	} else if (layer->to_output_data_layer() != NULL) {
		m_output_layers.push_back(layer);
	} else if (layer->to_data_layer() != NULL) {
		m_hidden_layers.push_back(layer);
	} else if (layer->to_linked_layer() != NULL) {
		m_linked_layers.push_back(layer);
	} else {
		basiclog_unsupport2();
	}
}

void ml_neural_network::setup() {
	for (int i = 0; i < (int)m_input_layers.size(); ++i) {
		m_input_layers[i]->compute_default_setting();
	}

	for (int i = 0; i < (int)m_linked_layers.size(); ++i) {
		m_linked_layers[i]->to_linked_layer()->init_need_learn_params(m_depth);
	}
}

bool ml_neural_network::has_batch_norm_linked_layer() {
	for (int i = 0; i < (int)m_linked_layers.size(); ++i) {
		if (m_linked_layers[i]->to_batch_norm_linked_layer() != NULL) {
			return true;
		}
	}

	return false;
}

void ml_neural_network::train(const ml_data& training_data, const ml_data& validation_data /* = NULL */) {
	if (!training_data.weight().is_empty()) {
		train(training_data.resample_align_weight(), validation_data);
		return;
	}
	
	if (m_label_for_categories.empty()) {
		training_data.statistic_category(m_label_for_categories);
	}
	
	basiclog_info2(sys_strcombine()<<L"label_for_categories "<<m_label_for_categories);

	ml_nn_layer_learning_params params(sys_false, 0, m_iteration_number);
	
	while (params.m_iteration_index < m_iteration_number) {
		ml_batcher batcher(training_data, m_training_batch_size, sys_true, sys_true);

		for (i32 i = 0; i < batcher.batch_number(); ++i) {
			mt_auto_derivative auto_derivative;

			sys_timer batch_timer(L"batch_timer", sys_false);
			batch_timer.begin();

			map<wstring, mt_mat> features;
			map<wstring, mt_mat> labels;

			params.m_sequence_length = batcher.feature(features, i);
			batcher.label(labels, m_label_for_categories, class_name(), i);

			for (int iter_input = 0; iter_input < (int)m_input_layers.size(); ++iter_input) {
				const wstring& input_layer_name = m_input_layers[iter_input]->name();
				features[input_layer_name].attach(&auto_derivative);
				m_input_layers[iter_input]->to_input_data_layer()->feedforward_by_input(features[input_layer_name], params);
			}

			auto_derivative.record_math_operation(sys_false);

			vector<mt_mat> losses;

			for (int iter_output = 0; iter_output < (int)m_output_layers.size(); ++iter_output) {
				const wstring& output_layer_name = m_output_layers[iter_output]->name();
				mt_mat predicted_label = m_output_layers[iter_output]->to_output_data_layer()->label();
				mt_mat loss = labels[output_layer_name].loss(predicted_label, m_output_layers[iter_output]->to_output_data_layer()->loss_func_type());
				
				i32 valid_sample_number = ml_helper::calculate_valid_sample_number(labels[output_layer_name]);
				loss = loss * m_output_layers[iter_output]->to_output_data_layer()->task_weight() / valid_sample_number;
				losses.push_back(loss);
			}

			batch_timer.end();

			output_batch_loss(losses, batch_timer.get_cost());

			for (int iter_input = 0; iter_input < (int)m_input_layers.size(); ++iter_input) {
				const wstring& input_layer_name = m_input_layers[iter_input]->name();
				m_input_layers[iter_input]->to_input_data_layer()->update_learning_param(losses, params);
			}

			if ((params.m_iteration_index + 1) % m_statistic_iteration_number == 0 || (params.m_iteration_index + 1 == m_iteration_number)) {
				statistic(training_data, validation_data);

				if (!m_auto_save_dir_path.empty()) {
					auto_save(params.m_iteration_index);
				}
			}

			++params.m_iteration_index;
		}
	}
}

void ml_neural_network::predict(ml_predict_result& res, const ml_data& features) const {
	res.set_model_name(class_name());
	res.set_label_for_category(m_label_for_categories);
	ml_batcher batcher(features, m_testing_batch_size, sys_true, sys_false);

	ml_nn_layer_learning_params params(sys_true, 0, batcher.batch_number());

	map<wstring, mt_mat> predict_res;

	for (i32 i = 0; i < batcher.batch_number(); ++i) {
		sys_timer batch_timer(L"batch_timer", sys_false);
		batch_timer.begin();

		map<wstring, mt_mat> batch_features;

		params.m_sequence_length = batcher.feature(batch_features, i);

		for (int iter_input = 0; iter_input < (int)m_input_layers.size(); ++iter_input) {
			const wstring& input_layer_name = m_input_layers[iter_input]->name();
			m_input_layers[iter_input]->to_input_data_layer()->feedforward_by_input(batch_features[input_layer_name], params);
		}

		map<wstring, mt_mat> labels;
		
		for (int iter_output = 0; iter_output < (int)m_output_layers.size(); ++iter_output) {
			const wstring& output_layer_name = m_output_layers[iter_output]->name();

			labels[output_layer_name] = m_output_layers[iter_output]->to_output_data_layer()->label();

			if (features.is_seuqnce(output_layer_name)) {
				labels[output_layer_name] = ml_helper::align_sequence_data(labels[output_layer_name], batch_features.begin()->second);
			}
		}

		batch_timer.end();
	}
}

b8 ml_neural_network::check_layer_name(const ml_nn_layer* layer) const {
	if (layer->to_input_data_layer() != NULL && layer->name().find(ml_Data_Description_Feature) == wstring::npos) {
		basiclog_assert_message2(sys_false, sys_strcombine()<<L"name of input layer must contains "<<ml_Data_Description_Feature);
		
		return sys_false;
	}

	if (layer->to_output_data_layer() != NULL && layer->name().find(ml_Data_Description_Response) == wstring::npos) {
		basiclog_assert_message2(sys_false, sys_strcombine()<<L"name of output layer must contains "<<ml_Data_Description_Response);

		return sys_false;
	}

	for (i32 i = 0; i < (i32)m_input_layers.size(); ++i) {
		if (m_input_layers[i]->name() == layer->name()) {
			basiclog_assert_message2(sys_false, sys_strcombine()<<L"duplicated layer name: "<<layer->name());
			return sys_false;
		}
	}

	for (i32 i = 0; i < (i32)m_output_layers.size(); ++i) {
		if (m_output_layers[i]->name() == layer->name()) {
			basiclog_assert_message2(sys_false, sys_strcombine()<<L"duplicated layer name: "<<layer->name());
			return sys_false;
		}
	}

	for (i32 i = 0; i < (i32)m_hidden_layers.size(); ++i) {
		if (m_hidden_layers[i]->name() == layer->name()) {
			basiclog_assert_message2(sys_false, sys_strcombine()<<L"duplicated layer name: "<<layer->name());
			return sys_false;
		}
	}

	for (i32 i = 0; i < (i32)m_linked_layers.size(); ++i) {
		if (m_linked_layers[i]->name() == layer->name()) {
			basiclog_assert_message2(sys_false, sys_strcombine()<<L"duplicated layer name: "<<layer->name());
			return sys_false;
		}
	}

	return sys_true;
}

void ml_neural_network::output_batch_loss(const vector<mt_mat>& losses, i64 cost_time) const {
	mt_mat batch_loss_sum = mt_mat_helper::add(losses);

	wstring info = sys_strcombine()<<L"batch cost: "<<cost_time<<L", loss sum: "<<batch_loss_sum.get(0, 0)[0];

	for (int iter_output = 0; iter_output < (int)m_output_layers.size(); ++iter_output) {
		const wstring& output_layer_name = m_output_layers[iter_output]->name();

		info += sys_strcombine()<<L", task "<<output_layer_name<<L", loss: "<<losses[iter_output].get(0, 0)[0];
	}

	basiclog_info2(info);
}

void ml_neural_network::statistic(const ml_data& training_data, const ml_data& validation_data) {
	if (has_batch_norm_linked_layer()) {
		batch_norm_compute_mean_variance(training_data);
	}

	evaluate(m_statistic_on_train, training_data);
	evaluate(m_statistic_on_validation, validation_data);
}

void ml_neural_network::evaluate(ml_statistic_info& statistic_info, const ml_data& data) const {
	ml_predict_result res;
	predict(res, data);

	map<wstring, mt_mat> response;
	data.response(response);

	res.statistic_from_response(statistic_info, response);
}

void ml_neural_network::auto_save(i32 iteration_nuumber) {

}

void ml_neural_network::batch_norm_compute_mean_variance(const ml_data& training_data) {
	ml_batcher batcher(training_data, m_training_batch_size, sys_true, sys_true);
	ml_nn_layer_learning_params params(sys_true, 0, batcher.batch_number(), sys_true);

	map<wstring, i32> sequence_lengths;

	for (; params.m_iteration_index < params.m_total_iteration_number; ++params.m_iteration_index) {
		sys_timer batch_timer(L"batch_timer", sys_false);
		batch_timer.begin();

		map<wstring, mt_mat> batch_features;

		params.m_sequence_length = batcher.feature(batch_features, params.m_iteration_index);

		for (int iter_input = 0; iter_input < (int)m_input_layers.size(); ++iter_input) {
			const wstring& input_layer_name = m_input_layers[iter_input]->name();
			m_input_layers[iter_input]->to_input_data_layer()->feedforward_by_input(batch_features[input_layer_name], params);
		}

		batch_timer.end();
	}
}


ml_algorithm* ml_neural_network::clone() const {
	ml_neural_network* nn = new ml_neural_network();

	ml_helper::clone_layer(nn->m_input_layers, nn->m_hidden_layers, nn->m_output_layers, nn->m_linked_layers, m_input_layers, m_hidden_layers, m_output_layers, m_linked_layers);
	
	nn->m_statistic_on_train = m_statistic_on_train;
	nn->m_statistic_on_validation = m_statistic_on_validation;
	
	nn->m_statistic_iteration_number = m_statistic_iteration_number;
	nn->m_iteration_number = m_iteration_number;
	nn->m_auto_save_iteration_number = m_auto_save_iteration_number;
	nn->m_auto_save_dir_path = m_auto_save_dir_path;
	nn->m_auto_save_as_text = m_auto_save_as_text;
	nn->m_training_batch_size = m_training_batch_size;
	nn->m_testing_batch_size = m_testing_batch_size;

	return nn;
}