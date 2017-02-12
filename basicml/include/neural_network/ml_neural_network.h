#pragma once

#include "ml_supervised_algorithm.h"

namespace basicml {
	class ml_nn_layer_learning_params;
	class ml_nn_layer_creator;
	class ml_nn_layer;
	class ml_learning_data;

	class ml_neural_network : public ml_supervised_algorithm {
	public:

		basicsys_class_name_method(ml_neural_network)

		ml_neural_network() {
			m_iteration_number = 2000;
			m_statistic_iteration_number = 100;
			m_auto_save_as_text = sys_false;
			m_training_batch_size = 100;
			m_testing_batch_size = 1000;
		}

		~ml_neural_network();

		void set_statistic_iteration_number(i32 number) {
			m_statistic_iteration_number = number;
		}

		void set_iteration_number(i32 number) {
			m_iteration_number = number;
		}

		void set_auto_save_iteration_number(i32 number) {
			m_auto_save_iteration_number = number;
		}

		void set_auto_save(const wstring& dir_path, b8 save_as_text = sys_true) {
			m_auto_save_dir_path = dir_path;
			m_auto_save_as_text = save_as_text;
		}

		void set_training_batch_size(i32 batch_size) {
			m_training_batch_size = batch_size;
		}

		void set_testing_batch_size(i32 batch_szie) {
			m_testing_batch_size = batch_szie;
		}
		
		void add_layer(ml_nn_layer* layer);

		void setup();

		void train(const ml_data& training_data, const ml_data& validation_data = ml_data());
		void predict(ml_predict_result& res, const ml_data& features) const;
		void evaluate(ml_statistic_info& statistic_info, const ml_data& data) const;

		virtual void write(const wstring& path, b8 text_type = sys_true, b8 write_learned_param = sys_true) const;
		virtual void write(sys_json_writer& writer, b8 write_learned_param = sys_true) const;
		static ml_neural_network* read(const wstring& path, b8 text_type = sys_true);
		static ml_neural_network* read(sys_json_reader& reader);

		virtual b8 empty() {return sys_true;}

		ml_algorithm* clone() const;

	protected:

		void output_batch_loss(const vector<mt_mat>& losses, i64 cost_time) const;
		void statistic(const ml_data& training_data, const ml_data& validation_data);
		
		void auto_save(i32 iteration_nuumber);
		void batch_norm_compute_mean_variance(const ml_data& training_data);

		void create_layer();
		void link_data_layer();

		bool has_batch_norm_linked_layer();
		bool has_sequence_linked_layer();
		
		void remove_all_zero_out_degree_nonoutput_layer(vector<ml_nn_layer*>& data_layers);

		void write(sys_buffer_writer* buffer_writer, b8 write_learned_params) const;
		static ml_neural_network* read(sys_buffer_reader* buffer_reader);

		ml_statistic_info m_statistic_on_train;
		ml_statistic_info m_statistic_on_validation;

		vector<ml_nn_layer*> m_input_layers;
		vector<ml_nn_layer*> m_hidden_layers;
		vector<ml_nn_layer*> m_output_layers;

		vector<ml_nn_layer*> m_linked_layers;
		

		i32 m_statistic_iteration_number;
		i32 m_iteration_number;
		i32 m_auto_save_iteration_number;
		wstring m_auto_save_dir_path;
		i32 m_auto_save_as_text;
		i32 m_training_batch_size;
		i32 m_testing_batch_size;
	};
}