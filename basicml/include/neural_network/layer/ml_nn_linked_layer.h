#pragma once

#include "ml_nn_layer.h"
#include "ml_nn_layer_learning_params.h"

namespace basicml {
	
	class ml_learning_param_updater;

	class ml_nn_linked_layer : public ml_nn_layer {
	public:

		ml_nn_linked_layer()
			: m_output(NULL)
			, m_input(NULL) {
		}

		ml_nn_linked_layer(const wstring& layer_name, ml_nn_data_layer* input_layer, ml_nn_data_layer* output_layer, ml_learning_param_updater* weight_updater = NULL, ml_learning_param_updater* bias_updater = NULL);

		virtual ~ml_nn_linked_layer();

		ml_nn_linked_layer* to_linked_layer() {return this;}
		const ml_nn_linked_layer* to_linked_layer() const {return this;}

		void set_input(ml_nn_data_layer* input);
		void set_output(ml_nn_data_layer* output);

		ml_nn_data_layer* get_input() {
			return m_input;
		}

		ml_nn_data_layer* get_out() {
			return m_output;
		}

		const ml_nn_data_layer* get_input() const {
			return m_input;
		}

		const ml_nn_data_layer* get_out() const {
			return m_output;
		}

		const wstring& input_name() const;

		const wstring& output_name() const;

		void compute_default_setting();

		virtual void init_need_learn_params(int data_type) {
			basiclog_assert2(m_input != NULL && m_output != NULL);	
		}

		ml_nn_data_layer* get_input_layer() const {
			return m_input;
		}

		ml_nn_data_layer* get_output_layer() const {
			return m_output;
		}

		virtual void copy_learned_param(const ml_nn_linked_layer* other) {};

		virtual void feedforward(const ml_nn_layer_learning_params& pars) {};
		
		virtual void update_learning_param(const vector<mt_mat>& losses, const ml_nn_layer_learning_params& pars) {};

		void set_weight_updater(ml_learning_param_updater* updater) {
			m_weight_updater = updater;
		}

		ml_learning_param_updater* weight_updater() const {
			return m_weight_updater;
		}

		void set_bias_updater(ml_learning_param_updater* updater) {
			m_bias_updater = updater;
		}

		ml_learning_param_updater* bias_updater() const {
			return m_bias_updater;
		}

		virtual void write(sys_json_writer& writer, b8 write_learned_param = sys_true) const;
		static ml_nn_linked_layer* read(const sys_json_reader& reader);

	protected:

		virtual void inner_compute_default_setting() {}

		ml_nn_data_layer* m_input;
		ml_nn_data_layer* m_output;

		ml_learning_param_updater* m_weight_updater;
		ml_learning_param_updater* m_bias_updater;

		wstring m_input_name;
		wstring m_output_name;
	};
}