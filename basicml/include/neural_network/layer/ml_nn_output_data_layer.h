#pragma once

#include "ml_nn_data_layer.h"

namespace basicml {
	class ml_nn_output_data_layer : public ml_nn_data_layer {
	public:

		ml_nn_output_data_layer() {

		}

		ml_nn_output_data_layer(const wstring& layer_name, i32 unit_number, mt_Activate_Type activate_type = mt_Activate_Type_Softmax, mt_Loss_Type loss_type = mt_Loss_Type_Logarithmic, b8 classifier = sys_true, f64 task_weight = 1.0)
			: ml_nn_data_layer(layer_name, unit_number, activate_type) {
				m_loss_func_type = loss_type;
				m_is_classifier = classifier;
				m_task_weight = task_weight;
		}

		/** Get loss from output layer

		@return loss mat
		*/
		mt_mat label() const;

		void set_is_classifier(bool classifier) {
			m_is_classifier = classifier;
		}

		b8 is_classifier() {
			return m_is_classifier;
		}

		void set_loss_func_type(mt_Loss_Type loss_func_type) {
			m_loss_func_type = loss_func_type;
		}

		mt_Loss_Type loss_func_type() const {
			return m_loss_func_type;
		}

		void set_task_weight(double weight) {
			m_task_weight = weight;
		}

		double task_weight() const {
			return m_task_weight;
		}

		ml_nn_output_data_layer* to_output_data_layer() {
			return this;
		}

		const ml_nn_output_data_layer* to_output_data_layer() const {
			return this;
		}

		ml_nn_layer* clone() const;

		virtual void write(sys_json_writer& writer, b8 write_learned_param = sys_true) const;
		static ml_nn_output_data_layer* read(const sys_json_reader& reader);

	protected:

		mt_Loss_Type m_loss_func_type;
		double m_task_weight;
		b8 m_is_classifier;
	};
}