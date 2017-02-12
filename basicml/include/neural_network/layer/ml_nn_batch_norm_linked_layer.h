#pragma once

#include "ml_nn_linked_layer.h"

namespace basicml {
	class ml_nn_batch_norm_linked_layer : public ml_nn_linked_layer {
	public:

		enum Batch_Norm_Type {
			Batch_Norm_Type_Unit,
			Batch_Norm_Type_Channel,
		};

		virtual ~ml_nn_batch_norm_linked_layer() {}

		void init_need_learn_params(int data_type);

		virtual void feedforward(const ml_nn_layer_learning_params& pars);
		void update_learning_param(const vector<mt_mat>& losses, const ml_nn_layer_learning_params& pars);

		ml_nn_layer* clone() const;

		void set_regular_term(f64 term) {
			m_regular_term = term;
		}

		virtual ml_nn_batch_norm_linked_layer* to_batch_norm_linked_layer() {return this;}
		virtual const ml_nn_batch_norm_linked_layer* to_batch_norm_linked_layer() const {return this;}

		virtual void write(sys_json_writer& writer, b8 write_learned_param = sys_true) const;
		static ml_nn_batch_norm_linked_layer* read(const sys_json_reader& reader);

	protected:

		virtual void inner_compute_default_setting();

		vector<mt_mat> m_gmmas;
		vector<mt_mat> m_betas;

		Batch_Norm_Type m_batch_norm_type;

		f64 m_regular_term;

		vector<mt_mat> m_total_train_means;
		vector<mt_mat> m_total_train_variances;
	};
}