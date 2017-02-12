#pragma once


#include "ml_learning_param_updater.h"

namespace basicml {

	class ml_bsgd_learning_param_updater : public ml_learning_param_updater {
	public:

		ml_bsgd_learning_param_updater(f64 learning_ratio = 0.01, f64 learning_ratio_scaled_ratio = 0.99, f64 momentum = 0.5, ml_Penalty_Type penalty_type = ml_Penalty_Type_Null, f64 penalty_alpha = 0) {
			m_learning_ratio = learning_ratio;
			m_learning_ratio_scaled_ratio = learning_ratio_scaled_ratio;
			m_momentum = momentum;
			m_penalty_type = penalty_type;
			m_penalty_alpha = penalty_alpha;
		}

		void update(mt_mat& learning_param, mt_mat& gradient, const ml_nn_layer_learning_params& pars);

		void init(vector<mt_mat>& learning_params);

		virtual ml_bsgd_learning_param_updater* to_bsgd_updater() {return this;}
		const virtual ml_bsgd_learning_param_updater* to_bsgd_updater() const {return this;}

		void on_copy_from_other();

		virtual void write(sys_json_writer& writer, b8 write_learned_param) const;

		vector<mt_mat> m_learning_params;
		vector<mt_mat> m_v_gradients;

		i32 m_auto_scaled_iteration_number;
		f64 m_learning_ratio;
		f64 m_learning_ratio_scaled_ratio;
		f64 m_momentum;

		ml_Penalty_Type m_penalty_type;
		f64 m_penalty_alpha;
	};

}