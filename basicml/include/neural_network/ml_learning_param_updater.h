#pragma once

#include "ml_types.h"

namespace basicml {

	class ml_bsgd_learning_param_updater;

	enum ml_Penalty_Type {
		ml_Penalty_Type_Null = 0,
		ml_Penalty_Type_L1,
		ml_Penalty_Type_L2,
	};

	static const wstring ml_Penalty_Type_Descriptions[] = {L"null", L"l1", L"l2"};

	class ml_learning_param_updater {
	public:

		basicsys_class_name_method(ml_learning_param_updater)

		ml_learning_param_updater() {
			m_init_type = ml_Learning_Param_Init_Type_Gaussian;
			m_init_params.push_back(0);
			m_init_params.push_back(1);
		}

		virtual void update(mt_mat& learning_param, mt_mat& gradient, const ml_nn_layer_learning_params& pars) = 0;

		void update(vector<mt_mat>& learning_params, vector<mt_mat>& gradients, const ml_nn_layer_learning_params& pars) {
			for (i32 i = 0; i < (i32)learning_params.size(); ++i) {
				update(learning_params[i], gradients[i], pars);
			}
		}
		
		void init(mt_mat& learning_param) {
			vector<mt_mat> learning_params;
			init(learning_params);
		}

		virtual void init(vector<mt_mat>& learning_params) = 0;

		void set_init_config(ml_Learning_Param_Init_Type type, const vector<f64>& params) {
			m_init_type = type;
			m_init_params = params;
		}

		ml_Learning_Param_Init_Type init_type() const {
			return m_init_type;
		}

		const vector<f64>& init_param() const {
			return m_init_params;
		}

		virtual void on_copy_from_other() {}

		virtual ml_bsgd_learning_param_updater* to_bsgd_updater() {return NULL;}
		const virtual ml_bsgd_learning_param_updater* to_bsgd_updater() const {return NULL;}

		virtual void write(sys_json_writer& writer, b8 write_learned_param) const = 0;
		static ml_learning_param_updater* read(const sys_json_reader& reader);

	protected:

		void add_penalty(mt_mat& gradient, mt_mat& weight, ml_Penalty_Type type, f64 alpha);

		ml_Learning_Param_Init_Type m_init_type;
		vector<f64> m_init_params;
	};


	
}