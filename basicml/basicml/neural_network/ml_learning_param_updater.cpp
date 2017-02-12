#include "stdafx.h"

#include "ml_learning_param_updater.h"

namespace basicml {

	class private_ml_updater {
	public:

		template<class T>
		static void add_l1_penalty(mt_mat& gradient, const mt_mat& learning_param, f64 alpha) {
			const u8* ptr_learning_param_dim0 = learning_param.data();
			u8* ptr_gradient_dim0 = gradient.data();

			for (int row = 0; row < gradient.size()[0]; ++row) {
				const T* ptr_learning_param_dim1 = (const T*)ptr_learning_param_dim0;
				T* ptr_gradient_dim1 = (T*)ptr_gradient_dim0;

				for (int col = 0; col < learning_param.size()[1]; ++col) {
					if (*ptr_learning_param_dim1 > 0) {
						*ptr_gradient_dim1 += (T)alpha;
					} else {
						*ptr_gradient_dim1 += -(T)alpha;
					}

					++ptr_learning_param_dim1;
					++ptr_gradient_dim1;
				}

				ptr_learning_param_dim0 += learning_param.step()[0];
				ptr_gradient_dim0 += gradient.step()[0];
			}
		}
	};
}

void ml_learning_param_updater::add_penalty(mt_mat& gradient, mt_mat& learning_param, ml_Penalty_Type type, f64 alpha) {
	if (type == ml_Penalty_Type_L1) {
		if (gradient.depth() == mt_F32) {
			private_ml_updater::add_l1_penalty<f32>(gradient, learning_param, alpha);
		} else if (gradient.depth() == mt_F64) {
			private_ml_updater::add_l1_penalty<f64>(gradient, learning_param, alpha);
		} else {
			basiclog_assert2(sys_false);
		}
	} else if (type == ml_Penalty_Type_L2) {
		gradient += learning_param * alpha;
	}
}

ml_learning_param_updater* ml_learning_param_updater::read(const sys_json_reader& reader) {

}