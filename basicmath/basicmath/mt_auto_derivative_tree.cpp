#include "stdafx.h"
#include "mt_auto_derivative_tree.h"


class private_ad_helper {
public:

	template<class T>
	static mt_mat softmax_derivate(const mt_mat& softmax_res) {
		mt_mat res(softmax_res, mt_mat::Construct_Type_Create_As_Size);
		const u8* ptr_softmax_res_dim0 = softmax_res.data();
		u8* ptr_derivate_dim0 = res.data();

		for (i32 row = 0; row < softmax_res.size()[0]; ++row) {
			const T* ptr_softmax_res_dim1 = (const T*)ptr_softmax_res_dim0;
			T* ptr_derivate_dim1 = (T*)ptr_derivate_dim0;

			for (i32 col = 0; col < softmax_res.size()[1]; ++col) {
				for (i32 i = 0; i < softmax_res.size()[1]; ++i) {
					if (col == i) {
						ptr_derivate_dim1[col] += ptr_softmax_res_dim1[i] * (1 - ptr_softmax_res_dim1[i]);
					} else {
						ptr_derivate_dim1[col] += -ptr_softmax_res_dim1[i] * ptr_softmax_res_dim1[col];
					}
				}
			}

			ptr_softmax_res_dim0 += softmax_res.step()[0];
			ptr_derivate_dim0 += res.step()[0];
		}

		return res;
	}

	template<class T>
	static mt_mat relu_derivate(const mt_mat& relu_res, f64 negative_slope) {
		mt_mat res(relu_res, mt_mat::Construct_Type_Create_As_Size);

		mt_array_element_const_iterator relu_res_iter(relu_res);
		mt_array_element_iterator res_iter(res);

		for (;;) {
			const T* ptr_relu_res = (const T*)relu_res_iter.data();

			if (ptr_relu_res == NULL) {
				break;
			}

			T* ptr_res = (T*)res_iter.data();

			for (i32 c = 0; c < relu_res.channel(); ++c) {
				if (ptr_relu_res[c] > 0) {
					ptr_res[c] = 1;
				} else {
					ptr_res[c] = (T)negative_slope;
				}
			}
		}

		return res;
	}
};

mt_mat mt_ad_sub_stride_tree_node::derivate_child_on_operation( mt_ad_mat_tree_node* child_node, const mt_mat& derivative_res) {
	mt_mat child_derivative = mt_mat(child_node->m_mat, mt_mat::Construct_Type_Create_As_Size);

	mt_mat temp = child_derivative.sub_stride((i32)m_strides.size(), &m_strides[0]);
	temp.set(derivative_res);

	return child_derivative;
}

mt_mat mt_ad_conv_tree_node::derivate_child_on_operation(mt_ad_mat_tree_node* child_node, const mt_mat& derivative_res) {
	mt_mat derivate_mat = derivative_res;
	
	i32 half_index = (i32)m_childs.size() / 2;

	if (!m_strides.empty()) {
		basicmath_mat_request_memory(i32, valid_conv_sizes, derivate_mat.dim());
		mt_mat_helper::get_conv_result_size(derivate_mat.dim(), valid_conv_sizes, m_childs[0]->to_mat_tree_node()->m_mat.size(), m_childs[half_index]->to_mat_tree_node()->m_mat.size(), NULL, mt_Conv_Boundary_Type_Valid);
		derivate_mat = derivative_res.unpooling(valid_conv_sizes, mt_mat(), mt_Pooling_Type_First_Value, (i32)m_strides.size(), &m_strides[0], &m_strides[0]);
	}

	for (i32 i = 0; i < (i32)half_index; ++i) {
		if (child_node == m_childs[i]) {
			return derivate_mat.conv(m_childs[i + half_index]->to_mat_tree_node()->m_mat.flip_all_dim(), mt_Conv_Boundary_Type_Full);
		}
	}

	for (i32 i = half_index; i < (i32)m_childs.size(); ++i) {
		if (child_node == m_childs[i]) {
			return m_childs[i - half_index]->to_mat_tree_node()->m_mat.flip_all_dim().conv(derivate_mat, mt_Conv_Boundary_Type_Valid);			
		}
	}

	basiclog_assert2(sys_false);
	return mt_mat();
}

mt_mat mt_ad_log_tree_node::derivate_child_on_operation(mt_ad_mat_tree_node* child_node, const mt_mat& derivative_res) {
	mt_mat child_derivative = 1 / child_node->m_mat;
	
	if (mt_helper::compare_double(m_base, mt_E) != 0) {
		child_derivative *= log(m_base);	
	}

	return child_derivative * derivative_res;
}

mt_mat mt_ad_activate_tree_node::derivate_child_on_operation(mt_ad_mat_tree_node* child_node, const mt_mat& derivative_res) {
	switch (m_activate_type) {
	case mt_Activate_Type_Linear:
		return derivative_res.clone();
	case mt_Activate_Type_Sigmoid:
		return derivative_res * (1 - child_node->m_mat) * child_node->m_mat;
	case mt_Activate_Type_Softmax:
		return derivative_res * softmax_derivate(child_node->m_mat);
	case mt_Activate_Type_Relu:
		return derivative_res * relu_derivate(child_node->m_mat);
	default:
		basiclog_unsupport2();
		return mt_mat();
	}
}

mt_mat mt_ad_activate_tree_node::softmax_derivate(const mt_mat& softmax_res) {
	basiclog_assert2(softmax_res.dim() == 2);
	basiclog_assert2(softmax_res.channel() == 1);

	if (softmax_res.depth() == mt_F32) {
		return private_ad_helper::softmax_derivate<f32>(softmax_res);
	} else if (softmax_res.depth() == mt_F64) {
		return private_ad_helper::softmax_derivate<f64>(softmax_res);
	} else {
		basiclog_unsupport2();
		return mt_mat();
	}
}

mt_mat mt_ad_activate_tree_node::relu_derivate(const mt_mat& softmax_res) {
	if (softmax_res.depth() == mt_F32) {
		return private_ad_helper::relu_derivate<f32>(softmax_res, m_activated_params.empty() ? 0 : m_activated_params[0]);
	} else if (softmax_res.depth() == mt_F64) {
		return private_ad_helper::relu_derivate<f64>(softmax_res, m_activated_params.empty() ? 0 : m_activated_params[0]);
	} else {
		basiclog_unsupport2();
		return mt_mat();
	}
}

mt_mat mt_ad_loss_tree_node::derivate_child_on_operation(mt_ad_mat_tree_node* child_node, const mt_mat& derivative_res) {
	switch (m_loss_type) {
	case mt_Loss_Type_0_1:
		return derivative_res.clone();
	case mt_Loss_Type_Logarithmic:
		
		if (child_node == m_childs[0]) {
			return derivative_res * ((m_childs[0]->to_mat_tree_node()->m_mat - m_childs[1]->to_mat_tree_node()->m_mat) / (m_childs[0]->to_mat_tree_node()->m_mat * (1 - m_childs[0]->to_mat_tree_node()->m_mat)));
		} else if (child_node == m_childs[1]) {
			return derivative_res * (m_childs[0]->to_mat_tree_node()->m_mat.self_log() - (1 - m_childs[0]->to_mat_tree_node()->m_mat).self_log());
		}

	case mt_Loss_Type_Quadratic:
		
		if (child_node == m_childs[0]) {
			return derivative_res * (m_childs[0]->to_mat_tree_node()->m_mat - m_childs[1]->to_mat_tree_node()->m_mat);
		} else if (child_node == m_childs[1]) {
			return derivative_res * (m_childs[1]->to_mat_tree_node()->m_mat - m_childs[0]->to_mat_tree_node()->m_mat);
		}
	default:
		basiclog_unsupport2();
		return mt_mat();
	}
}