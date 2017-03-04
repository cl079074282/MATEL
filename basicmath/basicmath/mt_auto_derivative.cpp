#include "stdafx.h"

#include "mt_auto_derivative.h"
#include "mt_mat_helper.h"

mt_auto_derivative::~mt_auto_derivative() {
	basicsys_delete_vector_ptr(m_nodes);
}

mt_mat mt_auto_derivative::derivate(const mt_mat& target, const mt_mat& src) {
	basiclog_assert2(target.auto_derivative() != NULL && target.auto_derivative() == src.auto_derivative());

	mt_mat res(target, mt_mat::Construct_Type_Create_As_Size);

	if (src.is_same(target)) {
		res.set(1);
	} else {
		res.set(0);

		m_computed_flag = sys_false;

		for (i32 i = 0; i < (i32)m_nodes.size(); ++i) {
			if (m_nodes[i]->match(src)) {
				m_nodes[i]->to_mat_tree_node()->add_derivative_res(mt_mat(src, mt_mat::Construct_Type_Create_As_Size).set(1), src);

				derivate(res, target, src, m_nodes[i]);
				break;
			}
		}

		if (!m_computed_flag) {
			basiclog_warning2(L"there is no target mat in the computation process!");
		}
	}

	return res;
}

mt_mat mt_auto_derivative::derivate(const mt_mat& target, const vector<mt_mat>& srcs) {
	mt_mat res = derivate(target, srcs[0]);

	for (i32 i = 1; i < (i32)srcs.size(); ++i) {
		res += derivate(target, srcs[i]);
	}

	return res;
}

void mt_auto_derivative::derivate(vector<mt_mat>& reses, const vector<mt_mat>& targets, const mt_mat& src) {
	reses.resize(targets.size());

	for (i32 i = 0; i < (i32)targets.size(); ++i) {
		reses[i] = derivate(targets[i], src);
	}
}

vector<mt_mat> mt_auto_derivative::derivate(vector<mt_mat>& targets, const mt_mat& src) {
	vector<mt_mat> reses;
	derivate(reses, targets, src);

	return reses;
}

void mt_auto_derivative::derivate(vector<mt_mat>& reses, const vector<mt_mat>& targets, const vector<mt_mat>& srcs) {
	derivate(reses, targets, srcs[0]);

	for (i32 i = 1; i < (i32)srcs.size(); ++i) {
		for (i32 j = 0; j < (i32)targets.size(); ++j) {
			reses[j] += derivate(targets[i], srcs[i]);
		}
	}
}

vector<mt_mat> mt_auto_derivative::derivate(vector<mt_mat>& targets, const vector<mt_mat>& srcs) {
	vector<mt_mat> reses;
	derivate(reses, targets, srcs);

	return reses;
}

void mt_auto_derivative::clone(const mt_mat& res, const mt_mat& src) {
	m_nodes.push_back(new mt_ad_clone_tree_node(res, get_node(src), m_max_cache_size));
}

void mt_auto_derivative::add(const mt_mat& res, const mt_mat& a, const vector<double>& b) {
	m_nodes.push_back(new mt_ad_add_tree_node(res, get_node(a), get_node(b), m_max_cache_size));
}

void mt_auto_derivative::add(const mt_mat& res, const mt_mat& a, const mt_mat& b) {
	m_nodes.push_back(new mt_ad_add_tree_node(res, get_node(a), get_node(b), m_max_cache_size));
}

void mt_auto_derivative::add(const mt_mat& res, const vector<mt_mat>& elements) {
	vector<mt_ad_tree_node*> child_nodes;
	child_nodes.resize(elements.size());

	for (i32 i = 0; i < (i32)elements.size(); ++i) {
		child_nodes.push_back(get_node(elements[i]));
	}

	m_nodes.push_back(new mt_ad_add_tree_node(res, child_nodes, m_max_cache_size));
}

void mt_auto_derivative::subtract(const mt_mat& res, const mt_mat& a, const mt_mat& b) {
	m_nodes.push_back(new mt_ad_subtract_tree_node(res, get_node(a), get_node(b), m_max_cache_size));
}

void mt_auto_derivative::mul(const mt_mat& res, const mt_mat& a, const mt_mat& b) {
	m_nodes.push_back(new mt_ad_mul_tree_node(res, get_node(a), get_node(b), m_max_cache_size));
}

void mt_auto_derivative::cnov(const mt_mat& res, const mt_mat& src, const mt_mat& kernel, mt_Conv_Boundary_Type boundary_type, i32 size, const i32* strides) {
	if (boundary_type != mt_Conv_Boundary_Type_Valid) {
		basiclog_unsupport2();
	}
	
	m_nodes.push_back(new mt_ad_conv_tree_node(res, get_node(src), get_node(kernel), size, strides, m_max_cache_size));
}

void mt_auto_derivative::flip(const mt_mat& res, const mt_mat& src, i32 size, const b8* flip_flags) {
	m_nodes.push_back(new mt_ad_flip_tree_node(res, get_node(src), size, flip_flags, m_max_cache_size));
}

void mt_auto_derivative::reshape(const mt_mat& res, const mt_mat& src) {
	m_nodes.push_back(new mt_ad_reshape_tree_node(res, get_node(src), m_max_cache_size));
}

void mt_auto_derivative::sub(const mt_mat& res, const mt_mat& a, i32 size, const mt_range* ranges) {
	m_nodes.push_back(new mt_ad_sub_tree_node(res, get_node(a), size, ranges, m_max_cache_size));
}

void mt_auto_derivative::sub_stride(const mt_mat& res, const mt_mat& a, i32 size, const i32* strides) {
	m_nodes.push_back(new mt_ad_sub_stride_tree_node(res, get_node(a), size, strides, m_max_cache_size));
}

void mt_auto_derivative::expand(const mt_mat& res, const mt_mat& a, i32 size, const mt_range* ranges) {
	m_nodes.push_back(new mt_ad_expand_tree_node(res, get_node(a), size, ranges, m_max_cache_size));
}

void mt_auto_derivative::repeat(const mt_mat& res, const mt_mat& a) {
	m_nodes.push_back(new mt_ad_repeat_tree_node(res, get_node(a), m_max_cache_size));
}

void mt_auto_derivative::activate(const mt_mat& res, const mt_mat& src, mt_Activate_Type type, const vector<f64>& activate_params) {
	m_nodes.push_back(new mt_ad_activate_tree_node(res, get_node(src), type, activate_params, m_max_cache_size));
}

void mt_auto_derivative::exp(const mt_mat& res, const mt_mat& src) {
	m_nodes.push_back(new mt_ad_exp_tree_node(res, get_node(src), m_max_cache_size));
}

void mt_auto_derivative::pow(const mt_mat& res, const mt_mat& src, f64 number) {
	m_nodes.push_back(new mt_ad_pow_tree_node(res, get_node(src), number, m_max_cache_size));
}

void mt_auto_derivative::log(const mt_mat& res, const mt_mat& src, f64 base) {
	m_nodes.push_back(new mt_ad_log_tree_node(res, get_node(src), base, m_max_cache_size));
}

void mt_auto_derivative::record_math_operation(b8 enable) {
	m_enable_math_operation = enable;
}

b8 mt_auto_derivative::math_operation_recorded() const {
	return m_enable_math_operation;
}

void mt_auto_derivative::reset() {
	basicsys_delete_vector_ptr(m_nodes);
	m_enable_math_operation = sys_true;
}

mt_ad_tree_node* mt_auto_derivative::find_same(const mt_mat& mat) const {
	for (int i = 0; i < (int)m_nodes.size(); ++i) {
		if (m_nodes[i]->match(mat)) {
			return m_nodes[i];
		}
	}

	return NULL;
}

mt_ad_tree_node* mt_auto_derivative::get_node(const mt_mat& mat) {
	mt_ad_tree_node* node = find_same(mat);

	if (node == NULL) {
		node = new mt_ad_leaf_tree_node(mat);
		m_nodes.push_back(node);
	}

	return node;
}

mt_ad_tree_node* mt_auto_derivative::get_node(const vector<double>& const_value) {
	mt_ad_tree_node* node = new mt_ad_const_leaf_tree_node(const_value);
	return node;
}

//void mt_auto_derivative::get_path_from_target_to_src(vector<mt_ad_mat_tree_node*>& path_nodes, mt_ad_tree_node* target_node, mt_ad_tree_node* src_node) {
//	while (target_node != src_node && target_node != NULL) {
//		path_nodes.push_back((mt_ad_mat_tree_node*)target_node);
//
//		target_node = target_node->m_parent;
//	}
//
//	if (target_node == NULL) {
//		path_nodes.clear();
//	} else {
//		path_nodes.push_back((mt_ad_mat_tree_node*)src_node);
//	}
//}

void mt_auto_derivative::derivate(mt_mat& derivated_mat, const mt_mat& target_mat, const mt_mat& src_mat, mt_ad_tree_node* src_node) {
	if (src_node->m_childs.empty()) {
		return;
	}

	for (i32 i = 0; i < (i32)src_node->m_childs.size(); ++i) {
		if (src_node->m_childs[i]->match(target_mat)) {
			src_node->to_mat_tree_node()->derivate_child(src_node->m_childs[i]->to_mat_tree_node(), src_mat);

			derivated_mat += src_node->m_childs[i]->to_mat_tree_node()->derivative_res(src_mat);

			m_computed_flag = sys_true;

		} else if (is_need_derivate(target_mat, src_node->m_childs[i])) {
			src_node->to_mat_tree_node()->derivate_child(src_node->m_childs[i]->to_mat_tree_node(), src_mat);
			derivate(derivated_mat, target_mat, src_mat, src_node->m_childs[i]);
		}
	}
}

b8 mt_auto_derivative::is_need_derivate(const mt_mat& target_mat, mt_ad_tree_node* child_node) const {
	if (child_node->to_mat_tree_node() == NULL) {
		return sys_false;
	}

	for (i32 i = 0; i < (i32)child_node->to_mat_tree_node()->m_childs.size(); ++i) {
		if (child_node->to_mat_tree_node()->m_childs[i]->match(target_mat)) {
			return sys_true;
		} else {
			return is_need_derivate(target_mat, child_node->to_mat_tree_node()->m_childs[i]);
		}
	}

	return sys_false;
}

//
//void mt_auto_derivative::derivate(mt_mat& derivated_mat, const mt_mat& target_mat, mt_ad_mat_tree_node* target_node, mt_ad_tree_node* src_node) {
//	if (target_node->m_derivate_src_node != src_node) {
//		vector<mt_ad_mat_tree_node*> path_nodes;
//		get_path_from_target_to_src(path_nodes, target_node, src_node);
//
//		for (i32 i = (i32)path_nodes.size() - 1; i > 0; --i) {
//			path_nodes[i]->derivate_child(m_mat_cache, path_nodes[i - 1]);
//		}
//	}
//
//	//The derivate mat may be bigger or smaller than the derivate mat in the target node (cause by sub() method) 
//	if (derivated_mat.element_number() > target_node->m_derivated_mat.element_number()) {
//		vector<int> indexs_in_target_mat;
//		mt_array_element_iterator iter_target_derivative_mat(target_node->m_derivated_mat);
//
//		for (;;) {
//			u8* ptr_data = iter_target_derivative_mat.data();
//
//			if (ptr_data == NULL) {
//				break;
//			}
//
//			u8* ptr_node_mat_data = target_node->m_mat.ptr<u8>(iter_target_derivative_mat.position());
//			target_mat.get_index(indexs_in_target_mat, ptr_node_mat_data);
//
//			if (derivated_mat.valid_index(indexs_in_target_mat)) {
//				u8* ptr_derivated_data = derivated_mat.ptr<u8>(indexs_in_target_mat);
//				mt_mat_helper::data_operation(mt_mat_helper::Math_Op_Code_Add, derivated_mat.depth_channel(), ptr_derivated_data, ptr_derivated_data, ptr_data);
//			}
//		}
//	} else {
//		vector<int> indexs_in_node_data_mat;
//		mt_array_element_iterator iter_derivated_mat(derivated_mat);
//
//		for (;;) {
//			u8* ptr_data = iter_derivated_mat.data();
//
//			if (ptr_data == NULL) {
//				break;
//			}
//
//			const u8* ptr_target_data = target_mat.ptr<u8>(iter_derivated_mat.position());
//			target_node->m_mat.get_index(indexs_in_node_data_mat, ptr_target_data);
//
//			if (target_node->m_derivated_mat.valid_index(indexs_in_node_data_mat)) {
//				u8* ptr_node_derivated_data = target_node->m_derivated_mat.ptr<u8>(indexs_in_node_data_mat);
//				mt_mat_helper::data_operation(mt_mat_helper::Math_Op_Code_Add, derivated_mat.depth_channel(), ptr_data, ptr_data, ptr_node_derivated_data);
//			}
//		}
//	}
//
//
//}