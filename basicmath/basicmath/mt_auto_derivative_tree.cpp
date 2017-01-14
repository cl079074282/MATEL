#include "stdafx.h"
#include "mt_auto_derivative_tree.h"

void mt_ad_sub_stride_tree_node::derivate_child_on_operation( mt_ad_mat_tree_node* child_node) {
	child_node->m_derivated_mat = mt_mat(child_node->m_mat, mt_mat::Construct_Type_Create_As_Size);
	child_node->m_derivated_mat.set(0);

	mt_mat temp = child_node->m_derivated_mat.sub_stride((i32)m_strides.size(), &m_strides[0]);
	temp.set(m_derivated_mat);
}

void mt_ad_conv_tree_node::derivate_child_on_operation(mt_ad_mat_tree_node* child_node) {
	mt_mat derivate_mat = m_derivated_mat;
	
	i32 half_index = (i32)m_childs.size() / 2;

	if (!m_strides.empty()) {
		basicmath_mat_request_memory(i32, valid_conv_sizes, derivate_mat.dim());
		mt_mat_helper::get_conv_result_size(derivate_mat.dim(), valid_conv_sizes, m_childs[0]->to_mat_tree_node()->m_mat.size(), m_childs[half_index]->to_mat_tree_node()->m_mat.size(), NULL, mt_Conv_Boundary_Type_Valid);
		derivate_mat = m_derivated_mat.unpooling(valid_conv_sizes, mt_mat(), mt_Pooling_Type_First_Value, (i32)m_strides.size(), &m_strides[0], &m_strides[0]);
	}

	for (i32 i = 0; i < (i32)half_index; ++i) {
		if (child_node == m_childs[i]) {
			child_node->m_derivated_mat = derivate_mat.conv(m_childs[i + half_index]->to_mat_tree_node()->m_mat.flip_all_dim(), mt_Conv_Boundary_Type_Full);
			return;
		}
	}

	for (i32 i = half_index; i < (i32)m_childs.size(); ++i) {
		if (child_node == m_childs[i]) {
			child_node->m_derivated_mat = m_childs[i - half_index]->to_mat_tree_node()->m_mat.flip_all_dim().conv(derivate_mat, mt_Conv_Boundary_Type_Valid);
			
			return;
		}
	}
}