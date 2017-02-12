#pragma once

#include "mt_mat.h"
#include "mt_mat_cache.h"

namespace basicmath {

	

	enum mt_Operation_Type {
		mt_Operation_Type_Const_Leaf,
		mt_Operation_Type_Mat_Leaf,
		mt_Operation_Type_Add,
		mt_Operation_Type_Subtract,
		mt_Operation_Type_Pow,
		mt_Operation_Type_Exp,



		mt_Operation_Type_Mul,
		mt_Operation_Type_Sub,
		mt_Operation_Type_Flip,
		mt_Operation_Type_Expand,
		mt_Operation_Type_Transpose,
		mt_Operation_Type_Pooling,
		mt_Operation_Type_Sub_Stride,
		mt_Operation_Type_Mat_Conv,
		mt_Operation_Type_Activate,
		mt_Operation_Type_Loss,
	};

	class mt_ad_mat_tree_node;

	class mt_ad_tree_node {
	public:

		virtual mt_Operation_Type op_type() = 0;

		virtual b8 match(const mt_mat& other) const = 0;
		virtual b8 related(const mt_mat& other) const = 0;

		virtual void derivate_child(mt_ad_mat_tree_node* child_node) = 0;
		virtual mt_ad_mat_tree_node* to_mat_tree_node() {return NULL;}

		vector<mt_ad_tree_node*> m_childs;
	};

	class mt_ad_const_leaf_tree_node : public mt_ad_tree_node {
	public:

		mt_ad_const_leaf_tree_node(const vector<double>& const_value) {
			m_values = const_value;
		}

		mt_Operation_Type op_type() {
			return mt_Operation_Type_Const_Leaf;
		}

		b8 match(const mt_mat& other) const {
			return sys_false;
		}

		 b8 related(const mt_mat& other) const {
			 return sys_false;
		 }

		void derivate_child(mt_ad_mat_tree_node* child_node) {}

		vector<double> m_values;
	};

	class mt_ad_mat_tree_node : public mt_ad_tree_node {
	public:

		b8 match(const mt_mat& other) const {
			return m_mat.same(other);
		}

		b8 related(const mt_mat& other) const {
			return m_mat.memory_shared(other);
		}

		mt_ad_mat_tree_node* to_mat_tree_node() {
			return this;
		}

		void derivate_child(mt_ad_mat_tree_node* child_node) {
			child_node->m_derivate_src_node = m_derivate_src_node;

			derivate_child_on_operation(child_node);
		}

		void init_construct(const mt_mat& res, mt_ad_tree_node* node) {
			m_mat = res;
			m_modified_number = m_mat.get_modified_number();

			m_childs.push_back(node);
		}

		void init_construct(const mt_mat& res, mt_ad_tree_node* left, mt_ad_tree_node* right) {
			m_mat = res;
			m_modified_number = m_mat.get_modified_number();

			m_childs.push_back(left);
			m_childs.push_back(right);
		}

		mt_mat m_mat;
		mt_mat m_derivated_mat;
		mt_ad_tree_node* m_derivate_src_node;	//derivate source / derivate cur
		int m_modified_number;

	protected:

		virtual void derivate_child_on_operation(mt_ad_mat_tree_node* child_node) = 0;
	};

	class mt_ad_leaf_tree_node : public mt_ad_mat_tree_node {
	public:

		mt_ad_leaf_tree_node(const mt_mat& mat) {
			m_mat = mat;
			m_modified_number = m_mat.get_modified_number();

		}

		mt_Operation_Type op_type() {
			return mt_Operation_Type_Mat_Leaf;
		}

	protected:

		void derivate_child_on_operation(mt_ad_mat_tree_node* child_node) {}
	};

	class mt_ad_add_tree_node : public mt_ad_mat_tree_node {
	public:

		mt_ad_add_tree_node(const mt_mat& add_res, mt_ad_tree_node* left, mt_ad_tree_node* right) {
			init_construct(add_res, left, right);
		}

		mt_ad_add_tree_node(const mt_mat& res, vector<mt_ad_tree_node*>& elements) {
			m_mat = res;
			m_childs = elements;
		}

		mt_Operation_Type op_type() {
			return mt_Operation_Type_Add;
		}

	protected:

		void derivate_child_on_operation(mt_ad_mat_tree_node* child_node) {
			child_node->m_derivated_mat = m_derivated_mat.clone();
		}
	};

	class mt_ad_subtract_tree_node : public mt_ad_mat_tree_node {
	public:
		mt_ad_subtract_tree_node(const mt_mat& res, mt_ad_tree_node* left, mt_ad_tree_node* right) {
			init_construct(res, left, right);
		}

		mt_Operation_Type op_type() {
			return mt_Operation_Type_Subtract;
		}

	protected:

		void derivate_child_on_operation(mt_ad_mat_tree_node* child_node) {
			child_node->m_derivated_mat = child_node->m_mat.clone();

			if (child_node == m_childs[0]) {
			} else if (child_node == m_childs[1]) {
				child_node->m_derivated_mat *= -1;
			} else {
				basiclog_assert_message2(sys_false, L"input child_node is not in the m_childs!");
			}
		}

	};

	class mt_ad_mul_tree_node : public mt_ad_mat_tree_node {
	public:

		mt_ad_mul_tree_node(const mt_mat& res, mt_ad_tree_node* left, mt_ad_tree_node* right) {
			init_construct(res, left, right);
		}

		mt_Operation_Type op_type() {
			return mt_Operation_Type_Mul;
		}

		void derivate_child_on_operation(mt_ad_mat_tree_node* child_node) {
			if (child_node == m_childs[0]) {
				child_node->m_derivated_mat = m_derivated_mat.mul(m_childs[1]->to_mat_tree_node()->m_mat.t());
			} else if (child_node == m_childs[1]) {
				child_node->m_derivated_mat = m_childs[0]->to_mat_tree_node()->m_mat.t().mul(m_derivated_mat.t());
			}
		}
	};

	class mt_ad_sub_tree_node : public mt_ad_mat_tree_node {
	public:

		mt_ad_sub_tree_node(const mt_mat& res, mt_ad_tree_node* src, const vector<mt_range>& ranges) {
			init_construct(res, src);

			m_ranges = ranges;
		}

		mt_Operation_Type op_type() {
			return mt_Operation_Type_Sub;
		}

		void derivate_child_on_operation(mt_ad_mat_tree_node* child_node) {
			child_node->m_derivated_mat = mt_mat(child_node->m_mat, mt_mat::Construct_Type_Create_As_Size);
			child_node->m_derivated_mat.set(0);

			child_node->m_derivated_mat.sub(m_ranges).set(m_derivated_mat);
		}

	protected:

		vector<mt_range> m_ranges;
	};

	class mt_ad_sub_stride_tree_node : public mt_ad_mat_tree_node {
	public:

		mt_ad_sub_stride_tree_node(const mt_mat& res, mt_ad_tree_node* src, const vector<i32>& strides) {
			init_construct(res, src);

			m_strides = strides;
		}

		mt_Operation_Type op_type() {
			return mt_Operation_Type_Sub_Stride;
		}

		void derivate_child_on_operation(mt_ad_mat_tree_node* child_node);

	protected:

		vector<i32> m_strides;
	};

	class mt_ad_expand_tree_node : public mt_ad_mat_tree_node {
	public:

		mt_ad_expand_tree_node(const mt_mat& res, mt_ad_tree_node* src, const vector<mt_range>& ranges) {
			init_construct(res, src);

			m_ranges = ranges;
		}

		mt_Operation_Type op_type() {
			return mt_Operation_Type_Sub;
		}

		void derivate_child_on_operation( mt_ad_mat_tree_node* child_node) {
			child_node->m_derivated_mat = mt_mat(child_node->m_mat, mt_mat::Construct_Type_Create_As_Size);

			child_node->m_derivated_mat.set(m_derivated_mat.sub(m_ranges));
		}

	protected:

		vector<mt_range> m_ranges;
	};

	/** Only support valid type conv
	*/
	class mt_ad_conv_tree_node : public mt_ad_mat_tree_node {
	public:

		mt_ad_conv_tree_node(const mt_mat& res, mt_ad_tree_node* src, mt_ad_tree_node* kernel, const i32* strides) {
			m_mat = res;
			m_childs.push_back(src);
			m_childs.push_back(kernel);

			if (strides != NULL) {
				mt_helper::vec_from_array(m_strides, res.dim(), strides);
			}
		}

		mt_ad_conv_tree_node(const mt_mat& res, vector<mt_ad_tree_node*>& srcs, const vector<mt_ad_tree_node*>& kernels, const i32* strides) {
			m_mat = res;
			m_childs = srcs;
			m_childs.insert(m_childs.end(), kernels.begin(), kernels.end());

			if (strides != NULL) {
				mt_helper::vec_from_array(m_strides, res.dim(), strides);
			}
		}

		mt_Operation_Type op_type() {
			return mt_Operation_Type_Sub;
		}

		

	protected:

		void derivate_child_on_operation(mt_ad_mat_tree_node* child_node);

		vector<i32> m_strides;
	};

	class mt_ad_exp_tree_node : public mt_ad_mat_tree_node {
	public:

		mt_ad_exp_tree_node(const mt_mat& res, mt_ad_tree_node* src) {
			init_construct(res, src);
		}

		mt_Operation_Type op_type() {
			return mt_Operation_Type_Exp;
		}

		void derivate_child_on_operation( mt_ad_mat_tree_node* child_node) {
			child_node->m_derivated_mat = mt_mat(child_node->m_mat, mt_mat::Construct_Type_Create_As_Size);

			child_node->m_derivated_mat.set(m_derivated_mat);
		}

	protected:

		vector<mt_range> m_ranges;
	};

	class mt_ad_pow_tree_node : public mt_ad_mat_tree_node {
	public:

		mt_ad_pow_tree_node(const mt_mat& res, mt_ad_tree_node* src, f64 number) {
			init_construct(res, src);

			m_number = number;
		}

		mt_Operation_Type op_type() {
			return mt_Operation_Type_Pow;
		}

		void derivate_child_on_operation( mt_ad_mat_tree_node* child_node) {
			child_node->m_derivated_mat = mt_mat(child_node->m_mat, mt_mat::Construct_Type_Create_As_Size);

			child_node->m_derivated_mat.set(child_node->m_mat.pow(m_number - 1));
		}

	protected:

		f64 m_number;
	};

	class mt_ad_activate_tree_node : public mt_ad_mat_tree_node {
	public:

		mt_ad_activate_tree_node(const mt_mat& res, mt_ad_tree_node* src, mt_Activate_Type type) {
			init_construct(res, src);

			m_activate_type = type;
		}

		mt_Operation_Type op_type() {
			return mt_Operation_Type_Activate;
		}

	protected:

		void derivate_child_on_operation( mt_ad_mat_tree_node* child_node);
		mt_mat softmax_derivate(const mt_mat& softmax_res);
		mt_mat relu_derivate(const mt_mat& relu_res);

		mt_Activate_Type m_activate_type;
	};
}