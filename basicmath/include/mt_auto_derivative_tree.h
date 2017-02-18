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
		mt_Operation_Type_Log,



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

	static const i32 mt_Auto_Derivative_Default_Max_Cache_Size = 30;

	class mt_ad_mat_tree_node;

	class mt_ad_tree_node {
	public:

		virtual mt_Operation_Type op_type() = 0;

		virtual b8 match(const mt_mat& other) const = 0;
		virtual b8 related(const mt_mat& other) const = 0;

		virtual void derivate_child(mt_ad_mat_tree_node* child_node, const mt_mat& src) = 0;
		virtual mt_ad_mat_tree_node* to_mat_tree_node() {return NULL;}

		vector<mt_ad_tree_node*> m_childs;
	};

	class mt_ad_const_leaf_tree_node : public mt_ad_tree_node {
	public:

		mt_ad_const_leaf_tree_node(const vector<f64>& const_value) {
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

		void derivate_child(mt_ad_mat_tree_node* child_node, const mt_mat& src) {}

		vector<f64> m_values;
	};

	class mt_ad_mat_tree_node : public mt_ad_tree_node {
	public:

		mt_ad_mat_tree_node(i32 max_cache_size = mt_Auto_Derivative_Default_Max_Cache_Size) {
			m_max_cache_size = max_cache_size;
		}

		b8 match(const mt_mat& other) const {
			return m_mat.is_same(other);
		}

		b8 related(const mt_mat& other) const {
			return m_mat.is_memory_shared(other);
		}

		mt_ad_mat_tree_node* to_mat_tree_node() {
			return this;
		}

		void derivate_child(mt_ad_mat_tree_node* child_node, const mt_mat& src) {
			if (child_node->derivative_res(src).is_empty()) {
				child_node->add_derivative_res(derivate_child_on_operation(child_node, derivative_res(src)), src);
			}
		}

		void init_construct(const mt_mat& res, mt_ad_tree_node* node) {
			m_mat = res;

			m_childs.push_back(node);
		}

		void init_construct(const mt_mat& res, mt_ad_tree_node* left, mt_ad_tree_node* right) {
			m_mat = res;

			m_childs.push_back(left);
			m_childs.push_back(right);
		}

		void add_derivative_res(const mt_mat& derivative_res, const mt_mat& src) {
			m_derivated_mats.push_back(derivative_res);
			m_src_mats.push_back(src);

			if ((i32)m_derivated_mats.size() > m_max_cache_size) {
				m_derivated_mats.erase(m_derivated_mats.begin());
				m_src_mats.erase(m_src_mats.begin());
			}
		}

		void set_max_cache_size(i32 size) {
			m_max_cache_size = size;
		}

		mt_mat derivative_res(const mt_mat& src) {
			for (i32 i = 0; i < (i32)m_src_mats.size(); ++i) {
				if (m_src_mats[i].is_same(src)) {
					return m_derivated_mats[i];
				}
			}

			return mt_mat();
		}

		mt_mat m_mat;

		vector<mt_mat> m_derivated_mats;
		vector<mt_mat> m_src_mats;
		

	protected:

		i32 m_max_cache_size;

		virtual mt_mat derivate_child_on_operation(mt_ad_mat_tree_node* child_node, const mt_mat& derivative_res) = 0;
	};

	class mt_ad_leaf_tree_node : public mt_ad_mat_tree_node {
	public:

		mt_ad_leaf_tree_node(const mt_mat& mat) {
			m_mat = mat;
		}

		mt_Operation_Type op_type() {
			return mt_Operation_Type_Mat_Leaf;
		}

	protected:

		mt_mat derivate_child_on_operation(mt_ad_mat_tree_node* child_node, const mt_mat& derivative_res) {return mt_mat();}
	};

	class mt_ad_add_tree_node : public mt_ad_mat_tree_node {
	public:

		mt_ad_add_tree_node(const mt_mat& add_res, mt_ad_tree_node* left, mt_ad_tree_node* right, i32 max_cache_size = mt_Auto_Derivative_Default_Max_Cache_Size) {
			init_construct(add_res, left, right);

			m_max_cache_size = max_cache_size;
		}

		mt_ad_add_tree_node(const mt_mat& res, vector<mt_ad_tree_node*>& elements, i32 max_cache_size = mt_Auto_Derivative_Default_Max_Cache_Size) {
			m_mat = res;
			m_childs = elements;

			m_max_cache_size = max_cache_size;
		}

		mt_Operation_Type op_type() {
			return mt_Operation_Type_Add;
		}

	protected:

		mt_mat derivate_child_on_operation(mt_ad_mat_tree_node* child_node, const mt_mat& derivative_res) {
			return derivative_res.clone();
		}
	};

	class mt_ad_subtract_tree_node : public mt_ad_mat_tree_node {
	public:
		mt_ad_subtract_tree_node(const mt_mat& res, mt_ad_tree_node* left, mt_ad_tree_node* right, i32 max_cache_size = mt_Auto_Derivative_Default_Max_Cache_Size) {
			init_construct(res, left, right);

			m_max_cache_size = max_cache_size;
		}

		mt_Operation_Type op_type() {
			return mt_Operation_Type_Subtract;
		}

	protected:

		mt_mat derivate_child_on_operation(mt_ad_mat_tree_node* child_node, const mt_mat& derivative_res) {
			mt_mat child_derivative = derivative_res.clone();

			if (child_node == m_childs[0]) {
			} else if (child_node == m_childs[1]) {
				child_derivative *= -1;
			} else {
				basiclog_assert_message2(sys_false, L"input child_node is not in the m_childs!");
			}

			return child_derivative;
		}

	};

	class mt_ad_mul_tree_node : public mt_ad_mat_tree_node {
	public:

		mt_ad_mul_tree_node(const mt_mat& res, mt_ad_tree_node* left, mt_ad_tree_node* right, i32 max_cache_size = mt_Auto_Derivative_Default_Max_Cache_Size) {
			init_construct(res, left, right);

			m_max_cache_size = max_cache_size;
		}

		mt_Operation_Type op_type() {
			return mt_Operation_Type_Mul;
		}

	protected:

		mt_mat derivate_child_on_operation(mt_ad_mat_tree_node* child_node, const mt_mat& derivative_res) {
			if (child_node == m_childs[0]) {
				return derivative_res.mul(m_childs[1]->to_mat_tree_node()->m_mat.t());
			} else if (child_node == m_childs[1]) {
				return m_childs[0]->to_mat_tree_node()->m_mat.t().mul(derivative_res.t());
			} else {
				basiclog_assert2(sys_false);
				return mt_mat();
			}
		}
	};

	class mt_ad_sub_tree_node : public mt_ad_mat_tree_node {
	public:

		mt_ad_sub_tree_node(const mt_mat& res, mt_ad_tree_node* src, const vector<mt_range>& ranges, i32 max_cache_size = mt_Auto_Derivative_Default_Max_Cache_Size) {
			init_construct(res, src);

			m_ranges = ranges;
			m_max_cache_size = max_cache_size;
		}

		mt_Operation_Type op_type() {
			return mt_Operation_Type_Sub;
		}

	protected:

		mt_mat derivate_child_on_operation(mt_ad_mat_tree_node* child_node, const mt_mat& derivative_res) {
			mt_mat child_derivative = mt_mat(child_node->m_mat, mt_mat::Construct_Type_Create_As_Size);
			child_derivative.set(0);

			child_derivative.sub(m_ranges).set(derivative_res);

			return child_derivative;
		}

		vector<mt_range> m_ranges;
	};

	class mt_ad_sub_stride_tree_node : public mt_ad_mat_tree_node {
	public:

		mt_ad_sub_stride_tree_node(const mt_mat& res, mt_ad_tree_node* src, const vector<i32>& strides, i32 max_cache_size = mt_Auto_Derivative_Default_Max_Cache_Size) {
			init_construct(res, src);

			m_strides = strides;
			m_max_cache_size = max_cache_size;
		}

		mt_Operation_Type op_type() {
			return mt_Operation_Type_Sub_Stride;
		}

		

	protected:

		mt_mat derivate_child_on_operation(mt_ad_mat_tree_node* child_node, const mt_mat& derivative_res);

		vector<i32> m_strides;
	};

	class mt_ad_expand_tree_node : public mt_ad_mat_tree_node {
	public:

		mt_ad_expand_tree_node(const mt_mat& res, mt_ad_tree_node* src, const vector<mt_range>& ranges, i32 max_cache_size = mt_Auto_Derivative_Default_Max_Cache_Size) {
			init_construct(res, src);

			m_ranges = ranges;
			m_max_cache_size = max_cache_size;
		}

		mt_Operation_Type op_type() {
			return mt_Operation_Type_Sub;
		}

	protected:

		mt_mat derivate_child_on_operation( mt_ad_mat_tree_node* child_node, const mt_mat& derivative_res) {
			mt_mat child_derivative = mt_mat(child_node->m_mat, mt_mat::Construct_Type_Create_As_Size);

			child_derivative.set(derivative_res.sub(m_ranges));

			return child_derivative;
		}

		vector<mt_range> m_ranges;
	};

	/** Only support valid type conv
	*/
	class mt_ad_conv_tree_node : public mt_ad_mat_tree_node {
	public:

		mt_ad_conv_tree_node(const mt_mat& res, mt_ad_tree_node* src, mt_ad_tree_node* kernel, const i32* strides, i32 max_cache_size = mt_Auto_Derivative_Default_Max_Cache_Size) {
			m_mat = res;
			m_childs.push_back(src);
			m_childs.push_back(kernel);

			if (strides != NULL) {
				mt_helper::vec_from_array(m_strides, res.dim(), strides);
			}

			m_max_cache_size = max_cache_size;
		}

		mt_ad_conv_tree_node(const mt_mat& res, vector<mt_ad_tree_node*>& srcs, const vector<mt_ad_tree_node*>& kernels, const i32* strides, i32 max_cache_size = mt_Auto_Derivative_Default_Max_Cache_Size) {
			m_mat = res;
			m_childs = srcs;
			m_childs.insert(m_childs.end(), kernels.begin(), kernels.end());

			if (strides != NULL) {
				mt_helper::vec_from_array(m_strides, res.dim(), strides);
			}

			m_max_cache_size = max_cache_size;
		}

		mt_Operation_Type op_type() {
			return mt_Operation_Type_Sub;
		}

		

	protected:

		mt_mat derivate_child_on_operation(mt_ad_mat_tree_node* child_node, const mt_mat& derivative_res);

		vector<i32> m_strides;
	};

	class mt_ad_exp_tree_node : public mt_ad_mat_tree_node {
	public:

		mt_ad_exp_tree_node(const mt_mat& res, mt_ad_tree_node* src, i32 max_cache_size = mt_Auto_Derivative_Default_Max_Cache_Size) {
			init_construct(res, src);

			m_max_cache_size = max_cache_size;
		}

		mt_Operation_Type op_type() {
			return mt_Operation_Type_Exp;
		}

	protected:

		mt_mat derivate_child_on_operation( mt_ad_mat_tree_node* child_node, const mt_mat& derivative_res) {
			return m_mat * derivative_res;
		}

		vector<mt_range> m_ranges;
	};

	class mt_ad_pow_tree_node : public mt_ad_mat_tree_node {
	public:

		mt_ad_pow_tree_node(const mt_mat& res, mt_ad_tree_node* src, f64 number, i32 max_cache_size = mt_Auto_Derivative_Default_Max_Cache_Size) {
			init_construct(res, src);

			m_number = number;
			m_max_cache_size = max_cache_size;
		}

		mt_Operation_Type op_type() {
			return mt_Operation_Type_Pow;
		}

		mt_mat derivate_child_on_operation( mt_ad_mat_tree_node* child_node, const mt_mat& derivative_res) {
			return derivative_res * child_node->m_mat.pow(m_number - 1);
		}

	protected:

		f64 m_number;
	};

	class mt_ad_log_tree_node : public mt_ad_mat_tree_node {
	public:

		mt_ad_log_tree_node(const mt_mat& res, mt_ad_tree_node* src, f64 base, i32 max_cache_size = mt_Auto_Derivative_Default_Max_Cache_Size) {
			init_construct(res, src);

			m_base = base;
			m_max_cache_size = max_cache_size;
		}

		mt_Operation_Type op_type() {
			return mt_Operation_Type_Log;
		}

	protected:

		mt_mat derivate_child_on_operation( mt_ad_mat_tree_node* child_node, const mt_mat& derivative_res);

		f64 m_base;
	};

	class mt_ad_activate_tree_node : public mt_ad_mat_tree_node {
	public:

		mt_ad_activate_tree_node(const mt_mat& res, mt_ad_tree_node* src, mt_Activate_Type type, const vector<f64>& activated_params, i32 max_cache_size = mt_Auto_Derivative_Default_Max_Cache_Size) {
			init_construct(res, src);

			m_activate_type = type;
			m_activated_params = activated_params;

			m_max_cache_size = max_cache_size;
		}

		mt_Operation_Type op_type() {
			return mt_Operation_Type_Activate;
		}

	protected:

		mt_mat derivate_child_on_operation(mt_ad_mat_tree_node* child_node, const mt_mat& derivative_res);
		mt_mat softmax_derivate(const mt_mat& softmax_res);
		mt_mat relu_derivate(const mt_mat& relu_res);

		mt_Activate_Type m_activate_type;
		vector<f64> m_activated_params;
	};

	class mt_ad_loss_tree_node : public mt_ad_mat_tree_node {
	public:

		mt_ad_loss_tree_node(const mt_mat& res, mt_ad_tree_node* data_1, mt_ad_tree_node* data_2, mt_Loss_Type type, i32 max_cache_size = mt_Auto_Derivative_Default_Max_Cache_Size) {
			init_construct(res, data_1, data_2);

			m_loss_type = type;
			m_max_cache_size = max_cache_size;
		}

		mt_Operation_Type op_type() {
			return mt_Operation_Type_Activate;
		}

	protected:

		mt_mat derivate_child_on_operation( mt_ad_mat_tree_node* child_node, const mt_mat& derivative_res);
		mt_mat softmax_derivate(const mt_mat& softmax_res);
		mt_mat relu_derivate(const mt_mat& relu_res);

		mt_Loss_Type m_loss_type;
	};
}