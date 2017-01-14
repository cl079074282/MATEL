#pragma once

#include "mt_auto_derivative_tree.h"
#include "mt_mat_cache.h"

namespace basicmath {

	class mt_mat;



	class mt_auto_derivative {
	public:
		mt_auto_derivative() {
			m_enable_math_operation = sys_true;
		}

		~mt_auto_derivative();
		
		
		/** res = derivative src / derivative target
		*/
		mt_mat derivate(const mt_mat& target, const mt_mat& src);


		/**
		@note all variable must be represented as mt_mat, here b is a const value!
		*/
		void add(const mt_mat& res, const mt_mat& a, const vector<double>& b);
		void add(const mt_mat& res, const mt_mat& a, const mt_mat& b);


		void add(const mt_mat& res, const vector<mt_mat>& elements);


		void subtract(const mt_mat& res, const mt_mat& a, const mt_mat& b);
		
		void mul(const mt_mat& res, const mt_mat& a, const mt_mat& b);

		void sub(const mt_mat& res, const mt_mat& a, const vector<mt_range>& ranges);
		void sub_stride(const mt_mat& res, const mt_mat& a, const vector<i32>& strides);

		void expand(const mt_mat& res, const mt_mat& a, const vector<mt_range>& ranges);

		void activate(const mt_mat& res, const mt_mat& src, mt_Activate_Type type, const vector<f64>& activate_params);

		void exp(const mt_mat& res, const mt_mat& src);
		void pow(const mt_mat& res, const mt_mat& src, f64 number);

		/**
		@param enable = sys_true indicates the auto_derivative instance will record the math operation in the derivative tree.
		*/
		void enable_math_operation(b8 enable);
		b8 is_enable_math_operation() const; 


		void reset();

		

	protected:

		void check_res_mat(const mt_mat& mat) const;
		void check_math_op_mat(const mt_mat& mat) const;

		mt_ad_tree_node* find_same(const mt_mat& mat) const;

		b8 has_related(const mt_mat& mat) const;
		mt_ad_tree_node* get_node(const mt_mat& mat);
		mt_ad_tree_node* get_node(const vector<double>& const_value);

		void get_path_from_target_to_src(vector<mt_ad_mat_tree_node*>& path_nodes, mt_ad_tree_node* target_nodes, mt_ad_tree_node* src_node);

		void derivate(mt_mat& derivated_mat, const mt_mat& target_mat, mt_ad_tree_node* src_node);
		void derivate(mt_mat& derivated_mat, const mt_mat& target_mat, mt_ad_mat_tree_node* target_nodes, mt_ad_tree_node* src_node);

		int find_unused_derivated_mat(mt_ad_mat_tree_node* target_node);

		b8 is_need_derivate(const mt_mat& target_mat, mt_ad_tree_node* child_node) const;

		vector<mt_ad_tree_node*> m_nodes;
		b8 m_computed_flag;
		b8 m_enable_math_operation;
	};
}