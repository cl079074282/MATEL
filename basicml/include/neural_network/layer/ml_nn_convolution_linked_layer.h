#pragma once

#include "ml_nn_linked_layer.h"

namespace basicml {

	class ml_nn_convolution_linked_layer : public ml_nn_linked_layer {
	public:

		ml_nn_convolution_linked_layer() 
		: m_flip_flags(NULL)
		, m_flip_flag_size(0) {

		}

		virtual ~ml_nn_convolution_linked_layer() {
			BASICML_SAFE_DELETE_ARRAY(m_flip_flags);
		}

		void init_need_learn_params(int data_type);

		void feedforward(const ml_nn_layer_learning_params& pars);
		void backpropagation(const ml_nn_layer_learning_params& pars);
		ml_nn_layer* clone() const;

		virtual bool has_learned_param() const {return true;}
		virtual void write_learned_param(ml_file_storage& fs) const;
		virtual void read_learned_param(const ml_file_node& node);

		virtual ml_nn_convolution_linked_layer* to_convolution_linked_layer() {return this;}
		virtual const ml_nn_convolution_linked_layer* to_convolution_linked_layer() const {return this;}

	protected:

		virtual void inner_compute_default_setting();

		vector<vector<Mat>> m_kernels;
		vector<double> m_biases;
		vector<vector<Mat>> m_v_kernels;
		vector<double> m_v_biases;

		vector<Mat> m_prev_data_padded_caches;
		vector<Mat> m_bp_singal_padded_caches;
		vector<Mat> m_next_delta_restore_by_stride_caches;

		vector<Mat> m_ff_singal_caches;


		vector<Mat> m_bp_singal_caches;
		vector<Mat> m_bp_singal_conv_result;

		vector<Mat> m_conv_results;
		vector<Mat> m_back_kernel_conv_results;

		bool* m_flip_flags;
		int m_flip_flag_size;
		vector<int> m_next_batch_sizes;
	};

}