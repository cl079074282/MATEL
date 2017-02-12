#include "stdafx.h"

#include "ml_nn_layer.h"
#include "ml_nn_data_layer.h"
#include "ml_nn_input_data_layer.h"
#include "ml_nn_inner_product_linked_layer.h"
#include "ml_nn_output_data_layer.h"

ml_nn_layer* ml_nn_layer::read(const sys_json_reader& reader) {
	ml_nn_layer* layer = NULL;
	
	layer = ml_nn_input_data_layer::read(reader);

	if (layer != NULL) {
		return layer;
	}

	layer = ml_nn_inner_product_linked_layer::read(reader);

	if (layer != NULL) {
		return layer;
	}

	layer = ml_nn_data_layer::read(reader);

	if (layer != NULL) {
		return layer;
	}

	layer = ml_nn_output_data_layer::read(reader);

	if (layer != NULL) {
		return layer;
	}

	return layer;
}


//
//#define GRADIENT_CHECK	0
//
//class private_nn_layer {
//public:
//
//	template<class T>
//	static void max_pooling_backpropagation(Mat& prev_delta, const Mat& next_delta, const Mat& max_mask) {
//		const uchar* ptr_latter_delta_mat_dim0 = next_delta.data;
//		const uchar* ptr_latter_mat_index_mat_dim0 = max_mask.data;
//
//		for (int i = 0; i < next_delta.size.p[0]; ++i) {
//			const uchar* ptr_latter_delta_mat_dim1 = ptr_latter_delta_mat_dim0;
//			const uchar* ptr_latter_mat_index_mat_dim1 = ptr_latter_mat_index_mat_dim0;
//
//			for (int j = 0; j < next_delta.size.p[1]; ++j) {
//				T* ptr_latter_delta_mat = (T*)ptr_latter_delta_mat_dim1;
//				int* ptr_latter_mat_index_mat = (int*)ptr_latter_mat_index_mat_dim1;
//
//				for (int k = 0; k < next_delta.size.p[2]; ++k) {
//					int row = (*ptr_latter_mat_index_mat) / prev_delta.size.p[2];
//					int col = *ptr_latter_mat_index_mat - row * prev_delta.size.p[2];
//
//					prev_delta.at<T>(i, row, col) = *ptr_latter_delta_mat++;
//					++ptr_latter_mat_index_mat;
//				}
//
//				ptr_latter_delta_mat_dim1 += next_delta.step.p[1];
//				ptr_latter_mat_index_mat_dim1 += max_mask.step.p[1];
//			}
//
//			ptr_latter_delta_mat_dim0 += next_delta.step.p[0];
//			ptr_latter_mat_index_mat_dim0 += max_mask.step.p[0];
//		}
//	}
//
//	static void statistic_weight_update_info(Mat& weight) {
//		double variance = ml_mat_op::ml_variance(weight)[0];
//		double max_value = ml_mat_op::ml_max(weight)[0];
//		double min_value = ml_mat_op::ml_min(weight)[0];
//
//		Mat abs_weight = abs(weight);
//		double abs_variance = ml_mat_op::ml_variance(abs_weight)[0];
//		double abs_max_value = ml_mat_op::ml_max(abs_weight)[0];
//		double abs_min_value = ml_mat_op::ml_min(abs_weight)[0];
//		
//		wstring info = ml_string_define::combine(L"variance: %.15f, max: %.15f, min: %.15f, abs_variance: %.15f, abs_max: %.15f, abs_min: %.15f", variance, max_value, min_value, abs_variance, abs_max_value, abs_min_value);
//		BASICLOGTRACE_MESSAGE(info);
//	}
//
//	template<class T>
//	static void max_out_2d(Mat& out, Mat& max_mask_index, const Mat& input, int k) {
//		uchar* ptr_out_mat_dim0 = out.data;
//		uchar* ptr_out_max_index_dim0 = max_mask_index.data;
//		uchar* ptr_input_dim0 = input.data;
//
//		for (int iter_sample = 0; iter_sample < out.size.p[0]; ++iter_sample) {
//			uchar* ptr_out_mat_dim1 = ptr_out_mat_dim0;
//			uchar* ptr_out_max_index_dim1 = ptr_out_max_index_dim0;
//			uchar* ptr_input_dim1 = ptr_input_dim0;
//
//			for (int out_row = 0; out_row < out.size.p[1]; ++out_row) {
//				T* ptr_out_mat = (T*)ptr_out_mat_dim1;
//				int* ptr_out_max_index = (int*)ptr_out_max_index_dim1;
//				T* ptr_input = (T*)ptr_input_dim1;
//
//				for (int out_col = 0; out_col < out.size.p[2]; ++out_col) {
//					if (*ptr_input > *ptr_out_mat) {
//						*ptr_out_mat = *ptr_input;
//						*ptr_out_max_index = k;
//					}
//
//					++ptr_input;
//					++ptr_out_mat;
//					++ptr_out_max_index;
//				}
//
//				ptr_out_mat_dim1 += out.step.p[1];
//				ptr_out_max_index_dim1 += max_mask_index.step.p[1];
//				ptr_input_dim1 += input.step.p[1];
//			}
//
//			ptr_out_mat_dim0 += out.step.p[0];
//			ptr_out_max_index_dim0 += max_mask_index.step.p[0];
//			ptr_input_dim0 += input.step.p[0];
//		}
//	}
//
//	template<class T>
//	static void max_out_1d(Mat& out, Mat& out_max_index, const Mat& input, int max_k) {
//		uchar* ptr_out_dim0 = out.data;
//		uchar* ptr_out_max_index_dim0 = out_max_index.data;
//		uchar* ptr_input_dim0 = input.data;
//		
//		for (int row = 0; row < out.rows; ++row) {
//			T* ptr_output = (T*)ptr_out_dim0;
//			int* ptr_max_index = (int*)ptr_out_max_index_dim0;
//			T* ptr_input = (T*)ptr_input_dim0;
//
//			int start_index = 0;
//			int stop_index = max_k;
//
//			for (int col = 0; col < out.cols; ++col) {
//				if (stop_index > input.cols) {
//					stop_index = input.cols;
//				}	
//
//				*ptr_output = *ptr_input++;
//				*ptr_max_index = start_index;
//
//				for (int k = start_index + 1; k < stop_index; ++k) {
//					if (*ptr_input > *ptr_output) {
//						*ptr_output = *ptr_input;
//						*ptr_max_index = k;
//					}
//
//					++ptr_input;
//				}
//
//				++ptr_output;
//				++ptr_max_index;
//
//				start_index += max_k;
//				stop_index += max_k;
//			}
//
//			ptr_out_dim0 += out.step.p[0];
//			ptr_out_max_index_dim0 += out_max_index.step.p[0];
//			ptr_input_dim0 += input.step.p[0]; 
//		}
//	}
//
//	template<class T>
//	static void restore_max_out_1d(Mat& prev_delta, const Mat& next_delta, const Mat& max_mask) {
//		uchar* ptr_prev_delta_dim0 = prev_delta.data;
//		uchar* ptr_next_delta_dim0 = next_delta.data;
//		uchar* ptr_max_mask_dim0 = max_mask.data;
//
//		for (int row = 0; row < next_delta.rows; ++row) {
//			T* ptr_prev_delta_dim1 = (T*)ptr_prev_delta_dim0;
//			int* ptr_max_mask_dim1 = (int*)ptr_max_mask_dim0;
//			T* ptr_next_delta_dim1 = (T*)ptr_next_delta_dim0;
//
//			for (int col = 0; col < next_delta.cols; ++col) {
//				ptr_prev_delta_dim1[*ptr_max_mask_dim1++] = *ptr_next_delta_dim1++;
//			}
//
//			ptr_max_mask_dim0 += max_mask.step.p[0];
//			ptr_next_delta_dim0 += next_delta.step.p[0];
//			ptr_prev_delta_dim0 += prev_delta.step.p[0];
//		}
//	}
//};
//
//ml_nn_layer* ml_nn_layer::new_layer(ml_nn_layer_config* config) {
//	
//}
//
//void ml_nn_linked_layer::set_input(ml_nn_data_layer* input){
//	if (m_input == input) {
//		return;
//	}
//
//	if (NULL != m_input) {
//		vector<ml_nn_linked_layer*>::iterator iter = find(m_input->m_next_linked_layers.begin(), m_input->m_next_linked_layers.end(), this);
//
//		if (iter != m_input->m_next_linked_layers.end()) {
//			m_input->m_next_linked_layers.erase(iter);
//		}
//	}
//
//	m_input = input;
//	m_input->m_next_linked_layers.push_back(this);
//}
//
//void ml_nn_linked_layer::set_output(ml_nn_data_layer* output) {
//	if (m_output == output) {
//		return;
//	}
//
//	if (NULL != m_output) {
//		vector<ml_nn_linked_layer*>::iterator iter = find(m_output->m_prev_linked_layers.begin(), m_output->m_prev_linked_layers.end(), this);
//
//		if (iter != m_output->m_prev_linked_layers.end()) {
//			m_output->m_prev_linked_layers.erase(iter);
//		}
//	}
//
//	m_output = output;
//	m_output->m_prev_linked_layers.push_back(this);
//}
//
//void ml_nn_data_layer::set_check_auto_params() {
//	for (int i = 0; i < (int)m_next_linked_layers.size(); ++i) {
//		m_next_linked_layers[i]->set_check_auto_params();
//	}
//}
//
//void ml_nn_inner_product_linked_layer::init_need_learn_params(int data_type) {	
//	__super::init_need_learn_params(data_type);
//
//	if (m_weight_update_setting.m_learned_param_init_type == ml_Learned_Param_Init_Type_Gaussian) {
//		ml_random::gaussian_iid(m_weight, m_input->to_1d_data_layer()->get_unit_number(), m_output->to_1d_data_layer()->get_unit_number(), data_type, 0, m_weight_update_setting.m_learned_param_init_params[0]);		
//	} else {
//		BASICLOGASSERT(false);
//	}
//
//	if (m_bias_update_setting.m_learned_param_init_type == ml_Learned_Param_Init_Type_Gaussian) {
//		ml_random::gaussian_iid(m_bias, 1, m_output->to_1d_data_layer()->get_unit_number(), data_type, 0, m_bias_update_setting.m_learned_param_init_params[0]);
//	}
//
//	m_v_weight.create(m_input->to_1d_data_layer()->get_unit_number(), m_output->to_1d_data_layer()->get_unit_number(), data_type);
//	ml_mat_op::set(m_v_weight, Scalar(0));
//	m_v_bias.create(1, m_output->to_1d_data_layer()->get_unit_number(), data_type);
//	ml_mat_op::set(m_v_bias, Scalar(0));
//}
//
//void ml_nn_inner_product_linked_layer::feedforward(const ml_nn_layer_learning_params& pars) {
//	Mat& prev_input_data = m_input->to_1d_data_layer()->get_data();
//	
//	if (pars.m_inference_stage) {
//		if (m_drop_type == Drop_Null) {
//			ml_mat_op::mkl_mul(m_ff_singal_cache, prev_input_data, m_weight);
//			ml_mat_op::self_add_row(m_ff_singal_cache, m_bias);	
//			m_output->to_1d_data_layer()->feedforward_singal(m_ff_singal_cache, pars);
//		} else {
//			if (m_inference_type == Inference_By_Average) {
//				ml_mat_op::mkl_mul(m_ff_singal_cache, prev_input_data, m_weight);
//				m_ff_singal_cache *= m_ratio;
//
//				ml_mat_op::self_add_row(m_ff_singal_cache, m_bias);	
//				m_output->to_1d_data_layer()->feedforward_singal(m_ff_singal_cache, pars);
//
//			} else {
//				BASICLOGASSERT(m_drwan_number > 0);
//
//				Mat drop_means;
//				ml_mat_op::mkl_mul(drop_means, prev_input_data, m_weight);
//				drop_means *= m_ratio;
//
//				Mat prev_input_square;
//				Mat weight_square;
//				cv::pow(prev_input_data, 2.0, prev_input_square);
//				cv::pow(m_weight, 2.0, weight_square);
//
//				Mat drop_standard_deviation;
//
//				ml_mat_op::mkl_mul(drop_standard_deviation, prev_input_square, weight_square);
//
//				drop_standard_deviation *= m_ratio * (1 - m_ratio);
//				cv::pow(drop_standard_deviation, 0.5, drop_standard_deviation);
//
//				vector<Mat> drop_samples;
//				drop_samples.resize(m_drwan_number);						
//
//				for (int iter_sample = 0; iter_sample < m_drwan_number; ++iter_sample) {
//					drop_samples[iter_sample].create(prev_input_data.rows, m_weight.cols, prev_input_data.type());
//
//					uchar* ptr_sample_dim0 = drop_samples[iter_sample].data;
//					uchar* ptr_mean_dim0 = drop_means.data;;
//					uchar* ptr_std_deviation_dim0 = drop_standard_deviation.data;
//
//					for (int iter_batch = 0; iter_batch < prev_input_data.rows; ++iter_batch) {
//						if (CV_32FC1 == prev_input_data.type()) {
//							float* ptr_sample = (float*)ptr_sample_dim0;
//							float* ptr_mean = (float*)ptr_mean_dim0;
//							float* ptr_std_deviation = (float*)ptr_std_deviation_dim0;
//
//							for (int col = 0; col < m_weight.cols; ++col) {
//								*ptr_sample++ = (float)ml_random::gaussian(*ptr_mean++, *ptr_std_deviation++);
//							}
//						} else if (CV_64FC1 == prev_input_data.type()){
//							double* ptr_sample = (double*)ptr_sample_dim0;
//							double* ptr_mean = (double*)ptr_mean_dim0;
//							double* ptr_std_deviation = (double*)ptr_std_deviation_dim0;
//
//							for (int col = 0; col < m_weight.cols; ++col) {
//								*ptr_sample++ = ml_random::gaussian(*ptr_mean++, *ptr_std_deviation++);
//							}
//						}					
//
//						ptr_sample_dim0 += drop_samples[iter_sample].step.p[0];
//						ptr_mean_dim0 += drop_means.step.p[0];
//						ptr_std_deviation_dim0 += drop_standard_deviation.step.p[0];
//					}
//
//					ml_mat_op::self_add_row(drop_samples[iter_sample], m_bias);
//				}
//
//				m_output->to_1d_data_layer()->feedforward_singal(drop_samples, pars);
//			}
//		}
//	} else {
//		if (m_drop_type == Drop_Out) {
//			ml_random::bernoulli_iid(m_drop_out_mask, 
//				prev_input_data.rows, 
//				prev_input_data.cols, 
//				prev_input_data.type(), 
//				1,
//				m_ratio);
//
//			multiply(prev_input_data, m_drop_out_mask, m_drop_out_cache);
//			ml_mat_op::mkl_mul(m_ff_singal_cache, m_drop_out_cache, m_weight);
//		} else if (m_drop_type == Drop_Connection) {
//			ml_random::bernoulli_iid(m_drop_connection_mask, 
//				m_weight.rows, 
//				m_weight.cols, 
//				m_weight.type(), 
//				1,
//				m_ratio);
//
//			multiply(m_weight, m_drop_connection_mask, m_drop_connection_cache);
//			ml_mat_op::mkl_mul(m_ff_singal_cache, prev_input_data, m_drop_connection_cache);
//		} else {
//			ml_mat_op::mkl_mul(m_ff_singal_cache, prev_input_data, m_weight);
//		}
//
//		ml_mat_op::self_add_row(m_ff_singal_cache, m_bias);	
//		m_output->to_1d_data_layer()->feedforward_singal(m_ff_singal_cache, pars);
//	}
//}
//
//void ml_nn_inner_product_linked_layer::backpropagation(const ml_nn_layer_learning_params& pars) {
//	Mat& next_delta = m_output->to_1d_data_layer()->get_delta();
//	Mat& input_data = m_input->to_1d_data_layer()->get_data();
//	
//	if (m_drop_type == Drop_Out) {
//		ml_mat_op::mkl_mul(m_bp_singal_cache, next_delta, m_weight, false, true);
//		multiply(m_bp_singal_cache, m_drop_out_mask, m_bp_singal_cache);
//
//		ml_mat_op::mkl_mul(m_weight_update_cache, m_drop_out_cache, next_delta, true, false);
//
//	} else if (m_drop_type == Drop_Connection){
//		ml_mat_op::mkl_mul(m_bp_singal_cache, next_delta, m_drop_connection_cache, false, true);
//
//		ml_mat_op::mkl_mul(m_weight_update_cache, input_data, next_delta, true, false);
//		multiply(m_weight_update_cache, m_drop_connection_mask, m_weight_update_cache);
//	} else {
//		ml_mat_op::mkl_mul(m_bp_singal_cache, next_delta, m_weight, false, true);
//		ml_mat_op::mkl_mul(m_weight_update_cache, input_data, next_delta, true, false);
//	}
//
//	ml_mat_op::ml_mean(m_bias_update_cache, next_delta, 0);
//
//	m_weight_update_cache /= next_delta.rows;
//
//	ml_define::add_penalty(m_weight_update_cache, m_weight, m_weight_update_setting.m_penalty_type, m_weight_update_setting.m_penalty_alpha);
//	ml_define::add_penalty(m_bias_update_cache, m_bias, m_bias_update_setting.m_penalty_type, m_bias_update_setting.m_penalty_alpha);
//	
//	if (GRADIENT_CHECK) {
//		BASICLOGTRACE_MESSAGE(ml_string()<<m_unique_name<<L" weight update");
//		private_nn_layer::statistic_weight_update_info(m_weight_update_cache);
//
//		BASICLOGTRACE_MESSAGE(L"current weight");
//		private_nn_layer::statistic_weight_update_info(m_weight);
//	}
//	
//	ml_define::update_learned_param(m_weight, m_v_weight, m_weight_update_cache, m_weight_update_setting.m_momentum, m_weight_update_setting.m_learning_rate);
//	ml_define::update_learned_param(m_bias, m_v_bias, m_bias_update_cache, m_bias_update_setting.m_momentum, m_bias_update_setting.m_learning_rate);
//
//	if (pars.m_iteration_index % m_weight_update_setting.m_learning_rate_scale_intereation == 0) {
//		m_weight_update_setting.m_learning_rate *= m_weight_update_setting.m_learning_scale_ratio;
//	}
//
//	if (pars.m_iteration_index % m_bias_update_setting.m_learning_rate_scale_intereation == 0) {
//		m_bias_update_setting.m_learning_rate *= m_bias_update_setting.m_learning_scale_ratio;
//	}
//
//	m_input->to_1d_data_layer()->backprapogation_singal(m_bp_singal_cache, pars);
//}
//
//ml_nn_layer* ml_nn_inner_product_linked_layer::clone() const {
//	ml_nn_inner_product_linked_layer* layer = new ml_nn_inner_product_linked_layer(m_unique_name);
//	layer->m_input = m_input;
//	layer->m_output = m_output;
//	layer->m_weight_update_setting = m_weight_update_setting;
//	layer->m_bias_update_setting = m_bias_update_setting;
//
//	layer->m_weight = m_weight.clone();
//	layer->m_v_weight = m_v_weight.clone();
//	layer->m_bias = m_bias.clone();
//	layer->m_v_bias = m_v_bias.clone();
//
//	layer->m_drop_type = m_drop_type;
//
//	layer->m_inference_type = m_inference_type;
//	layer->m_drwan_number = m_drwan_number;
//	layer->m_weight = m_weight.clone();
//	layer->m_v_weight = m_v_weight.clone();
//	layer->m_bias = m_bias.clone();
//	layer->m_v_bias = m_v_bias.clone();
//
//	return layer;
//}
//
//void ml_nn_convolution_linked_layer::init_need_learn_params(int data_type) {
//	__super::init_need_learn_params(data_type);
//
//	int next_layer_image_number = (int)m_output->to_2d_data_layer()->get_image_number();
//	int prev_layer_image_number = (int)m_input->to_2d_data_layer()->get_image_number();
//
//	vector<Mat> tempKernels;
//	tempKernels.resize(next_layer_image_number);
//	m_kernels.resize(prev_layer_image_number, tempKernels);
//	m_biases.resize(next_layer_image_number, 0);
//
//	m_v_kernels = m_kernels;
//	m_v_biases = m_biases;
//
//	for (int iterOutput = 0; iterOutput < next_layer_image_number; ++iterOutput) {
//		if (m_bias_update_setting.m_learned_param_init_type == ml_Learned_Param_Init_Type_Gaussian) {
//			m_biases[iterOutput] = ml_random::gaussian(0, m_bias_update_setting.m_learned_param_init_params[0]);
//		} else {
//
//		}
//		
//		for (int iterInput = 0; iterInput < prev_layer_image_number; ++iterInput) {
//			if (m_weight_update_setting.m_learned_param_init_type == ml_Learned_Param_Init_Type_Gaussian) {
//				ml_random::gaussian_iid(m_kernels[iterInput][iterOutput], m_kernel_size.height, m_kernel_size.width, data_type, 0, m_weight_update_setting.m_learned_param_init_params[0]);
//			} else {
//
//			}
//			
//			m_v_kernels[iterInput][iterOutput].create(m_kernel_size.height, m_kernel_size.width, data_type);
//			ml_mat_op::set(m_v_kernels[iterInput][iterOutput], Scalar(0));
//		}
//	}
//}
//
//void ml_nn_convolution_linked_layer::feedforward(const ml_nn_layer_learning_params& pars) {
//	ml_timer timer(m_unique_name + L" feedforward", false);
//	timer.begin();
//	
//	int next_layer_image_number = (int)m_kernels.front().size();
//	int prev_layer_image_number = (int)m_kernels.size();
//	const Size& next_image_size = m_output->to_2d_data_layer()->get_image_size();
//
//	if (m_tl_pad_size != Size(0, 0) || m_br_pad_size != Size(0, 0)) {
//		m_prev_data_padded_caches.resize(m_input->to_2d_data_layer()->get_data().size());
//
//		for (int iter_input = 0; iter_input < (int)m_prev_data_padded_caches.size(); ++iter_input) {
//			ml_mat_op::pand_mat(m_prev_data_padded_caches[iter_input], m_input->to_2d_data_layer()->get_data()[iter_input], m_tl_pad_size, m_br_pad_size, Scalar(0));
//		}
//	} else {
//		m_prev_data_padded_caches = m_input->to_2d_data_layer()->get_data();
//	}
//
//	int sample_count = m_prev_data_padded_caches.front().size.p[0];
//
//	m_ff_singal_caches.resize(next_layer_image_number);
//	int dimSizes[] = {sample_count, next_image_size.height, next_image_size.width};
//	int strides[] = {1, m_stride.height, m_stride.width};
//
//	if (m_kernel_size.area() > 256 && next_image_size.area() > 256) {
//		m_conv_results.resize(1);
//
//		for (int iterOut = 0; iterOut < next_layer_image_number; ++iterOut) {		
//			m_ff_singal_caches[iterOut].create(3, dimSizes, m_prev_data_padded_caches.front().type());
//			ml_mat_op::set(m_ff_singal_caches[iterOut], Scalar(0));
//
//			for (int iterInput = 0; iterInput < prev_layer_image_number; ++iterInput) {
//				ml_mat_op::mkl_conv(m_conv_results[0], m_prev_data_padded_caches[iterInput], m_kernels[iterInput][iterOut], ml_mat_op::Conv_Boundary_Type_Valid, m_conv_type, strides, ml_mat_op::Conv_Compute_Type_FFT);
//				m_ff_singal_caches[iterOut] += m_conv_results[0];
//			}
//
//			m_ff_singal_caches[iterOut] += m_biases[iterOut];
//		}
//	} else {
//
//		m_conv_results.resize(next_layer_image_number);
//
//		#pragma omp parallel for num_threads(omp_get_max_threads())
//		for (int iterOut = 0; iterOut < next_layer_image_number; ++iterOut) {		
//			m_ff_singal_caches[iterOut].create(3, dimSizes, m_prev_data_padded_caches.front().type());
//			ml_mat_op::set(m_ff_singal_caches[iterOut], Scalar(0));
//
//			for (int iterInput = 0; iterInput < prev_layer_image_number; ++iterInput) {
//				ml_mat_op::mkl_conv(m_conv_results[iterOut], m_prev_data_padded_caches[iterInput], m_kernels[iterInput][iterOut], ml_mat_op::Conv_Boundary_Type_Valid, m_conv_type, strides, ml_mat_op::Conv_Compute_Type_Direct);
//				m_ff_singal_caches[iterOut] += m_conv_results[iterOut];
//			}
//
//			m_ff_singal_caches[iterOut] += m_biases[iterOut];
//		}
//	}
//
//	timer.end();
//	m_output->to_2d_data_layer()->feedforward_singal(m_ff_singal_caches, pars);
//}
//
//void ml_nn_convolution_linked_layer::backpropagation(const ml_nn_layer_learning_params& pars) {
//	ml_timer timer(m_unique_name + L" backpropagation", false);
//	timer.begin();
//
//	int next_layer_image_number = (int)m_kernels.front().size();
//	int prev_layer_image_number = (int)m_kernels.size();
//	Size prev_image_size = m_input->to_2d_data_layer()->get_image_size();
//	
//	int sample_count = m_prev_data_padded_caches.front().size.p[0];
//	Size temp_size = prev_image_size + m_tl_pad_size + m_br_pad_size;
//	int prev_convolved_delta_image_size[] = {sample_count, temp_size.height, temp_size.width};
//		
//	if (m_stride.width != 1 || m_stride.height != 1) {
//		m_next_delta_restore_by_stride_caches.resize(next_layer_image_number);
//		int strides[] = {m_stride.height, m_stride.width};
//
//		for (int iter_out = 0; iter_out < next_layer_image_number; ++iter_out) {
//			ml_mat_op::restore_mat_by_stride(m_next_delta_restore_by_stride_caches[iter_out], prev_convolved_delta_image_size, m_output->to_2d_data_layer()->get_delta()[iter_out], strides);
//		}
//	} else {
//		m_next_delta_restore_by_stride_caches = m_output->to_2d_data_layer()->get_delta();
//	}
//	
//	bool flip_2d_flags[2] = {true, true};	
//	bool flip_3d_flags[3] = {true, true, true};	
//
//	m_bp_singal_padded_caches.resize(prev_layer_image_number);
//
//	if (m_kernel_size.area() > 256 && temp_size.area() > 256) {
//		m_bp_singal_conv_result.resize(1);
//
//		for (int iter_input = 0; iter_input < prev_layer_image_number; ++iter_input) {
//			m_bp_singal_padded_caches[iter_input].create(3, prev_convolved_delta_image_size, m_next_delta_restore_by_stride_caches.front().type());
//			ml_mat_op::set(m_bp_singal_padded_caches[iter_input], Scalar(0));
//
//			for (int iter_output = 0; iter_output < next_layer_image_number; ++iter_output) {		
//				ml_mat_op::Conv_Type conv_type;
//
//				if (m_conv_type == ml_mat_op::Conv_Type_Conv) {
//					conv_type = ml_mat_op::Conv_Type_Corr;
//				} else {
//					conv_type = ml_mat_op::Conv_Type_Conv;
//				}
//
//				ml_mat_op::mkl_conv_ext(m_bp_singal_conv_result[0], m_next_delta_restore_by_stride_caches[iter_output], m_kernels[iter_input][iter_output], ml_mat_op::Conv_Boundary_Type_Full, conv_type, NULL, NULL, NULL, NULL, ml_mat_op::Conv_Compute_Type_FFT);
//
//				m_bp_singal_padded_caches[iter_input] += m_bp_singal_conv_result[0];						
//			}
//		}
//
//	} else {
//		m_bp_singal_conv_result.resize(prev_layer_image_number);
//
//		#pragma omp parallel for num_threads(omp_get_max_threads())
//		for (int iter_input = 0; iter_input < prev_layer_image_number; ++iter_input) {
//			m_bp_singal_padded_caches[iter_input].create(3, prev_convolved_delta_image_size, m_next_delta_restore_by_stride_caches.front().type());
//			ml_mat_op::set(m_bp_singal_padded_caches[iter_input], Scalar(0));
//
//			for (int iter_output = 0; iter_output < next_layer_image_number; ++iter_output) {
//				ml_mat_op::Conv_Type conv_type;
//
//				if (m_conv_type == ml_mat_op::Conv_Type_Conv) {
//					conv_type = ml_mat_op::Conv_Type_Corr;
//				} else {
//					conv_type = ml_mat_op::Conv_Type_Conv;
//				}
//
//				ml_mat_op::mkl_conv(m_bp_singal_conv_result[iter_input], m_next_delta_restore_by_stride_caches[iter_output], m_kernels[iter_input][iter_output], ml_mat_op::Conv_Boundary_Type_Full, conv_type, NULL, ml_mat_op::Conv_Compute_Type_Direct);
//				m_bp_singal_padded_caches[iter_input] += m_bp_singal_conv_result[iter_input];						
//			}
//		}
//	}
//
//	m_back_kernel_conv_results.resize(next_layer_image_number);
//	
//	#pragma omp parallel for num_threads(omp_get_max_threads())
//	for (int iter_output = 0; iter_output < next_layer_image_number; ++iter_output) {
//		for (int iter_input = 0; iter_input < prev_layer_image_number; ++iter_input) {
//			bool* dst_filp_flags;
//			if (m_conv_type == ml_mat_op::Conv_Type_Conv) {
//				dst_filp_flags = flip_3d_flags;
//			} else {
//				dst_filp_flags = NULL;
//			}
//
//			ml_mat_op::mkl_conv_ext(m_back_kernel_conv_results[iter_output], m_prev_data_padded_caches[iter_input], m_next_delta_restore_by_stride_caches[iter_output], ml_mat_op::Conv_Boundary_Type_Valid, ml_mat_op::Conv_Type_Corr, NULL, dst_filp_flags, NULL, NULL, ml_mat_op::Conv_Compute_Type_Direct);
//			
//			Mat d_kernel = ml_mat_op::decrease_dim(m_back_kernel_conv_results[iter_output]);
//			d_kernel /= sample_count;
//
//			ml_define::add_penalty(d_kernel, m_kernels[iter_input][iter_output], m_weight_update_setting.m_penalty_type, m_weight_update_setting.m_penalty_alpha);
//
//			ml_define::update_learned_param(m_kernels[iter_input][iter_output], m_v_kernels[iter_input][iter_output], d_kernel, m_weight_update_setting.m_momentum, m_weight_update_setting.m_learning_rate);
//
//			if (GRADIENT_CHECK) {
//				BASICLOGTRACE_MESSAGE(ml_string()<<m_unique_name<<L"kernel_"<<iter_input<<L"_"<<iter_output<<L" kernel update");
//				private_nn_layer::statistic_weight_update_info(m_v_kernels[iter_input][iter_output]);
//
//				BASICLOGTRACE_MESSAGE(L"current kernel");
//				private_nn_layer::statistic_weight_update_info(m_kernels[iter_input][iter_output]);
//			}
//		}
//
//		double d_kernel_bias = cv::sum(m_next_delta_restore_by_stride_caches[iter_output]).val[0] / sample_count;
//		ml_define::add_penalty(d_kernel_bias, m_biases[iter_output], m_bias_update_setting.m_penalty_type, m_bias_update_setting.m_penalty_alpha);
//		ml_define::update_learned_param(m_biases[iter_output], m_v_biases[iter_output], d_kernel_bias, m_weight_update_setting.m_momentum, m_weight_update_setting.m_learning_rate);
//	}
//
//	if (pars.m_iteration_index % m_weight_update_setting.m_learning_rate_scale_intereation == 0) {
//		m_weight_update_setting.m_learning_rate *= m_weight_update_setting.m_learning_scale_ratio;
//	}
//
//	if (pars.m_iteration_index % m_bias_update_setting.m_learning_rate_scale_intereation == 0) {
//		m_bias_update_setting.m_learning_rate *= m_bias_update_setting.m_learning_scale_ratio;
//	}
//
//	if (m_tl_pad_size != Size(0, 0) || m_br_pad_size != Size(0, 0)) {
//		m_bp_singal_caches.resize(m_bp_singal_padded_caches.size());
//
//		for (int iter_input = 0; iter_input < (int)m_bp_singal_caches.size(); ++iter_input) {
//			ml_mat_op::unpand_mat(m_bp_singal_caches[iter_input], m_bp_singal_padded_caches[iter_input], m_tl_pad_size, m_br_pad_size);
//		}
//	} else {
//		m_bp_singal_caches = m_bp_singal_padded_caches;
//	}
//
//	timer.end();
//	m_input->to_2d_data_layer()->backprapogation_singal(m_bp_singal_caches, pars);	
//}
//
//ml_nn_layer* ml_nn_convolution_linked_layer::clone() const {
//	ml_nn_convolution_linked_layer* layer = new ml_nn_convolution_linked_layer(m_unique_name);
//	layer->m_input = m_input;
//	layer->m_output = m_output;
//	layer->m_weight_update_setting = m_weight_update_setting;
//	layer->m_bias_update_setting = m_bias_update_setting;
//
//	layer->m_kernels = m_kernels;
//	layer->m_v_kernels = m_v_kernels;
//	layer->m_biases = m_biases;
//	layer->m_v_biases = m_v_biases;
//
//	layer->m_kernel_size = m_kernel_size;
//	layer->m_stride = m_stride;
//	layer->m_tl_pad_size = m_tl_pad_size;
//	layer->m_br_pad_size = m_br_pad_size;
//	
//	for (int i = 0; i < (int)m_kernels.size(); ++i) {
//		for (int j = 0; j < (int)m_kernels[i].size(); ++j) {
//			layer->m_kernels[i][j] = m_kernels[i][j].clone();
//			layer->m_v_kernels[i][j] = m_v_kernels[i][j].clone();
//		}
//	}
//
//	return layer;
//}
//
//void ml_nn_convolution_linked_layer::inner_compute_default_setting() {
//	m_output->to_2d_data_layer()->set_image_size(ml_mat_op::after_conv(m_input->to_2d_data_layer()->get_image_size() + m_tl_pad_size + m_br_pad_size, m_kernel_size, ml_mat_op::Conv_Boundary_Type_Valid, m_stride));
//}
//
//void ml_nn_pooling_linked_layer::feedforward(const ml_nn_layer_learning_params& pars) {
//	const vector<Mat>& prev_datas = m_input->to_2d_data_layer()->get_data();
//	const Size& prev_image_size = m_input->to_2d_data_layer()->get_image_size();
//	int prev_image_number = m_input->to_2d_data_layer()->get_image_number();
//
//	int sample_count = prev_datas.front().size.p[0];
//
//	m_ff_singal_caches.resize(prev_image_number);
//	m_input_max_masks.resize(prev_image_number);
//
//	for (int iter_output = 0; iter_output < prev_image_number; ++iter_output) {
//		ml_mat_op::ml_pooling(m_ff_singal_caches[iter_output], m_input_max_masks[iter_output], prev_datas[iter_output], m_kernel_size, m_type, m_stride);
//	}
//
//	m_output->to_2d_data_layer()->feedforward_singal(m_ff_singal_caches, pars);
//}
//
//void ml_nn_pooling_linked_layer::backpropagation(const ml_nn_layer_learning_params& pars) {
//	int next_layer_image_number = m_output->to_2d_data_layer()->get_image_number();
//	int prev_layer_image_number = m_input->to_2d_data_layer()->get_image_number();
//
//	const vector<Mat>& next_deltas = m_output->to_2d_data_layer()->get_delta();
//	const Size& prev_image_size = m_input->to_2d_data_layer()->get_image_size();
//
//	m_bp_singal_caches.resize(prev_layer_image_number);
//
//	for (int iter_image = 0; iter_image < prev_layer_image_number; ++iter_image) {		
//		ml_mat_op::ml_unpooling(m_bp_singal_caches[iter_image], prev_image_size, next_deltas[iter_image], m_input_max_masks[iter_image], m_kernel_size, m_type, m_stride);
//	}
//
//	m_input->to_2d_data_layer()->backprapogation_singal(m_bp_singal_caches, pars);
//}
//
//ml_nn_layer* ml_nn_pooling_linked_layer::clone() const {
//	ml_nn_pooling_linked_layer* layer = new ml_nn_pooling_linked_layer(m_unique_name);
//	layer->m_input = m_input;
//	layer->m_output = m_output;
//	layer->m_weight_update_setting = m_weight_update_setting;
//	layer->m_bias_update_setting = m_bias_update_setting;
//
//	layer->m_kernel_size = m_kernel_size;
//	layer->m_type = m_type;
//	layer->m_stride = m_stride;
//
//	return layer;
//}
//
//void ml_nn_pooling_linked_layer::inner_compute_default_setting() {
//	m_output->to_2d_data_layer()->set_image_size(ml_mat_op::get_pooling_result_size(m_input->to_2d_data_layer()->get_image_size(), m_kernel_size, m_stride));
//	m_output->to_2d_data_layer()->set_image_number(m_input->to_2d_data_layer()->get_image_number());
//}
//
//void ml_nn_2d_maxout_linked_layer::inner_set_check_auto_params() {
//	m_output->to_2d_data_layer()->set_image_number((m_input->to_2d_data_layer()->get_image_number() - 1) / m_k + 1);
//	m_output->to_2d_data_layer()->set_image_size(m_input->to_2d_data_layer()->get_image_size());
//}
//
//void ml_nn_2d_maxout_linked_layer::feedforward(const ml_nn_layer_learning_params& pars) {
//	int prev_image_number = m_input->to_2d_data_layer()->get_image_number();
//	int next_image_number = m_output->to_2d_data_layer()->get_image_number();
//	const vector<Mat>& prev_data = m_input->to_2d_data_layer()->get_data();
//
//	m_ff_singal_caches.resize(next_image_number);
//	m_next_input_max_masks.resize(next_image_number);
//
//	int input_start_index = 0;
//	int input_stop_index = m_k;
//
//	for (int iter_out_image = 0; iter_out_image < next_image_number; ++iter_out_image) {		
//		if (input_stop_index > next_image_number) {
//			input_stop_index = next_image_number;
//		}
//
//		Mat& out_mat = m_ff_singal_caches[iter_out_image];
//		Mat& out_max_index_mat = m_next_input_max_masks[iter_out_image];
//
//		prev_data[input_start_index].copyTo(out_mat);
//		out_max_index_mat.create(3, prev_data.front().size, CV_32S);
//		ml_mat_op::set(out_max_index_mat, Scalar(input_start_index));
//
//		for (int k = input_start_index + 1; k < input_stop_index; ++k) {
//			if (prev_data.front().type() == CV_32FC1) {
//				private_nn_layer::max_out_2d<float>(out_mat, out_max_index_mat, prev_data[k], k);
//			} else if (prev_data.front().type() == CV_64FC1) {
//				private_nn_layer::max_out_2d<double>(out_mat, out_max_index_mat, prev_data[k], k);
//			} else {
//				BASICLOGASSERT(false);
//			}
//		}
//
//		input_start_index += m_k;
//		input_stop_index += m_k;
//	}
//
//	m_output->to_2d_data_layer()->feedforward_singal(m_ff_singal_caches, pars);
//}
//
//void ml_nn_2d_maxout_linked_layer::backpropagation(const ml_nn_layer_learning_params& pars) {
//	int prev_image_number = m_input->to_2d_data_layer()->get_image_number();
//	int next_image_number = m_output->to_2d_data_layer()->get_image_number();
//	const vector<Mat>& prev_data = m_input->to_2d_data_layer()->get_data();
//
//	m_bp_singal_caches.resize(prev_image_number);
//
//	for (int iter_input = 0; iter_input < prev_image_number; ++iter_input) {
//		m_bp_singal_caches[iter_input].create(3, prev_data.front().size, prev_data.front().type());
//		ml_mat_op::set(m_bp_singal_caches[iter_input], Scalar(0));
//	}
//
//	const vector<Mat>& next_deltas = m_output->to_2d_data_layer()->get_delta();
//
//	for (int iter_out = 0; iter_out < next_image_number; ++iter_out) {
//		const Mat& next_layer_dalta_mat = next_deltas[iter_out];
//		Mat& next_layer_max_index_mat = m_next_input_max_masks[iter_out];
//
//		uchar* ptr_next_layer_dalta_dim0 = next_layer_dalta_mat.data;
//		uchar* ptr_next_layer_max_index_dim0 = next_layer_max_index_mat.data;
//
//		for (int iter_sample = 0; iter_sample < next_layer_dalta_mat.size.p[0]; ++iter_sample) {
//			uchar* ptr_next_layer_dalta_dim1 = ptr_next_layer_dalta_dim0;
//			uchar* ptr_next_layer_max_index_dim1 = ptr_next_layer_max_index_dim0;
//
//			for (int next_delta_row = 0; next_delta_row < next_layer_dalta_mat.size.p[1]; ++next_delta_row) {
//				if (CV_32FC1 == prev_data.front().type()) {
//					float* ptr_next_layer_dalta = (float*)ptr_next_layer_dalta_dim1;
//					int* ptr_next_layer_max_index = (int*)ptr_next_layer_max_index_dim1;	
//
//					for (int next_delta_col = 0; next_delta_col < next_layer_dalta_mat.size.p[2]; ++next_delta_col) {
//						Mat& max_index_corresponding_mat = m_bp_singal_caches[*ptr_next_layer_max_index++];
//						max_index_corresponding_mat.at<float>(iter_sample, next_delta_row, next_delta_col) = *ptr_next_layer_dalta++;
//					}
//				} else if (CV_64FC1 == prev_data.front().type()) {
//					double* ptr_next_layer_dalta = (double*)ptr_next_layer_dalta_dim1;
//					int* ptr_next_layer_max_index = (int*)ptr_next_layer_max_index_dim1;	
//
//					for (int next_delta_col = 0; next_delta_col < next_layer_dalta_mat.size.p[2]; ++next_delta_col) {
//						Mat& max_index_corresponding_mat = m_bp_singal_caches[*ptr_next_layer_max_index++];
//						max_index_corresponding_mat.at<double>(iter_sample, next_delta_row, next_delta_col) = *ptr_next_layer_dalta++;
//					}
//				}						
//
//				ptr_next_layer_dalta_dim1 += next_layer_dalta_mat.step.p[1];
//				ptr_next_layer_max_index_dim1 += next_layer_max_index_mat.step.p[1];
//			}
//
//			ptr_next_layer_dalta_dim0 += next_layer_dalta_mat.step.p[0];
//			ptr_next_layer_max_index_dim0 += next_layer_max_index_mat.step.p[0];
//		}
//	}
//
//	m_input->to_2d_data_layer()->backprapogation_singal(m_bp_singal_caches, pars);
//}
//
//ml_nn_layer* ml_nn_2d_maxout_linked_layer::clone() const {
//	ml_nn_2d_maxout_linked_layer* layer = new ml_nn_2d_maxout_linked_layer(m_unique_name);
//	layer->m_input = m_input;
//	layer->m_output = m_output;
//	layer->m_k = m_k;
//	layer->m_weight_update_setting = m_weight_update_setting;
//	layer->m_bias_update_setting = m_bias_update_setting;
//
//	return layer;
//}
//
//void ml_nn_1d_maxout_linked_layer::inner_set_check_auto_params() {
//	m_output->to_1d_data_layer()->set_unit_number((m_input->to_1d_data_layer()->get_unit_number() - 1) / m_k + 1);
//}
//
//void ml_nn_1d_maxout_linked_layer::feedforward(const ml_nn_layer_learning_params& pars) {
//	Mat& prev_data = m_input->to_1d_data_layer()->get_data();
//	int next_unit_number = m_output->to_1d_data_layer()->get_unit_number();
//
//	m_ff_singal_caches.create(prev_data.rows, next_unit_number, prev_data.type());
//	m_next_input_max_masks.create(prev_data.rows, next_unit_number, CV_32SC1);
//
//	if (prev_data.type() == CV_32FC1) {
//		private_nn_layer::max_out_1d<float>(m_ff_singal_caches, m_next_input_max_masks, prev_data, m_k);
//	} else if (prev_data.type() == CV_64FC1) {
//		private_nn_layer::max_out_1d<double>(m_ff_singal_caches, m_next_input_max_masks, prev_data, m_k);
//	} else {
//		BASICLOGASSERT(false);
//	}
//
//	m_output->to_1d_data_layer()->feedforward_singal(m_ff_singal_caches, pars);
//}
//
//void ml_nn_1d_maxout_linked_layer::backpropagation(const ml_nn_layer_learning_params& pars) {
//	const Mat& next_delta = m_output->to_1d_data_layer()->get_delta();
//	int prev_unit_number = m_input->to_1d_data_layer()->get_unit_number();
//
//	m_bp_singal_cache.create(next_delta.rows, prev_unit_number, next_delta.type());
//	ml_mat_op::set(m_bp_singal_cache, Scalar(0));
//	
//	if (next_delta.type() == CV_32FC1) {
//		private_nn_layer::restore_max_out_1d<float>(m_bp_singal_cache, next_delta, m_next_input_max_masks);
//	} else if (next_delta.type() == CV_64FC1) {
//		private_nn_layer::restore_max_out_1d<double>(m_bp_singal_cache, next_delta, m_next_input_max_masks);
//	} else {
//		BASICLOGASSERT(false);
//	}
//	
//	m_input->to_1d_data_layer()->backprapogation_singal(m_bp_singal_cache, pars);
//}
//
//ml_nn_layer* ml_nn_1d_maxout_linked_layer::clone() const {
//	ml_nn_1d_maxout_linked_layer* layer = new ml_nn_1d_maxout_linked_layer(m_unique_name);
//	layer->m_input = m_input;
//	layer->m_output = m_output;
//
//	layer->m_k = m_k;
//	layer->m_weight_update_setting = m_weight_update_setting;
//	layer->m_bias_update_setting = m_bias_update_setting;
//
//	return layer;
//}
//
//void ml_nn_combination_linked_layer::feedforward(const ml_nn_layer_learning_params& pars) {
//	const vector<Mat>& prev_datas = m_input->to_2d_data_layer()->get_data();
//	const Size& prev_image_size = m_input->to_2d_data_layer()->get_image_size();
//	int prev_image_number = m_input->to_2d_data_layer()->get_image_number();
//
//	int sample_count = prev_datas.front().size.p[0];
//	int data_type = prev_datas.front().type();
//	int prev_image_size_area = prev_image_size.area();
//	m_ff_singal_cache.create(sample_count, prev_image_size_area * prev_image_number, data_type);
//
//	uchar* ptr_next_input_cache_dim0 = m_ff_singal_cache.data;
//
//	int image_byte = prev_image_size_area;
//
//	if (data_type == CV_32FC1) {
//		image_byte *= sizeof(float);
//	} else {
//		image_byte *= sizeof(double);
//	}
//
//	for (int iter_sample = 0; iter_sample < sample_count; ++iter_sample) {
//		uchar* ptr_next_input_cache_dim1 = ptr_next_input_cache_dim0;
//
//		for (int iter_input = 0; iter_input < prev_image_number; ++iter_input) {
//			const uchar* imagePointer = prev_datas[iter_input].ptr<uchar>(iter_sample);
//
//			memcpy(ptr_next_input_cache_dim1, imagePointer, image_byte);
//			ptr_next_input_cache_dim1 += image_byte;
//		}
//
//		ptr_next_input_cache_dim0 += m_ff_singal_cache.step.p[0];
//	}
//
//	//for (int iter_input = 0; iter_input < prev_image_number; ++iter_input) {
//	//	ml_mat_op::traceMat(prev_datas[iter_input]);	
//	//}
//
//	//ml_mat_op::traceMat(m_next_input_data_cache);
//	m_output->to_1d_data_layer()->feedforward_singal(m_ff_singal_cache, pars);
//}
//
//void ml_nn_combination_linked_layer::backpropagation(const ml_nn_layer_learning_params& pars) {
//	const Mat& next_delta = m_output->to_1d_data_layer()->get_delta();
//	const Size& prev_image_size = m_input->to_2d_data_layer()->get_image_size();
//	int prev_image_number = m_input->to_2d_data_layer()->get_image_number();
//
//	m_bp_singal_caches.resize(prev_image_number);
//
//	int prev_image_size_area = prev_image_size.area();
//	int dim_sizes[] = {next_delta.rows, prev_image_size.height, prev_image_size.width};
//
//	for (int iter_image = 0; iter_image < prev_image_number; ++iter_image) {
//		Mat col_range = next_delta.colRange(iter_image * prev_image_size_area, (iter_image + 1) * prev_image_size_area);
//
//		size_t dim_steps[] = {col_range.step.p[0], prev_image_size.width * col_range.step.p[1], col_range.step.p[1]};
//		m_bp_singal_caches[iter_image] = Mat(3, dim_sizes, col_range.type(), col_range.data);
//	}
//
//	m_input->to_2d_data_layer()->backprapogation_singal(m_bp_singal_caches, pars);
//}
//
//ml_nn_layer* ml_nn_combination_linked_layer::clone() const {
//	ml_nn_combination_linked_layer* layer = new ml_nn_combination_linked_layer(m_unique_name);
//	layer->m_input = m_input;
//	layer->m_output = m_output;
//	layer->m_weight_update_setting = m_weight_update_setting;
//	layer->m_bias_update_setting = m_bias_update_setting;
//
//	return layer;
//}
//
//void ml_nn_combination_linked_layer::inner_compute_default_setting() {
//	const Size& prev_image_size = m_input->to_2d_data_layer()->get_image_size();
//	int prev_image_number = m_input->to_2d_data_layer()->get_image_number();
//
//	m_output->to_1d_data_layer()->set_unit_number(prev_image_size.area() * prev_image_number);
//}
//
//void ml_nn_1d_data_layer::feedforward_singal(const Mat& singal, const ml_nn_layer_learning_params& pars) {
//	if (m_feedforward_count == 0 || m_data_cache.empty()) {
//		m_data_cache = singal;
//	} else {
//		m_data_cache += singal;
//	}
//
//	++m_feedforward_count;
//
//	if (m_feedforward_count >= (int)m_prev_linked_layers.size()) {
//		calculate_activate();
//		
//		for (int i = 0; i < (int)m_next_linked_layers.size(); ++i) {
//			m_next_linked_layers[i]->feedforward(pars);
//		}
//
//		m_feedforward_count = 0;
//	}
//}
//
//void ml_nn_1d_data_layer::feedforward_singal(vector<Mat>& drawn_singals, const ml_nn_layer_learning_params& pars) {
//	if (m_feedforward_count == 0 || m_drawn_singals.empty()) {
//		m_drawn_singals = drawn_singals;
//	} else {
//		BASICLOGASSERT(m_drawn_singals.size() == drawn_singals.size());
//
//		for (int i = 0; i < (int)drawn_singals.size(); ++i) {
//			m_drawn_singals[i] += drawn_singals[i];
//		}
//	}
//
//	++m_feedforward_count;
//
//	if (m_feedforward_count >= (int)m_prev_linked_layers.size()) {
//		calculate_activate();
//
//		for (int i = 0; i < (int)m_next_linked_layers.size(); ++i) {
//			m_next_linked_layers[i]->feedforward(pars);
//		}
//
//		m_feedforward_count = 0;
//	}
//}
//
//void ml_nn_1d_data_layer::calculate_activate() {
//	if (m_drawn_singals.empty()) {
//		ml_define::activate(m_data_cache, m_data_cache, m_activate_func_type, m_activate_params.empty() ? NULL : &m_activate_params[0]);
//	} else {
//		Mat temp_data = Mat::zeros(m_drawn_singals.front().rows, m_drawn_singals.front().cols, m_drawn_singals.front().type());
//
//		for (int i = 0; i < (int)m_drawn_singals.size(); ++i) {
//
//			if (!m_data_cache.empty()) {
//				m_drawn_singals[i] += m_data_cache;
//			}
//
//			ml_define::activate(m_drawn_singals[i], m_drawn_singals[i], m_activate_func_type, m_activate_params.empty() ? NULL : &m_activate_params[0]);
//			temp_data += m_drawn_singals[i];
//		}
//
//		m_data_cache = temp_data / (int)m_drawn_singals.size();
//	}	
//}
//
//void ml_nn_1d_data_layer::backprapogation_singal(const Mat& singal, const ml_nn_layer_learning_params& pars) {
//	if (m_backprapogation_count == 0) {
//		m_delta_cache = singal;
//	} else {
//		m_delta_cache += singal;
//	}
//
//	++m_backprapogation_count;
//
//	if (m_backprapogation_count >= (int)m_next_linked_layers.size()) {
//
//		if (ml_activate_func_linear != m_activate_func_type) {
//			ml_define::activate_derivative(m_data_derivative_cache, m_data_cache, m_activate_func_type, m_activate_params.empty() ? NULL : &m_activate_params[0]);
//			multiply(m_delta_cache, m_data_derivative_cache, m_delta_cache);
//		}
//
//		for (int i = 0; i < (int)m_prev_linked_layers.size(); ++i) {
//			m_prev_linked_layers[i]->backpropagation(pars);
//		}
//
//		m_backprapogation_count = 0;
//	}
//}
//
//ml_nn_layer* ml_nn_1d_data_layer::clone() const {
//	ml_nn_1d_data_layer* layer = new ml_nn_1d_data_layer(m_unique_name);
//	layer->m_activate_func_type = m_activate_func_type;
//	layer->m_activate_params = m_activate_params;
//	layer->m_unit_number = m_unit_number;
//	
//	return layer;
//}
//
//void ml_nn_2d_data_layer::feedforward_singal(const vector<Mat>& singals, const ml_nn_layer_learning_params& pars) {
//	if (m_feedforward_count == 0) {
//		m_data_caches = singals;
//	} else {
//		for (int i = 0; i < (int)singals.size(); ++i) {
//			m_data_caches[i] += singals[i];
//		}		
//	}
//
//	++m_feedforward_count;
//
//	if (m_feedforward_count >= (int)m_prev_linked_layers.size()) {
//
//		for (int i = 0; i < (int)m_data_caches.size(); ++i) {
//			ml_define::activate(m_data_caches[i], m_data_caches[i], m_activate_func_type, m_activate_params.empty() ? NULL : &m_activate_params[0]);
//		}
//
//		for (int i = 0; i < (int)m_next_linked_layers.size(); ++i) {
//			m_next_linked_layers[i]->feedforward(pars);
//		}
//
//		m_feedforward_count = 0;
//	}
//}
//
//void ml_nn_2d_data_layer::backprapogation_singal(const vector<Mat>& singals, const ml_nn_layer_learning_params& pars) {
//	if (m_backprapogation_count == 0) {
//		m_delta_caches = singals;
//	} else {
//		for (int i = 0; i < (int)singals.size(); ++i) {
//			m_delta_caches[i] += singals[i];
//		}	
//	}
//
//	++m_backprapogation_count;
//
//	if (m_backprapogation_count >= (int)m_next_linked_layers.size()) {
//
//		if (ml_activate_func_linear != m_activate_func_type) {
//			m_data_derivative_caches.resize(m_data_caches.size());
//
//			for (int i = 0; i < m_data_caches.size(); ++i) {
//				ml_define::activate(m_data_derivative_caches[i], m_data_caches[i], m_activate_func_type, m_activate_params.empty() ? NULL : &m_activate_params[0]);
//				multiply(m_delta_caches[i], m_data_derivative_caches[i], m_delta_caches[i]);
//			}					
//		}
//
//		for (int i = 0; i < (int)m_prev_linked_layers.size(); ++i) {
//			m_prev_linked_layers[i]->backpropagation(pars);
//		}
//
//		m_backprapogation_count = 0;
//	}
//}
//
//ml_nn_layer* ml_nn_2d_data_layer::clone() const {
//	ml_nn_2d_data_layer* layer = new ml_nn_2d_data_layer(m_unique_name);
//	layer->m_data_caches = m_data_caches;
//	layer->m_delta_caches = m_delta_caches;
//
//	layer->m_image_number = m_image_number;
//	layer->m_image_size = m_image_size;
//
//	return layer;
//}
//
//
//void ml_nn_1d_input_data_layer::feedforward_by_input(const Mat& input, const ml_nn_layer_learning_params& pars) {
//	BASICLOGASSERT(input.channels() == 1);
//	feedforward_singal(input, pars);
//}
//
//void ml_nn_1d_input_data_layer::backprapogation_singal(const Mat& delta, const ml_nn_layer_learning_params& pars) {
//	NULL;
//}
//
//ml_nn_layer* ml_nn_1d_input_data_layer::clone() const {
//	ml_nn_1d_input_data_layer* layer = new ml_nn_1d_input_data_layer(m_unique_name);
//	layer->m_activate_func_type = m_activate_func_type;
//	layer->m_activate_params = m_activate_params;
//	layer->m_unit_number = m_unit_number;
//
//	return layer;
//}
//
//void ml_nn_2d_input_data_layer::feedforward_by_input(const Mat& inputs, const ml_nn_layer_learning_params& pars) {
//	BASICLOG_ASSERT_MESSAGE(inputs.channels() == m_image_number, L"the channel of input feature does not match the image number of setting");
//	BASICLOG_ASSERT_MESSAGE(inputs.cols == m_image_size.area(), L"the dim of input feature does not match the image size of setting");
//	
//	m_data_caches.resize(inputs.channels());
//	m_splited_mats.resize(inputs.channels());
//	split(inputs, m_splited_mats);
//		
//	for (int iter_channel = 0; iter_channel < inputs.channels(); ++iter_channel) {
//		Mat& splited_mat = m_splited_mats[iter_channel];
//		int dimSizes[] = {splited_mat.rows, m_image_size.height, m_image_size.width};
//		size_t steps[] = {splited_mat.step.p[0], m_image_size.width * splited_mat.step.p[1], splited_mat.step.p[1]};
//		m_data_caches[iter_channel] = Mat(3, dimSizes, splited_mat.type(), splited_mat.data, steps);
//	}
//
//	for (int i = 0; i < (int)m_next_linked_layers.size(); ++i) {
//		m_next_linked_layers[i]->feedforward(pars);
//	}
//}
//
//void ml_nn_2d_input_data_layer::backprapogation_singal(const vector<Mat>& singals, const ml_nn_layer_learning_params& pars) {
//	NULL;
//}
//
//ml_nn_layer* ml_nn_2d_input_data_layer::clone() const {
//	ml_nn_2d_input_data_layer* layer = new ml_nn_2d_input_data_layer(m_unique_name);
//	layer->m_activate_func_type = m_activate_func_type;
//	layer->m_activate_params = m_activate_params;
//	layer->m_image_size = m_image_size;
//	layer->m_image_number = m_image_number;
//
//	return layer;
//}
//
//void ml_nn_output_data_layer::backprapogation_by_label(const Mat& label, const ml_nn_layer_learning_params& pars) {
//	BASICLOGASSERT(m_data_cache.cols == label.cols);
//
//	//ml_mat_op::traceMat(label);
//	//ml_mat_op::traceMat(m_data_cache);
//
//	Mat error = m_data_cache - label;
//	switch (m_loss_func_type) {
//	case ml_loss_func_quardratic:
//		if (ml_activate_func_linear == m_activate_func_type) {
//			m_delta_cache = error;
//		} else if (ml_activate_func_sigmoid == m_activate_func_type) {
//			ml_define::sigma_derivative(m_delta_cache, m_data_cache);
//			multiply(m_delta_cache, error, m_delta_cache);
//		}
//
//		break;
//	case ml_loss_func_logarithmic:
//		if (label.cols == 1) {
//			BASICLOGASSERT(ml_activate_func_sigmoid == m_activate_func_type);
//		} else {
//			BASICLOGASSERT(ml_activate_func_softmax == m_activate_func_type);
//		}
//
//		m_delta_cache = error;
//		break;
//	}
//
//	m_delta_cache *= m_task_weight;
//	
//	for (int i = 0; i < (int)m_prev_linked_layers.size(); ++i) {
//		m_prev_linked_layers[i]->backpropagation(pars);
//	}
//}
//
//ml_nn_layer* ml_nn_output_data_layer::clone() const {
//	ml_nn_output_data_layer* layer = new ml_nn_output_data_layer(m_unique_name);
//	layer->m_activate_func_type = m_activate_func_type;
//	layer->m_activate_params = m_activate_params;
//	layer->m_unit_number = m_unit_number;
//	
//	layer->m_is_classifier = m_is_classifier;
//	layer->m_loss_func_type = m_loss_func_type;
//	layer->m_task_weight = m_task_weight;
//
//	return layer;
//}
//
//void ml_nn_1d_data_layer::to_file(ml_file_storage& fs, bool write_learned_params) const {
//	fs<<L"{";
//	fs<<L"layer_type"<<get_layer_type();
//
//	if (m_unit_number != ml_nn_1d_data_layer_default_unit_number) {
//		fs<<L"unit_number"<<m_unit_number;
//	}
//	
//	if (m_activate_func_type != ml_nn_data_layer_default_activate_type) {
//		fs<<L"activate_type"<<ml_activate_funcs[m_activate_func_type];
//	}
//	
//	if (!m_activate_params.empty()) {
//		fs<<L"activate_params"<<m_activate_params;
//	}
//
//	fs<<L"}";
//}
//
//ml_nn_layer* ml_nn_1d_data_layer::from_file(const ml_file_node& node) {
//	if (wstring(node[L"layer_type"]) != ml_nn_1d_data_layer::get_layer_type()) {
//		return NULL;
//	}
//
//	ml_nn_1d_data_layer* layer = new ml_nn_1d_data_layer(node.name());
//
//	if (!node[L"unit_number"].empty()) {
//		layer->m_unit_number = node[L"unit_number"];
//	}
//
//	if (!node[L"activate_type"].empty()) {
//		layer->m_activate_func_type = ml_define::activate_func_index(node[L"activate_type"]);
//	}
//
//	if (!node[L"activate_params"].empty()) {
//		layer->m_activate_params = node[L"activate_params"];
//	}	
//
//	return layer;
//}
//
//void ml_nn_1d_input_data_layer::to_file(ml_file_storage& fs, bool write_learned_params) const {
//	fs<<L"{";
//	fs<<L"layer_type"<<get_layer_type();
//
//	fs<<L"unit_number"<<m_unit_number;
//	
//	fs<<L"}";
//}
//
//ml_nn_layer* ml_nn_1d_input_data_layer::from_file(const ml_file_node& node) {
//	if (wstring(node[L"layer_type"]) != ml_nn_1d_input_data_layer::get_layer_type()) {
//		return NULL;
//	}
//
//	ml_nn_1d_input_data_layer* layer = new ml_nn_1d_input_data_layer(node.name());
//
//	if (!node[L"unit_number"].empty()) {
//		layer->m_unit_number = node[L"unit_number"];
//	} else {
//		BASICLOGERROR_MESSAGE(L"1d input layer must be set its unit number");
//	}
//
//	
//
//	return layer;
//}
//
//
//void ml_nn_2d_input_data_layer::to_file(ml_file_storage& fs, bool write_learned_params) const {
//	fs<<L"{";
//	fs<<L"layer_type"<<get_layer_type();
//	fs<<L"image_number"<<m_image_number;
//	fs<<L"image_size"<<m_image_size;
//
//	fs<<L"}";
//}
//
//ml_nn_layer* ml_nn_2d_input_data_layer::from_file(const ml_file_node& node) {
//	if (wstring(node[L"layer_type"]) != ml_nn_2d_input_data_layer::get_layer_type()) {
//		return NULL;
//	}
//
//	ml_nn_2d_input_data_layer* layer = new ml_nn_2d_input_data_layer(node.name());
//
//	if (!node[L"image_number"].empty()) {
//		layer->m_image_number = node[L"image_number"];
//	} else {
//		BASICLOGERROR_MESSAGE(L"1d input layer must be set its image number");
//	}
//
//	if (!node[L"image_size"].empty()) {
//		layer->m_image_size = node[L"image_size"];
//	} else {
//		BASICLOGERROR_MESSAGE(L"1d input layer must be set its image size");
//	}
//
//	return layer;
//}
//
//void ml_nn_2d_data_layer::to_file(ml_file_storage& fs, bool write_learned_params) const {
//	fs<<L"{";
//	fs<<L"layer_type"<<get_layer_type();
//
//	if (m_image_number != ml_nn_2d_data_layer_default_image_number) {
//		fs<<L"image_number"<<m_image_number;
//	}
//
//	if (m_image_size != ml_nn_2d_data_layer_default_image_size) {
//		fs<<L"image_size"<<m_image_size;
//	}
//
//	if (m_activate_func_type != ml_nn_data_layer_default_activate_type) {
//		fs<<L"activate_type"<<ml_activate_funcs[m_activate_func_type];
//	}
//
//	if (!m_activate_params.empty()) {
//		fs<<L"activate_params"<<m_activate_params;
//	}
//
//	fs<<L"}";
//}
//
//ml_nn_layer* ml_nn_2d_data_layer::from_file(const ml_file_node& node) {
//	if (wstring(node[L"layer_type"]) != ml_nn_2d_data_layer::get_layer_type()) {
//		return NULL;
//	}
//
//	ml_nn_2d_data_layer* layer = new ml_nn_2d_data_layer(node.name());
//
//	if (!node[L"image_number"].empty()) {
//		layer->m_image_number = node[L"image_number"];
//	}
//
//	if (!node[L"image_size"].empty()) {
//		layer->m_image_size = node[L"image_size"];
//	}
//	
//	if (!node[L"activate_type"].empty()) {
//		layer->m_activate_func_type = ml_define::activate_func_index(node[L"activate_type"]);
//	}
//	
//	if (!node[L"activate_params"].empty()) {
//		layer->m_activate_params = node[L"activate_params"];
//	}
//	
//	return layer;
//}
//
//void ml_nn_inner_product_linked_layer::to_file(ml_file_storage& fs, bool write_learned_params) const {
//	fs<<L"{";
//	fs<<L"layer_type"<<get_layer_type();
//	fs<<L"input_layer"<<m_input->get_unique_name();
//	fs<<L"output_layer"<<m_output->get_unique_name();
//
//	if (m_weight_update_setting != ml_learned_param_update_setting()) {
//		fs<<L"weight_update_setting"<<m_weight_update_setting;
//	}
//
//	if (m_bias_update_setting != ml_learned_param_update_setting()) {
//		fs<<L"bias_update_setting"<<m_bias_update_setting;
//	}
//
//	if (m_drop_type != Drop_Null) {
//		fs<<L"drop_type"<<m_drop_type;
//		fs<<L"inference_type"<<m_inference_type;
//		fs<<L"drawn_number"<<m_drwan_number;
//		fs<<L"ratio"<<m_ratio;
//	}
//
//	if (write_learned_params) {
//		fs<<L"weight"<<m_weight;
//		fs<<L"bias"<<m_bias;
//		fs<<L"v_weight"<<m_v_weight;
//		fs<<L"v_bias"<<m_v_bias;
//	}
//
//	fs<<L"}";
//}
//
//ml_nn_layer* ml_nn_inner_product_linked_layer::from_file(const ml_file_node& node) {
//	if (wstring(node[L"layer_type"]) != ml_nn_inner_product_linked_layer::get_layer_type()) {
//		return NULL;
//	}
//
//	ml_nn_inner_product_linked_layer* layer = new ml_nn_inner_product_linked_layer(node.name());
//	if (!node[L"input_layer"].empty()) {
//		layer->m_input = new ml_nn_1d_data_layer(node[L"input_layer"]);
//	} else {
//		BASICLOGERROR_MESSAGE(L"linked layer must have input data layer");
//	}
//
//	if (!node[L"output_layer"].empty()) {
//		layer->m_output = new ml_nn_1d_data_layer(node[L"output_layer"]);
//	} else {
//		BASICLOGERROR_MESSAGE(L"linked layer must have output data layer");
//	}
//
//	node[L"weight_update_setting"]>>layer->m_weight_update_setting;
//	node[L"bias_update_setting"]>>layer->m_bias_update_setting;
//
//	if (!node[L"drop_type"].empty()) {
//		node[L"drop_type"]>>layer->m_drop_type;
//		node[L"inference_type"]>>layer->m_inference_type;
//		node[L"drawn_number"]>>layer->m_drwan_number;
//		node[L"ratio"]>>layer->m_ratio;
//	} else {
//		layer->m_drop_type = Drop_Null;
//	}
//
//	ml_file_node weight_node = node[L"weight"];
//
//	if (!weight_node.empty()) {
//		if (!node[L"weight"].empty()) {
//			layer->m_weight = node[L"weight"];
//
//			if (layer->m_weight.empty()) {
//				BASICLOGERROR_MESSAGE(L"weight matrix contains no element");
//			}
//		}
//		
//		if (!node[L"bias"].empty()) {
//			layer->m_bias = node[L"bias"];
//
//			if (layer->m_bias.empty()) {
//				BASICLOGERROR_MESSAGE(L"bias matrix contains no element");
//			}
//		} else {
//			BASICLOGERROR_MESSAGE(L"can not load bias");
//		}
//		
//		if (!node[L"v_weight"].empty()) {
//			layer->m_v_weight = node[L"v_weight"];
//
//			if (layer->m_bias.empty()) {
//				BASICLOGERROR_MESSAGE(L"v_weight matrix contains no element");
//			}
//
//		} else {
//			BASICLOGERROR_MESSAGE(L"can not load v_weight");
//		}
//
//		if (!node[L"v_bias"].empty()) {
//			layer->m_v_bias = node[L"v_bias"];
//
//			if (layer->m_bias.empty()) {
//				BASICLOGERROR_MESSAGE(L"v_bias matrix contains no element");
//			}
//
//		} else {
//			BASICLOGERROR_MESSAGE(L"can not load v_bias");
//		}		
//	}
//
//	return layer;
//}
//
//void ml_nn_convolution_linked_layer::to_file(ml_file_storage& fs, bool write_learned_params) const {
//	fs<<L"{";
//	fs<<L"layer_type"<<get_layer_type();
//	fs<<L"input_layer"<<m_input->get_unique_name();
//	fs<<L"output_layer"<<m_output->get_unique_name();
//
//	if (m_conv_type != ml_nn_convolution_linked_layer_default_conv_type) {
//		if (m_conv_type == ml_mat_op::Conv_Type_Conv) {
//			fs<<L"conv_type"<<L"convolution";
//		} else {
//			fs<<L"conv_type"<<L"correlation";
//		}
//	}
//	
//	if (m_weight_update_setting != ml_learned_param_update_setting()) {
//		fs<<L"weight_update_setting"<<m_weight_update_setting;
//	}
//
//	if (m_bias_update_setting != ml_learned_param_update_setting()) {
//		fs<<L"bias_update_setting"<<m_bias_update_setting;
//	}
//
//	fs<<L"kernel_size"<<m_kernel_size;
//
//	if (m_stride != ml_nn_convolution_linked_layer_default_stride) {
//		fs<<L"stride"<<m_stride;
//	}
//
//	if (m_tl_pad_size != ml_nn_convolution_linked_layer_default_pad_size) {
//		fs<<L"tl_pad_size"<<m_tl_pad_size;
//	}
//	
//	if (m_br_pad_size != ml_nn_convolution_linked_layer_default_pad_size) {
//		fs<<L"br_pad_size"<<m_br_pad_size;
//	}
//
//	if (write_learned_params) {
//		fs<<L"prev_image_number"<<(int)m_kernels.size();
//		fs<<L"next_image_number"<<(int)m_biases.size();
//
//		for (int i = 0; i < (int)m_kernels.size(); ++i) {
//			for (int j = 0; j < (int)m_biases.size(); ++j) {
//				fs<<(ml_string()<<L"kernel_"<<i<<L"_"<<j)<<m_kernels[i][j];
//				fs<<(ml_string()<<L"v_kernel_"<<i<<L"_"<<j)<<m_v_kernels[i][j];
//			}
//		}
//
//		fs<<(ml_string()<<L"biases")<<m_biases;
//		fs<<(ml_string()<<L"v_biases")<<m_v_biases;
//	}
//
//	fs<<L"}";
//}
//
//ml_nn_layer* ml_nn_convolution_linked_layer::from_file(const ml_file_node& node) {
//	if (wstring(node[L"layer_type"]) != ml_nn_convolution_linked_layer::get_layer_type()) {
//		return NULL;
//	}
//
//	ml_nn_convolution_linked_layer* layer = new ml_nn_convolution_linked_layer(node.name());
//	if (!node[L"input_layer"].empty()) {
//		layer->m_input = new ml_nn_2d_data_layer(node[L"input_layer"]);
//	} else {
//		BASICLOGERROR_MESSAGE(L"linked layer must have input data layer");
//	}
//
//	if (!node[L"output_layer"].empty()) {
//		layer->m_output = new ml_nn_2d_data_layer(node[L"output_layer"]);
//	} else {
//		BASICLOGERROR_MESSAGE(L"linked layer must have output data layer");
//	}
//
//	node[L"weight_update_setting"]>>layer->m_weight_update_setting;
//	node[L"bias_update_setting"]>>layer->m_bias_update_setting;
//
//	if (!node[L"kernel_size"].empty()) {
//		layer->m_kernel_size = node[L"kernel_size"];
//	} else {
//		BASICLOGERROR_MESSAGE(L"convolution linked layer must be set its kernel size");
//	}
//
//	if (!node[L"conv_type"].empty()) {
//		wstring conv_type = node[L"conv_type"];
//		if (conv_type == L"convolution") {
//			layer->m_conv_type = ml_mat_op::Conv_Type_Conv;
//		} else {
//			layer->m_conv_type = ml_mat_op::Conv_Type_Corr;
//		}
//	}
//
//	node[L"stride"]>>layer->m_stride;
//	node[L"tl_pad_size"]>>layer->m_tl_pad_size;
//	node[L"br_pad_size"]>>layer->m_br_pad_size;
//
//	if (!node[L"prev_image_number"].empty()) {
//		int prev_img_number;
//		int next_img_number;
//
//		if (!node[L"prev_image_number"].empty()) {
//			prev_img_number = node[L"prev_image_number"];
//		} else {
//			BASICLOGASSERT(false);
//		}
//
//		if (!node[L"next_image_number"].empty()) {
//			next_img_number	= node[L"next_image_number"];
//		} else {
//			BASICLOGASSERT(false);
//		}
//		
//		vector<Mat> temp;
//		temp.resize(next_img_number);
//		layer->m_kernels.resize(prev_img_number, temp);
//		layer->m_v_kernels.resize(prev_img_number, temp);
//
//		for (int i = 0; i < prev_img_number; ++i) {
//			for (int j = 0; j < next_img_number; ++j) {
//				layer->m_kernels[i][j] = node[(ml_string()<<L"kernel_"<<i<<L"_"<<j)];
//				layer->m_v_kernels[i][j] = node[(ml_string()<<L"v_kernel_"<<i<<L"_"<<j)];
//			}
//		}
//
//		layer->m_biases = node[L"biases"];
//		layer->m_v_biases = node[L"v_biases"];
//	}
//
//	return layer;
//}
//
//void ml_nn_pooling_linked_layer::to_file(ml_file_storage& fs, bool write_learned_params) const {
//	fs<<L"{";
//	fs<<L"layer_type"<<get_layer_type();
//	fs<<L"pooling_type"<<ml_pooling_types[m_type];
//	fs<<L"input_layer"<<m_input->get_unique_name();
//	fs<<L"output_layer"<<m_output->get_unique_name();
//	
//	fs<<L"kernel_size"<<m_kernel_size;
//
//	if (m_stride !=  m_kernel_size) {
//		fs<<L"stride"<<m_stride;
//	}
//	
//
//	fs<<L"}";
//}
//
//
//ml_nn_layer* ml_nn_pooling_linked_layer::from_file(const ml_file_node& node) {
//	if (wstring(node[L"layer_type"]) != ml_nn_pooling_linked_layer::get_layer_type()) {
//		return NULL;
//	}
//
//	ml_nn_pooling_linked_layer* layer = new ml_nn_pooling_linked_layer(node.name());
//	if (!node[L"input_layer"].empty()) {
//		layer->m_input = new ml_nn_2d_data_layer(node[L"input_layer"]);
//	} else {
//		BASICLOGERROR_MESSAGE(L"linked layer must have input data layer");
//	}
//
//	if (!node[L"output_layer"].empty()) {
//		layer->m_output = new ml_nn_2d_data_layer(node[L"output_layer"]);
//	} else {
//		BASICLOGERROR_MESSAGE(L"linked layer must have output data layer");
//	}
//
//	if (!node[L"pooling_type"].empty()) {
//		wstring pooling_type = node[L"pooling_type"];
//		layer->m_type = ml_define::pooling_type_index(pooling_type);
//	} else {
//		BASICLOGERROR_MESSAGE(L"pooling linked layer must be set its pooling type");
//	}
//
//	if (!node[L"kernel_size"].empty()) {
//		layer->m_kernel_size = node[L"kernel_size"];
//	} else {
//		BASICLOGERROR_MESSAGE(L"pooling linked layer must be set its kernel size");
//	}
//
//	if (!node[L"stride"].empty()) {
//		layer->m_stride = node[L"stride"];
//	} else {
//		layer->m_stride = layer->m_kernel_size;
//	}	
//
//	return layer;
//}
//
//void ml_nn_1d_maxout_linked_layer::to_file(ml_file_storage& fs, bool write_learned_params) const {
//	fs<<L"{";
//	fs<<L"layer_type"<<get_layer_type();
//	fs<<L"input_layer"<<m_input->get_unique_name();
//	fs<<L"output_layer"<<m_output->get_unique_name();
//	
//	fs<<L"max_k"<<m_k;
//
//	fs<<L"}";
//}
//
//ml_nn_layer* ml_nn_1d_maxout_linked_layer::from_file(const ml_file_node& node) {
//	if (wstring(node[L"layer_type"]) != ml_nn_1d_maxout_linked_layer::get_layer_type()) {
//		return NULL;
//	}
//
//	ml_nn_1d_maxout_linked_layer* layer = new ml_nn_1d_maxout_linked_layer(node.name());
//	if (!node[L"input_layer"].empty()) {
//		layer->m_input = new ml_nn_2d_data_layer(node[L"input_layer"]);
//	} else {
//		BASICLOGERROR_MESSAGE(L"linked layer must have input data layer");
//	}
//
//	if (!node[L"output_layer"].empty()) {
//		layer->m_output = new ml_nn_2d_data_layer(node[L"output_layer"]);
//	} else {
//		BASICLOGERROR_MESSAGE(L"linked layer must have output data layer");
//	}
//
//	if (!node[L"max_k"].empty()) {
//		layer->m_k = node[L"max_k"];
//	} else {
//		BASICLOGERROR_MESSAGE(L"1d_maxout_linked layer must be set its k");
//	}
//	
//	return layer;
//}
//
//void ml_nn_2d_maxout_linked_layer::to_file(ml_file_storage& fs, bool write_learned_params) const {
//	fs<<L"{";
//	fs<<L"layer_type"<<get_layer_type();
//	fs<<L"input_layer"<<m_input->get_unique_name();
//	fs<<L"output_layer"<<m_output->get_unique_name();
//	
//	fs<<L"max_k"<<m_k;
//
//	fs<<L"}";
//}
//
//ml_nn_layer* ml_nn_2d_maxout_linked_layer::from_file(const ml_file_node& node) {
//	if (wstring(node[L"layer_type"]) != ml_nn_2d_maxout_linked_layer::get_layer_type()) {
//		return NULL;
//	}
//
//	ml_nn_2d_maxout_linked_layer* layer = new ml_nn_2d_maxout_linked_layer(node.name());
//	if (!node[L"input_layer"].empty()) {
//		layer->m_input = new ml_nn_2d_data_layer(node[L"input_layer"]);
//	} else {
//		BASICLOGERROR_MESSAGE(L"linked layer must have input data layer");
//	}
//
//	if (!node[L"output_layer"].empty()) {
//		layer->m_output = new ml_nn_2d_data_layer(node[L"output_layer"]);
//	} else {
//		BASICLOGERROR_MESSAGE(L"linked layer must have output data layer");
//	}
//
//	if (!node[L"max_k"].empty()) {
//		layer->m_k = node[L"max_k"];
//	} else {
//		BASICLOGERROR_MESSAGE(L"1d_maxout_linked layer must be set its k");
//	}
//
//	return layer;
//}
//
//void ml_nn_combination_linked_layer::to_file(ml_file_storage& fs, bool write_learned_params) const {
//	fs<<L"{";
//	fs<<L"layer_type"<<get_layer_type();
//	fs<<L"input_layer"<<m_input->get_unique_name();
//	fs<<L"output_layer"<<m_output->get_unique_name();
//
//	fs<<L"}";
//}
//
//ml_nn_layer* ml_nn_combination_linked_layer::from_file(const ml_file_node& node) {
//	if (wstring(node[L"layer_type"]) != ml_nn_combination_linked_layer::get_layer_type()) {
//		return NULL;
//	}
//
//	ml_nn_combination_linked_layer* layer = new ml_nn_combination_linked_layer(node.name());
//	if (!node[L"input_layer"].empty()) {
//		layer->m_input = new ml_nn_1d_data_layer(node[L"input_layer"]);
//	} else {
//		BASICLOGERROR_MESSAGE(L"linked layer must have input data layer");
//	}
//
//	if (!node[L"output_layer"].empty()) {
//		layer->m_output = new ml_nn_1d_data_layer(node[L"output_layer"]);
//	} else {
//		BASICLOGERROR_MESSAGE(L"linked layer must have output data layer");
//	}
//
//	return layer;
//}
//
//void ml_nn_output_data_layer::to_file(ml_file_storage& fs, bool write_learned_params) const {
//	fs<<L"{";
//	fs<<L"layer_type"<<get_layer_type();
//	fs<<L"unit_number"<<m_unit_number;
//	fs<<L"activate_type"<<ml_activate_funcs[m_activate_func_type];
//	fs<<L"activate_params"<<m_activate_params;
//	fs<<L"loss_type"<<ml_loss_funcs[m_loss_func_type];
//	fs<<L"task_weight"<<m_task_weight;
//	fs<<L"is_classifier"<<m_is_classifier;
//
//	fs<<L"}";
//}
//
//ml_nn_layer* ml_nn_output_data_layer::from_file(const ml_file_node& node) {
//	if (wstring(node[L"layer_type"]) != ml_nn_output_data_layer::get_layer_type()) {
//		return NULL;
//	}
//
//	ml_nn_output_data_layer* layer = new ml_nn_output_data_layer(node.name());
//
//	if (!node[L"unit_number"].empty()) {
//		layer->m_unit_number = node[L"unit_number"];
//	} else {
//		BASICLOGERROR_MESSAGE(L"output layer must be set its unit number");
//	}
//
//	if (!node[L"activate_type"].empty()) {
//		layer->m_activate_func_type = ml_define::activate_func_index(node[L"activate_type"]);
//	} else {
//		BASICLOGERROR_MESSAGE(L"output layer must be set its activate function type");
//	}
//	
//	if (!node[L"activate_params"].empty()) {
//		layer->m_activate_params = node[L"activate_params"];
//	}
//
//	if (!node[L"loss_type"].empty()) {
//		layer->m_loss_func_type = ml_define::loss_func_index(node[L"loss_type"]);
//	} else {
//		BASICLOGERROR_MESSAGE(L"output layer must be set its loss function type");
//	}
//
//	if (!node[L"task_weight"].empty()) {
//		layer->m_task_weight = node[L"task_weight"];
//	}
//	
//	if (!node[L"is_classifier"].empty()) {
//		layer->m_is_classifier = node[L"is_classifier"];
//	} else {
//		BASICLOGERROR_MESSAGE(L"output layer must be set its classifier type");
//	}
//
//	return layer;
//}