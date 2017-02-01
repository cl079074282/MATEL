#include "stdafx.h"
#include "mt_mat.h"
#include "mt_auto_derivative.h"
#include "mt_mat_helper.h"

namespace basicmath {

	class private_math_operation {
	public:

		template<class T>
		static mt_mat pooling(mt_mat& mask_mat, const mt_mat& src, mt_Pooling_Type pooling_type, i32 size, const basicsys::i32* kernel_sizes, const basicsys::i32* strides) {
			basiclog_assert2(src.dim() == size);

			basicmath_mat_request_memory(i32, res_sizes, src.dim());
			basicmath_mat_request_memory(i32, src_start_indexs, src.dim());
			basicmath_mat_request_memory(f64, temp_values, src.dim());
			vector<i32> src_iter_indexs(src.dim());

			for (int i = 0; i < src.dim(); ++i) {
				res_sizes[i] = mt_mat_helper::get_pooling_result_size(src.size()[i], kernel_sizes[i], strides[i]);
			}

			mt_mat res(src.dim(), res_sizes, src.depth_channel(), 0);

			if (pooling_type == mt_Pooling_Type_Max || pooling_type == mt_Pooling_Type_Min) {
				mask_mat = mt_mat(src.dim(), res_sizes, mt_make_depth_channel(mt_S32, src.channel()), -1);
			}

			mt_array_element_iterator res_iter(res);
			mt_array_index_iterator kernel_iter;

			for (;;) {
				T* ptr_data = (T*)res_iter.data();

				if (ptr_data == NULL) {
					break;
				}

				const vector<i32>& cur_indexs = res_iter.position();

				for (int i = 0; i < src.dim(); ++i) {
					src_start_indexs[i] = cur_indexs[i] * strides[i];
				}

				if (pooling_type == mt_Pooling_Type_First_Value) {
					const T* ptr_src_data = src.ptr<T>(src.dim(), src_start_indexs);

					for (i32 c = 0; c < src.channel(); ++c) {
						ptr_data[c] = ptr_src_data[c];
					}
				} else if (pooling_type == mt_Pooling_Type_Mean || pooling_type == mt_Pooling_Type_Sum) {
					kernel_iter.init(src.dim(), kernel_sizes);
					memset(temp_values, 0, sizeof(f64) * src.dim());
					int in_src_number = 0;

					while (kernel_iter.next()) {
						for (int i = 0; i < src.dim(); ++i) {
							src_iter_indexs[i] = src_start_indexs[i] + kernel_iter.position()[i];
						}

						
						if (src.valid_index(src_iter_indexs)) {
							const T* ptr_src_data = src.ptr<T>(src_iter_indexs);
							++in_src_number;

							for (i32 c = 0; c < src.channel(); ++c) {
								temp_values[c] += ptr_src_data[c];
							}
						}
					}

					if (pooling_type == mt_Pooling_Type_Mean) {
						for (i32 c = 0; c < src.channel(); ++c) {
							temp_values[c] /= in_src_number;
						}
					}

					for (i32 c = 0; c < src.channel(); ++c) {
						ptr_data[c] = (T)temp_values[c];
					}

				} else if (pooling_type == mt_Pooling_Type_Max || pooling_type == mt_Pooling_Type_Min) {
					i32* ptr_mask_data = mask_mat.ptr<i32>(cur_indexs);

					kernel_iter.init(src.dim(), kernel_sizes);
					int in_src_number = 0;

					while (kernel_iter.next()) {
						for (int i = 0; i < src.dim(); ++i) {
							src_iter_indexs[i] = src_start_indexs[i] + kernel_iter.position()[i];
						}

						if (src.valid_index(src_iter_indexs)) {
							const T* ptr_src_data = src.ptr<T>(src_iter_indexs);
							++in_src_number;
							i32 mask_index = mt_helper::index_from_multi_index(src_iter_indexs, src.dim(), src.size());

							if (in_src_number == 1) {
								for (i32 c = 0; c < src.channel(); ++c) {
									ptr_data[c] = ptr_src_data[c];
									ptr_mask_data[c] = mask_index;
								}
							} else {
								if (pooling_type == mt_Pooling_Type_Max) {
									for (i32 c = 0; c < src.channel(); ++c) {
										if (ptr_src_data[c] > ptr_data[c]) {
											ptr_data[c] = ptr_src_data[c];
											ptr_mask_data[c] = mask_index;
										}
									}
								} else {
									for (i32 c = 0; c < src.channel(); ++c) {
										if (ptr_src_data[c] < ptr_data[c]) {
											ptr_data[c] = ptr_src_data[c];
											ptr_mask_data[c] = mask_index;
										}
									}
								}
							}
						}
					}
				}
			}

			basicmath_mat_release(res_sizes);
			basicmath_mat_release(src_start_indexs);
			basicmath_mat_release(temp_values);

			return res;
		}

		template<class T>
		static mt_mat unpooling(const int* src_size, const mt_mat& mask_mat, const mt_mat& pooling_res_mat, mt_Pooling_Type pooling_type, i32 size, const int* kernel_sizes, const int* strides) {
			basiclog_assert2(pooling_res_mat.dim() == size);

			basicmath_mat_request_memory(i32, src_start_indexs, pooling_res_mat.dim());
			vector<i32> src_iter_indexs(pooling_res_mat.dim());

			mt_mat src(pooling_res_mat.dim(), src_size, pooling_res_mat.depth_channel(), 0);

			mt_array_element_const_iterator res_iter(pooling_res_mat);
			mt_array_index_iterator kernel_iter;

			for (;;) {
				const T* ptr_data = (const T*)res_iter.data();

				if (ptr_data == NULL) {
					break;
				}

				const vector<i32>& cur_indexs = res_iter.position();

				for (int i = 0; i < src.dim(); ++i) {
					src_start_indexs[i] = cur_indexs[i] * strides[i];
				}

				if (pooling_type == mt_Pooling_Type_First_Value) {
					T* ptr_src_data = src.ptr<T>(src.dim(), src_start_indexs);

					for (i32 c = 0; c < src.channel(); ++c) {
						ptr_src_data[c] += ptr_data[c];
					}
				} else if (pooling_type == mt_Pooling_Type_Sum) {
					kernel_iter.init(src.dim(), kernel_sizes);

					while (kernel_iter.next()) {
						for (int i = 0; i < src.dim(); ++i) {
							src_iter_indexs[i] = src_start_indexs[i] + kernel_iter.position()[i];
						}

						if (src.valid_index(src_iter_indexs)) {
							T* ptr_src_data = src.ptr<T>(src_iter_indexs);

							for (i32 c = 0; c < src.channel(); ++c) {
								ptr_src_data[c] += ptr_data[c];
							}
						}
					}
				} else if (pooling_type == mt_Pooling_Type_Mean) {
					kernel_iter.init(src.dim(), kernel_sizes);
					int in_src_number = 0;

					while (kernel_iter.next()) {
						for (int i = 0; i < src.dim(); ++i) {
							src_iter_indexs[i] = src_start_indexs[i] + kernel_iter.position()[i];
						}

						if (src.valid_index(src_iter_indexs)) {
							++in_src_number;
						}
					}

					kernel_iter.init(src.dim(), kernel_sizes);

					while (kernel_iter.next()) {
						for (int i = 0; i < src.dim(); ++i) {
							src_iter_indexs[i] = src_start_indexs[i] + kernel_iter.position()[i];
						}

						if (src.valid_index(src_iter_indexs)) {
							T* ptr_src_data = src.ptr<T>(src_iter_indexs);

							for (i32 c = 0; c < src.channel(); ++c) {
								ptr_src_data[c] += T(ptr_data[c] / (f64)in_src_number);
							}
						}
					}

				} else if (pooling_type == mt_Pooling_Type_Max || pooling_type == mt_Pooling_Type_Min) {
					const i32* ptr_mask_data = mask_mat.ptr<i32>(cur_indexs);

					mt_helper::multi_index_from_index(src_iter_indexs, *ptr_mask_data, src.dim(), src.size());
					T* ptr_src_data = src.ptr<T>(src_iter_indexs);

					for (i32 c = 0; c < src.channel(); ++c) {
						ptr_src_data[c] += ptr_data[c];
					}
				}
			}

			basicmath_mat_release(src_start_indexs);

			return src;
		}

		template<class T>
		static void sigmoid(mt_mat& res, const mt_mat& src) {
			mt_array_element_const_iterator src_iter(src);
			mt_array_element_iterator res_iter(res);

			for (;;) {
				const T* ptr_src = (const T*)src_iter.data();

				if (ptr_src == NULL) {
					break;
				}

				T* ptr_res = (T*)res_iter.data();

				for (i32 c = 0; c < src.channel(); ++c) {
					ptr_res[c] = (T)(1.0 / (1.0 + std::exp(-ptr_src[c])));
				}
			}
		}

		template<class T>
		static void tanh(mt_mat& res, const mt_mat& src, f64 alpha = 1.7159, f64 belta = 0.666667) {

			mt_array_element_const_iterator src_iter(src);
			mt_array_element_iterator res_iter(res);

			for (;;) {
				const T* ptr_src = (const T*)src_iter.data();

				if (ptr_src == NULL) {
					break;
				}

				T* ptr_res = (T*)res_iter.data();

				for (i32 c = 0; c < src.channel(); ++c) {
					ptr_res[c] = (T)alpha * std::tanh((T)belta * ptr_src[c]);
				}
			}
		}

		template<class T>
		static void relu(mt_mat& res, const mt_mat& src, f64 negative_slope) {

			mt_array_element_const_iterator src_iter(src);
			mt_array_element_iterator res_iter(res);

			for (;;) {
				const T* ptr_src = (const T*)src_iter.data();

				if (ptr_src == NULL) {
					break;
				}

				T* ptr_res = (T*)res_iter.data();

				for (i32 c = 0; c < src.channel(); ++c) {
					if (ptr_src[c] > 0) {
						ptr_res[c] = ptr_src[c];
					} else {
						ptr_res[c] = (T)(ptr_src[c] * negative_slope);
					}
				}
			}
		}

		template<class T>
		static void softmax(mt_mat& res, const mt_mat& src) {
			basiclog_assert2(2 == src.dim());
			
			const u8* ptr_src_dim0 = src.data();
			u8* ptr_dst_dim0 = res.data();

			for (i32 row = 0; row < src.size()[0]; ++row) {
				const T* ptr_src = (T*)ptr_src_dim0;				
				T max_value = *ptr_src++;

				for (int col = 1; col < src.size()[1]; ++col) {
					if (*ptr_src > max_value) {
						max_value = *ptr_src;
					}

					++ptr_src;
				}

				ptr_src = (T*)ptr_src_dim0;
				T* ptr_dst = (T*)ptr_dst_dim0;
				T sum = (T)0.0;

				for (i32 col = 0; col < src.size()[1]; ++col) {
					*ptr_dst = exp(*ptr_src++ - max_value);
					sum += *ptr_dst++;
				}

				ptr_dst = (T*)ptr_dst_dim0;

				for (i32 col = 0; col < src.size()[1]; ++col) {
					*ptr_dst++ /= sum;
				}

				ptr_src_dim0 += src.step()[0];
				ptr_dst_dim0 += res.step()[0];
			}
		}

		template<class T>
		static void activate(mt_mat& res, const mt_mat& src, mt_Activate_Type type, i32 activate_param_size, const f64* activate_params) {
			if (type == mt_Activate_Type_Softmax) {
				basiclog_assert2(src.dim() == 2);
				basiclog_assert2(src.channel() == 1);

				softmax<T>(res, src);
			} else {
				if (type == mt_Activate_Type_Linear) {
					if (&res != &src) {
						res.set(src);
					}
				} else {
					switch (type) {
					case mt_Activate_Type_Relu:
						basiclog_assert2(activate_param_size == 0 || activate_param_size == 1);
						private_math_operation::relu<T>(res, src, activate_param_size == 0 ? 0 : activate_params[0]);
						break;
					case mt_Activate_Type_Sigmoid:
						basiclog_assert2(activate_param_size == 0);
						private_math_operation::sigmoid<T>(res, src);
						break;
					case mt_Activate_Type_Tanh:
						basiclog_assert2(activate_param_size == 0 || activate_param_size == 2);

						if (activate_param_size == 0) {
							private_math_operation::tanh<T>(res, src);
						} else {
							private_math_operation::tanh<T>(res, src, activate_params[0], activate_params[1]);
						}
						
						break;
					default:
						basiclog_unsupport2();
						break;
					}
				}
			}
		}

		template<class T>
		static mt_mat quardratic_loss(const mt_mat& src, const mt_mat& matching_mat) {
			mt_array_element_const_iterator src_iter(src);
			mt_array_element_const_iterator matching_iter(matching_mat);

			mt_mat loss_res(1, 1, src.depth_channel());
			T* ptr_dst = loss_res.ptr<T>(0, i32(0));

			for (;;) {
				const T* ptr_src = (const T*)src_iter.data();

				if (ptr_src == NULL) {
					break;
				}

				const T* ptr_matching = (const T*)matching_mat.data();

				for (i32 c = 0; c < src.dim(); ++c) {
					ptr_dst[c] += (ptr_src[c] - ptr_matching[c]) * (ptr_src[c] - ptr_matching[c]) / 2;
				}
			}

			return loss_res;
		}

		template<class T>
		static mt_mat logarithmic_loss(const mt_mat& src, const mt_mat& matching_mat) {
			basiclog_assert2(src.channel() == 1);
			basiclog_assert2(src.dim() == 2);

			mt_mat loss_res(1, 1, src.depth_channel());
			T* ptr_dst = loss_res.ptr<T>(0, i32(0));

			const u8* ptr_src_dim0 = src.data();
			const u8* ptr_matching_dim0 = matching_mat.data();

			if (1 == src.size()[1]) {
				for (i32 row = 0; row < src.size()[0]; ++row) {
					const T* ptr_src_dim1 = (const T*)ptr_src_dim0;
					const T* ptr_matching_dim1 = (const T*)ptr_matching_dim0;

					if (1 == (i32)*ptr_matching_dim1) {
						*ptr_dst -= (T)log(*ptr_src_dim1 + DBL_EPSILON);
					} else if (0 == (i32)*ptr_matching_dim1){
						*ptr_dst -= (T)log(1 - *ptr_src_dim1 + DBL_EPSILON);
					} else {
						basiclog_assert2(false);
					}

					ptr_src_dim0 += src.step()[0];
					ptr_matching_dim0 += matching_mat.step()[0];
				}
			} else {
				for (i32 row = 0; row < src.size()[0]; ++row) {
					const T* ptr_src_dim1 = (const T*)ptr_src_dim0;
					const T* ptr_matching_dim1 = (const T*)ptr_matching_dim0;

					for (i32 col = 0; col < src.size()[1]; ++col) {
						if (1 == (i32)*ptr_matching_dim1) {
							*ptr_dst -= (T)log(*ptr_src_dim1 + DBL_EPSILON);
							break;
						} else {
							++ptr_matching_dim1;
							++ptr_src_dim1;
						}
					}

					ptr_src_dim0 += src.step()[0];
					ptr_matching_dim0 += matching_mat.step()[0];
				}
			}

			return loss_res;
		}

		template<class T>
		static mt_mat loss(const mt_mat& src, const mt_mat& matching_mat, mt_Loss_Type type) {
			switch (type) {
			case mt_Loss_Type_Quardratic:
				return quardratic_loss<T>(src, matching_mat);
			case mt_Loss_Type_Logarithmic:
				return logarithmic_loss<T>(src, matching_mat);
			default:
				return mt_mat();
			}
		}

		template<class T>
		static void eigen(mt_mat& eigen_value, mt_mat& eigen_vector, const mt_mat& mat) {
			//TODO 
		
		}

		template<class T>
		static mt_mat reduce(const mt_mat& src, mt_mat::Reduce_Type type, i32 reduce_dim) {
			basicmath_mat_request_memory(i32, dst_sizes, src.dim());

			for (i32 i = 0; i < src.dim(); ++i) {
				dst_sizes[i] = src.size()[i];
			}

			dst_sizes[reduce_dim] = 1; 

			mt_mat dst(src.dim(), dst_sizes, src.depth_channel());

			mt_mat mean;

			if (type == mt_mat::Reduce_Type_Standard_Unbias_Variance 
				|| type == mt_mat::Reduce_Type_Standard_Variance 
				|| type == mt_mat::Reduce_Type_Unbias_Variance
				|| type == mt_mat::Reduce_Type_Variance) {
					mean = reduce<T>(src, mt_mat::Reduce_Type_Mean, reduce_dim);
			}
			
			if (type == mt_mat::Reduce_Type_Max || type == mt_mat::Reduce_Type_Min) {
				dst.set(src.sub(0, 1, reduce_dim));
			}

			mt_array_element_const_iterator src_iter(src);

			for (;;) {
				const T* ptr_src = (const T*)src_iter.data();

				if (ptr_src == NULL) {
					break;
				}

				for (i32 i = 0; i < src.dim(); ++i) {
					dst_sizes[i] = src_iter.position()[i];
				}

				dst_sizes[reduce_dim] = 0;

				T* ptr_res = dst.ptr<T>(dst.dim(), dst_sizes);
				T* ptr_mean = NULL;

				if (!mean.empty()) {
					ptr_mean = mean.ptr<T>(mean.dim(), dst_sizes);
				}

				for (i32 c = 0; c < src.channel(); ++c) {
					switch (type) {
					case mt_mat::Reduce_Type_Sum:
					case mt_mat::Reduce_Type_Mean:
						ptr_res[c] += ptr_src[c];
						break;
					case mt_mat::Reduce_Type_Max:
						if (ptr_src[c] > ptr_res[c]) {
							ptr_res[c] = ptr_src[c];
						}

						break;
					case mt_mat::Reduce_Type_Min:
						if (ptr_src[c] < ptr_res[c]) {
							ptr_res[c] = ptr_src[c];
						}

						break;
					case mt_mat::Reduce_Type_Standard_Unbias_Variance:
					case mt_mat::Reduce_Type_Standard_Variance:
					case mt_mat::Reduce_Type_Unbias_Variance:
					case mt_mat::Reduce_Type_Variance:
						ptr_res[c] += (ptr_src[c] - ptr_mean[c]) * (ptr_src[c] - ptr_mean[c]);
						break;
					}
				}

			}

			if (type == mt_mat::Reduce_Type_Mean) {
				dst /= src.size()[reduce_dim];
			} else if (type == mt_mat::Reduce_Type_Variance) {
				dst /= src.size()[reduce_dim];
			} else if (type == mt_mat::Reduce_Type_Standard_Variance) {
				dst /= src.size()[reduce_dim];
				dst.self_pow(0.5);
			} else if (type == mt_mat::Reduce_Type_Unbias_Variance) {
				dst /= src.size()[reduce_dim] - 1;
			} else if (type == mt_mat::Reduce_Type_Standard_Unbias_Variance) {
				dst /= src.size()[reduce_dim] - 1;
				dst.self_pow(0.5);
			}

			basicmath_mat_release(dst_sizes);
		}
	};

	mt_mat operator+(f64 value, const mt_mat& mat) {
		return mat.operator+(value);
	}

	mt_mat operator-(f64 value, const mt_mat& mat) {
		vector<double> vec_other;
		vec_other.resize(mat.channel(), value);
		mt_mat res(mat, mt_mat::Construct_Type_Create_As_Size);

		mt_mat_helper::mat_operation(res, vec_other, mat, mt_mat_helper::Math_Op_Code_Subtract);

		if (mat.auto_derivative() != NULL && mat.auto_derivative()->is_math_operation_recorded()) {
			res.attach(mat.auto_derivative());

			mat.auto_derivative()->add(res, mat, vec_other);
		}

		return res;
	}

	mt_mat operator*(f64 value, const mt_mat& mat) {
		return mat.operator*(value);
	}

	mt_mat operator/(f64 value, const mt_mat& mat) {
		vector<double> vec_other;
		vec_other.resize(mat.channel(), value);
		mt_mat res(mat, mt_mat::Construct_Type_Create_As_Size);

		mt_mat_helper::mat_operation(res, vec_other, mat, mt_mat_helper::Math_Op_Code_Dot_Div);

		if (mat.auto_derivative() != NULL && mat.auto_derivative()->is_math_operation_recorded()) {
			res.attach(mat.auto_derivative());

			mat.auto_derivative()->add(res, mat, vec_other);
		}

		return res;
	}

	mt_mat operator-(const mt_mat& mat) {
		return 0 - mat;
	}
}

mt_mat& mt_mat::operator+=(double other) {
	basiclog_assert2(m_auto_derivative == NULL || !m_auto_derivative->is_math_operation_recorded());

	vector<double> vec_other;
	vec_other.resize(channel(), other);

	return (*this)+=(vec_other);
}

mt_mat& mt_mat::operator+=(const mt_scalar& other) {
	basiclog_assert2(m_auto_derivative == NULL || !m_auto_derivative->is_math_operation_recorded());

	vector<double> vec_other;
	mt_helper::vec_from_scalar(vec_other, other);

	return (*this)+=(vec_other);
}

mt_mat& mt_mat::operator+=(const vector<double>& other) {
	basiclog_assert2(m_auto_derivative == NULL || !m_auto_derivative->is_math_operation_recorded());

	mt_mat_helper::mat_operation(*this, other, *this, mt_mat_helper::Math_Op_Code_Add);

	return *this;
}

mt_mat& mt_mat::operator+=(const mt_mat& other) {
	basiclog_assert2(m_auto_derivative == NULL || !m_auto_derivative->is_math_operation_recorded());

	mt_mat_helper::mat_operation(*this, *this, other, mt_mat_helper::Math_Op_Code_Add);

	return *this;
}

mt_mat& mt_mat::operator-=(double other) {
	basiclog_assert2(m_auto_derivative == NULL || !m_auto_derivative->is_math_operation_recorded());

	vector<double> vec_other;
	vec_other.resize(channel(), other);

	return (*this)-=(vec_other);
}

mt_mat& mt_mat::operator-=(const mt_scalar& other) {
	basiclog_assert2(m_auto_derivative == NULL || !m_auto_derivative->is_math_operation_recorded());

	vector<double> vec_other;
	mt_helper::vec_from_scalar(vec_other, other);

	return (*this)-=(vec_other);
}

mt_mat& mt_mat::operator-=(const vector<double>& other) {
	basiclog_assert2(m_auto_derivative == NULL || !m_auto_derivative->is_math_operation_recorded());

	mt_mat_helper::mat_operation(*this, other, *this, mt_mat_helper::Math_Op_Code_Subtract);

	return *this;
}

mt_mat& mt_mat::operator-=(const mt_mat& other) {
	basiclog_assert2(m_auto_derivative == NULL || !m_auto_derivative->is_math_operation_recorded());

	mt_mat_helper::mat_operation(*this, *this, other, mt_mat_helper::Math_Op_Code_Subtract);

	return *this;
}

mt_mat& mt_mat::operator*=(double other) {
	basiclog_assert2(m_auto_derivative == NULL || !m_auto_derivative->is_math_operation_recorded());

	vector<double> vec_other;
	vec_other.resize(channel(), other);

	return (*this)*=(vec_other);
}

mt_mat& mt_mat::operator*=(const mt_scalar& other) {
	basiclog_assert2(m_auto_derivative == NULL || !m_auto_derivative->is_math_operation_recorded());

	vector<double> vec_other;
	mt_helper::vec_from_scalar(vec_other, other);

	return (*this)*=(vec_other);
}

mt_mat& mt_mat::operator*=(const vector<double>& value) {
	basiclog_assert2(m_auto_derivative == NULL || !m_auto_derivative->is_math_operation_recorded());

	mt_mat_helper::mat_operation(*this, value, *this, mt_mat_helper::Math_Op_Code_Dot_Mul);

	return *this;
}

mt_mat& mt_mat::operator*=(const mt_mat& value) {
	basiclog_assert2(m_auto_derivative == NULL || !m_auto_derivative->is_math_operation_recorded());

	mt_mat_helper::mat_operation(*this, *this, value, mt_mat_helper::Math_Op_Code_Dot_Mul);

	return *this;
}

mt_mat& mt_mat::operator/=(double other) {
	basiclog_assert2(m_auto_derivative == NULL || !m_auto_derivative->is_math_operation_recorded());

	vector<double> vec_other;
	vec_other.resize(channel(), other);

	return (*this)/=(vec_other);
}

mt_mat& mt_mat::operator/=(const mt_scalar& other) {
	basiclog_assert2(m_auto_derivative == NULL || !m_auto_derivative->is_math_operation_recorded());

	vector<double> vec_other;
	mt_helper::vec_from_scalar(vec_other, other);

	return (*this)/=(vec_other);
}

mt_mat& mt_mat::operator/=(const vector<double>& value) {
	basiclog_assert2(m_auto_derivative == NULL || !m_auto_derivative->is_math_operation_recorded());

	mt_mat_helper::mat_operation(*this, value, *this, mt_mat_helper::Math_Op_Code_Dot_Div);

	return *this;
}

mt_mat& mt_mat::operator/=(const mt_mat& value) {
	basiclog_assert2(m_auto_derivative == NULL || !m_auto_derivative->is_math_operation_recorded());

	mt_mat_helper::mat_operation(*this, *this, value, mt_mat_helper::Math_Op_Code_Dot_Div);

	return *this;
}

mt_mat mt_mat::operator+(double value) const {
	vector<double> vec_other;
	vec_other.resize(channel(), value);

	return (*this) + vec_other;
}

mt_mat mt_mat::operator+(const mt_scalar& value) const {
	basiclog_assert2(channel() <= sizeof(value.value) / sizeof(f64));
	
	vector<double> vec_other;
	mt_helper::vec_from_scalar(vec_other, value);

	return (*this) + vec_other;
}

mt_mat mt_mat::operator+(const vector<double>& value) const {
	mt_mat res(*this, mt_mat::Construct_Type_Create_As_Size);

	mt_mat_helper::mat_operation(res, value, *this, mt_mat_helper::Math_Op_Code_Add);

	if (m_auto_derivative != NULL && m_auto_derivative->is_math_operation_recorded()) {
		res.m_auto_derivative = m_auto_derivative;

		m_auto_derivative->add(res, *this, value);
	}

	return res;
}



mt_mat mt_mat::operator+(const mt_mat& value) const {
	mt_mat res(*this, mt_mat::Construct_Type_Create_As_Size);

	mt_mat_helper::mat_operation(res, *this, value, mt_mat_helper::Math_Op_Code_Add);

	if (m_auto_derivative != NULL && m_auto_derivative->is_math_operation_recorded()) {
		res.m_auto_derivative = m_auto_derivative;
		value.m_auto_derivative = m_auto_derivative;

		m_auto_derivative->add(res, *this, value);
	}

	return res;
}

mt_mat mt_mat::operator-(double value) const {
	vector<double> vec_other;
	vec_other.resize(channel(), value);

	return (*this) - vec_other;
}

mt_mat mt_mat::operator-(const mt_scalar& value) const {
	basiclog_assert2(channel() <= sizeof(value.value) / sizeof(f64));

	vector<double> vec_other;
	mt_helper::vec_from_scalar(vec_other, value);

	return (*this) - vec_other;
}

mt_mat mt_mat::operator-(const vector<double>& value) const {
	mt_mat res(*this, mt_mat::Construct_Type_Create_As_Size);

	mt_mat_helper::mat_operation(res, value, *this, mt_mat_helper::Math_Op_Code_Subtract);

	if (m_auto_derivative != NULL && m_auto_derivative->is_math_operation_recorded()) {
		res.m_auto_derivative = m_auto_derivative;

		m_auto_derivative->add(res, *this, value);
	}

	return res;
}





mt_mat mt_mat::operator-(const mt_mat& value) const {
	mt_mat res(*this, mt_mat::Construct_Type_Create_As_Size);
	
	mt_mat_helper::mat_operation(res, *this, value, mt_mat_helper::Math_Op_Code_Subtract);

	if (m_auto_derivative != NULL && m_auto_derivative->is_math_operation_recorded()) {
		res.m_auto_derivative = m_auto_derivative;
		value.m_auto_derivative = m_auto_derivative;

		m_auto_derivative->subtract(res, *this, value);
	}
}

mt_mat mt_mat::operator*(double value) const {
	vector<double> vec_other;
	vec_other.resize(channel(), value);

	return (*this) * vec_other;
}

mt_mat mt_mat::operator*(const mt_scalar& value) const {
	basiclog_assert2(channel() <= sizeof(value.value) / sizeof(f64));

	vector<double> vec_other;
	mt_helper::vec_from_scalar(vec_other, value);

	return (*this) * vec_other;
}

mt_mat mt_mat::operator*(const vector<double>& value) const {
	mt_mat res(*this, mt_mat::Construct_Type_Create_As_Size);

	mt_mat_helper::mat_operation(res, value, *this, mt_mat_helper::Math_Op_Code_Dot_Mul);

	if (m_auto_derivative != NULL && m_auto_derivative->is_math_operation_recorded()) {
		res.m_auto_derivative = m_auto_derivative;

		m_auto_derivative->add(res, *this, value);
	}

	return res;
}

mt_mat mt_mat::operator*(const mt_mat& value) const {
	mt_mat res(*this, mt_mat::Construct_Type_Create_As_Size);

	mt_mat_helper::mat_operation(res, *this, value, mt_mat_helper::Math_Op_Code_Dot_Mul);

	if (m_auto_derivative != NULL && m_auto_derivative->is_math_operation_recorded()) {
		res.m_auto_derivative = m_auto_derivative;
		value.m_auto_derivative = m_auto_derivative;

		m_auto_derivative->subtract(res, *this, value);
	}

	return res;
}

mt_mat mt_mat::operator/(double value) const {
	vector<double> vec_other;
	vec_other.resize(channel(), value);

	return (*this) / vec_other;
}

mt_mat mt_mat::operator/(const mt_scalar& value) const {
	basiclog_assert2(channel() <= sizeof(value.value) / sizeof(f64));

	vector<double> vec_other;
	mt_helper::vec_from_scalar(vec_other, value);

	return (*this) / vec_other;
}

mt_mat mt_mat::operator/(const vector<double>& value) const {
	mt_mat res(*this, mt_mat::Construct_Type_Create_As_Size);

	mt_mat_helper::mat_operation(res, value, *this, mt_mat_helper::Math_Op_Code_Dot_Div);

	if (m_auto_derivative != NULL && m_auto_derivative->is_math_operation_recorded()) {
		res.m_auto_derivative = m_auto_derivative;

		m_auto_derivative->add(res, *this, value);
	}

	return res;
}

mt_mat mt_mat::operator/(const mt_mat& value) const {
	mt_mat res(*this, mt_mat::Construct_Type_Create_As_Size);

	mt_mat_helper::mat_operation(res, *this, value, mt_mat_helper::Math_Op_Code_Dot_Div);

	if (m_auto_derivative != NULL && m_auto_derivative->is_math_operation_recorded()) {
		res.m_auto_derivative = m_auto_derivative;
		value.m_auto_derivative = m_auto_derivative;

		m_auto_derivative->subtract(res, *this, value);
	}
}

mt_mat mt_mat::mul(const mt_mat& value) const {
	basiclog_assert2(depth_channel() == value.depth_channel());
	basiclog_assert2(depth() == mt_F32 || depth() == mt_F64);
	basiclog_assert2(channel() == 1);

#if defined BASICMATH_MKL

	if (is_step_positive() 
		&& value.is_step_positive()
		&& is_min_abs_step_equal_element_size()
		&& value.is_min_abs_step_equal_element_size()) {
			i32 row_a = size()[0];
			i32 col_a = size()[1];

			i32 row_b = value.size()[0];
			i32 col_b = value.size()[1];

			basiclog_assert2(col_a == row_b);

			mt_mat res(row_a, col_b, depth_channel());

			const u8* ptr_cur_data = data();
			const u8* ptr_other_data = value.data();
			const u8* ptr_res_data = res.data();

			CBLAS_TRANSPOSE cur_transpose = CblasNoTrans;
			CBLAS_TRANSPOSE other_transpose = CblasNoTrans;

			i32 cur_ld = (step()[0] / element_size());
			i32 other_ld = (value.step()[0] / value.element_size());
			i32 res_ld = (res.step()[0] / res.element_size());

			if (step()[0] < step()[1]) {
				cur_transpose = CblasTrans;
				cur_ld = (step()[1] / element_size());
			}

			if (value.step()[0] < value.step()[1]) {
				other_transpose = CblasTrans;
				other_ld = (value.step()[1] / value.element_size());
			}

			if (mt_F32 == res.depth()) {
				cblas_sgemm(CblasRowMajor, cur_transpose, other_transpose, row_a, col_b, col_a, 1.0f, (const f32*)ptr_cur_data, cur_ld, (const f32*)ptr_other_data, other_ld, 0.0f, (f32*)ptr_res_data, res_ld);
			} else if (mt_F64 == res.depth()) {
				cblas_dgemm(CblasRowMajor, cur_transpose, other_transpose, row_a, col_b, col_a, 1.0, (const f64*)ptr_cur_data, cur_ld, (const f64*)ptr_other_data, other_ld, 0.0, (f64*)ptr_res_data, res_ld);
			}

			return res;
	} else {
		mt_mat temp_cur = *this;
		if (is_step_negative()) {
			basiclog_warning(basiclog_performance_warning, L"the step of current mat has negative values, this will reduce the performance, you should better input a mat with all positive steps!");
			temp_cur = clone();
		} else if (!is_min_abs_step_equal_element_size()) {
			basiclog_warning(basiclog_performance_warning, L"this mat is result of the channel_at() on a mat with more than 1 channel, this will reduce the performance!");
			temp_cur = clone();
		}

		mt_mat temp_value = value;
		if (value.is_step_negative()) {
			basiclog_warning(basiclog_performance_warning, L"the step of other mat has negative values, this will reduce the performance, you should better input a mat with all positive steps!");
			temp_value = value.clone();
		} else if (!value.is_min_abs_step_equal_element_size()) {
			basiclog_warning(basiclog_performance_warning, L"other mat is result of the channel_at() on a mat with more than 1 channel, this will reduce the performance!");
			temp_value = value.clone();
		}

		return temp_cur.mul(temp_value);
	}

#else



#endif
}

mt_mat mt_mat::conv(const mt_mat& kernel, mt_Conv_Boundary_Type boundary_type /* = mt_Conv_Boundary_Type_Valid */, const int* conv_strides /* = NULL */) const {	
	basiclog_assert2(depth_channel() == kernel.depth_channel());
	basiclog_assert2(depth() == mt_F32 || depth() == mt_F64);
	basiclog_assert2(channel() == 1);

	basiclog_assert2(kernel.dim() <= dim());

	if (mt_Conv_Boundary_Type_Valid == boundary_type) {
		int min_dims = min(kernel.dim(), dim());

		for (int i = 0; i < min_dims; ++i) {
			basiclog_assert2(size()[dim() - 1 - i] >= kernel.size()[kernel.dim() - 1 - i]);
		}
	}

	int mode = mt_mat_helper::get_mkl_conv_calculate_type(*this, kernel);

	basicmath_mat_request_memory(i32, full_sizes, dim());
	basicmath_mat_request_memory(i32, sizes, dim());
	basicmath_mat_request_memory(i32, starts, dim());
	basicmath_mat_request_memory(i32, temp_strides, dim());

	for (int i = 0; i < dim(); ++i) {
		full_sizes[i] = size()[i] + kernel.size()[i] - 1;
		sizes[i] = full_sizes[i];
		starts[i] = 0;
		temp_strides[i] = 1;
	}

	if (conv_strides != NULL) {
		for (int i = 0; i < dim(); ++i) {
			temp_strides[i] = conv_strides[i];
		}
	}

	if (mt_Conv_Boundary_Type_Full != boundary_type) {
		if (mt_Conv_Boundary_Type_Valid == boundary_type) {
			for (int i = 0; i < dim(); ++i) {
				sizes[i] = size()[i] - kernel.size()[i] + 1;
			}
		} else if (mt_Conv_Boundary_Type_Same == boundary_type){
			for (int i = 0; i < dim(); ++i) {
				sizes[i] = size()[i];
			}
		}

		for (int i = 0; i < dim(); ++i) {
			starts[i] = (full_sizes[i] - sizes[i] + 1) / 2;
		}
	}

	for (int i = 0; i < dim(); ++i) {
		sizes[i] = (sizes[i] - 1) / temp_strides[i] + 1;
	}

	mt_mat res(dim(), sizes, depth_channel());	

	//reuse the full_sizes and sizes memory
	i32* src_steps = full_sizes;
	i32* kernel_steps = sizes;
	basicmath_mat_request_memory(i32, res_steps, dim());

	for (int i = 0; i < dim(); ++i) {
		src_steps[i] = step()[i] / element_size();
		kernel_steps[i] = kernel.step()[i] / element_size();
		res_steps[i] = res.step()[i] / element_size();
	}

	void* task = NULL;

	basiclog_assert2(VSL_STATUS_OK == vslsConvNewTask(&task, mode, dim(), size(), kernel.size(), res.size()));
	basiclog_assert2(VSL_STATUS_OK == vslConvSetDecimation(task, &temp_strides[0]));
	basiclog_assert2(VSL_STATUS_OK == vslConvSetStart(task, &starts[0]));

	if (mt_F32 == depth()) {
		basiclog_assert2(VSL_STATUS_OK == vslsConvExec(task, (float*)memory_data(), &src_steps[0], (float*)kernel.memory_data(), &kernel_steps[0], (float*)res.memory_data(), &res_steps[0]));			
	} else if (mt_F64 == depth()) {
		basiclog_assert2(VSL_STATUS_OK == vsldConvExec(task, (double*)memory_data(), &src_steps[0], (double*)kernel.memory_data(), &kernel_steps[0], (double*)res.memory_data(), &res_steps[0]));			
	} else {
		basiclog_assert2(false);
	}

	vslConvDeleteTask(&task);

	basicmath_mat_release(full_sizes);
	basicmath_mat_release(sizes);
	basicmath_mat_release(starts);
	basicmath_mat_release(temp_strides);
	basicmath_mat_release(res_steps);

	return res;
}

mt_mat mt_mat::pooling(mt_mat& mask_mat, mt_Pooling_Type pooling_type, i32 size, const basicsys::i32* kernel_sizes, const basicsys::i32* strides) const {
	switch (depth()) {
	case mt_U8:
		return private_math_operation::pooling<u8>(mask_mat, *this, pooling_type, size, kernel_sizes, strides);
	case mt_S8:
		return private_math_operation::pooling<i8>(mask_mat, *this, pooling_type, size, kernel_sizes, strides);
	case mt_U16:
		return private_math_operation::pooling<u16>(mask_mat, *this, pooling_type, size, kernel_sizes, strides);
	case mt_S16:
		return private_math_operation::pooling<i16>(mask_mat, *this, pooling_type, size, kernel_sizes, strides);
	case mt_U32:
		return private_math_operation::pooling<u32>(mask_mat, *this, pooling_type, size, kernel_sizes, strides);
	case mt_S32:
		return private_math_operation::pooling<i32>(mask_mat, *this, pooling_type, size, kernel_sizes, strides);
	case mt_U64:
		return private_math_operation::pooling<u64>(mask_mat, *this, pooling_type, size, kernel_sizes, strides);
	case mt_S64:
		return private_math_operation::pooling<i64>(mask_mat, *this, pooling_type, size, kernel_sizes, strides);
	case mt_F32:
		return private_math_operation::pooling<f32>(mask_mat, *this, pooling_type, size, kernel_sizes, strides);
	case mt_F64:
		return private_math_operation::pooling<f64>(mask_mat, *this, pooling_type, size, kernel_sizes, strides);
	default:
		basiclog_unsupport2();
		return mt_mat();
	}


}

mt_mat mt_mat::unpooling(const int* src_size, const mt_mat& mask_mat, mt_Pooling_Type pooling_type, i32 size, const int* kernel_sizes, const int* strides) const {	
	switch (depth()) {
	case mt_U8:
		return private_math_operation::unpooling<u8>(src_size, mask_mat, *this, pooling_type, size, kernel_sizes, strides);
	case mt_S8:
		return private_math_operation::unpooling<u8>(src_size, mask_mat, *this, pooling_type, size, kernel_sizes, strides);
	case mt_U16:
		return private_math_operation::unpooling<u16>(src_size, mask_mat, *this, pooling_type, size, kernel_sizes, strides);
	case mt_S16:
		return private_math_operation::unpooling<i16>(src_size, mask_mat, *this, pooling_type, size, kernel_sizes, strides);
	case mt_U32:
		return private_math_operation::unpooling<u32>(src_size, mask_mat, *this, pooling_type, size, kernel_sizes, strides);
	case mt_S32:
		return private_math_operation::unpooling<i32>(src_size, mask_mat, *this, pooling_type, size, kernel_sizes, strides);
	case mt_U64:
		return private_math_operation::unpooling<u64>(src_size, mask_mat, *this, pooling_type, size, kernel_sizes, strides);
	case mt_S64:
		return private_math_operation::unpooling<i64>(src_size, mask_mat, *this, pooling_type, size, kernel_sizes, strides);
	case mt_F32:
		return private_math_operation::unpooling<f32>(src_size, mask_mat, *this, pooling_type, size, kernel_sizes, strides);
	case mt_F64:
		return private_math_operation::unpooling<f64>(src_size, mask_mat, *this, pooling_type, size, kernel_sizes, strides);
	default:
		basiclog_unsupport2();
		return mt_mat();
	}
}

mt_mat mt_mat::expand(i32 size, const i32* side_sizes_1, const i32* side_sizes_2) const {
	basiclog_assert2(size == dim());

	basicmath_mat_request_memory(i32, new_sizes, dim());
	basicmath_mat_request_memory(mt_range, ranges, dim());

	for (i32 i = 0; i < dim(); ++i) {
		new_sizes[i] = this->size()[i];

		ranges[i].m_start = 0;
		
		if (NULL != side_sizes_1) {
			new_sizes[i] += side_sizes_1[i];
			ranges[i].m_start += side_sizes_1[i];
		}

		if (NULL != side_sizes_2) {
			new_sizes[i] += side_sizes_2[i];
		}

		ranges[i].m_end = ranges[i].m_start + this->size()[i];
	}

	mt_mat res(dim(), new_sizes, depth_channel(), 0);
	res.sub(dim(), ranges).set(*this);

	basicmath_mat_release(new_sizes);

	if (m_auto_derivative != NULL && m_auto_derivative->is_math_operation_recorded()) {
		res.attach(m_auto_derivative);

		vector<mt_range> vec_ranges;
		mt_helper::vec_from_array(vec_ranges, size, ranges);
		m_auto_derivative->expand(res, *this, vec_ranges);
	}
}

mt_mat mt_mat::sub_stride(i32 size, const i32* strides) const {
	basicmath_mat_request_memory(i32, res_sizes, dim());

	for (int i = 0; i < dim(); ++i) {
		res_sizes[i] = mt_mat_helper::get_pooling_result_size(this->size()[i], strides[i], strides[i]);
	}

	mt_mat res = *this;

	for (i32 i  = 0; i < dim(); ++i) {
		res.size()[i] = res_sizes[i];
		res.step()[i] *= strides[i];
	}

	basicmath_mat_release(res_sizes);

	if (m_auto_derivative != NULL && m_auto_derivative->is_math_operation_recorded()) {
		vector<i32> vec_strides;
		mt_helper::vec_from_array(vec_strides, size, strides);
		res.attach(m_auto_derivative);

		m_auto_derivative->sub_stride(res, *this, vec_strides);
	}

	return res;
}

mt_mat mt_mat::activate(mt_Activate_Type type, i32 activate_param_size, const f64* activate_params) const {
	mt_mat res(*this, mt_mat::Construct_Type_Create_As_Size);

	basiclog_assert2(depth() == mt_F32 || depth() == mt_F64);

	if (depth() == mt_F32) {
		private_math_operation::activate<f32>(res, *this, type, activate_param_size, activate_params);
	} else {
		private_math_operation::activate<f64>(res, *this, type, activate_param_size, activate_params);
	}

	if (m_auto_derivative != NULL && m_auto_derivative->is_math_operation_recorded()) {
		vector<f64> vec_activate_params;
		mt_helper::vec_from_array(vec_activate_params, activate_param_size, activate_params);

		res.attach(m_auto_derivative);
		m_auto_derivative->activate(res, *this, type, vec_activate_params);
	}
}

mt_mat& mt_mat::self_activate(mt_Activate_Type type, i32 activate_param_size, const f64* activate_params) {
	on_vaule_changed();
	basiclog_assert2(depth() == mt_F32 || depth() == mt_F64);
	basiclog_assert_message2(type != mt_Activate_Type_Softmax, L"softmax function can not invoke self_activate method!");

	if (depth() == mt_F32) {
		private_math_operation::activate<f32>(*this, *this, type, activate_param_size, activate_params);
	} else {
		private_math_operation::activate<f64>(*this, *this, type, activate_param_size, activate_params);
	}
}

mt_mat mt_mat::loss(const mt_mat& matching_mat, mt_Loss_Type type) const {
	basiclog_assert2(depth() == mt_F32 || depth() == mt_F64);
	basiclog_assert2(dim() == matching_mat.dim());
	basiclog_assert2(is_same_size(matching_mat));

	if (depth() == mt_F32) {
		return private_math_operation::loss<f32>(*this, matching_mat, type);
	} else {
		return private_math_operation::loss<f64>(*this, matching_mat, type);
	}
}

mt_mat& mt_mat::self_pow(f64 number) {
	vector<f64> params;
	params.push_back(number);

	mt_mat_helper::mat_operation(*this, params, *this, mt_mat_helper::Math_Op_Code_Pow);

	return *this;
}

mt_mat& mt_mat::self_exp() {
	on_vaule_changed();
	mt_mat_helper::mat_operation(*this, vector<f64>(), *this, mt_mat_helper::Math_Op_Code_Exp);

	return *this;
}

mt_mat mt_mat::pow(f64 number) const {
	mt_mat res(*this, mt_mat::Construct_Type_Create_As_Size);
	
	vector<f64> params;
	params.push_back(number);

	mt_mat_helper::mat_operation(res, params, *this, mt_mat_helper::Math_Op_Code_Pow);

	if (m_auto_derivative != NULL && m_auto_derivative->is_math_operation_recorded()) {
		res.attach(m_auto_derivative);
		m_auto_derivative->pow(res, *this, number);
	}

	return res;
}

mt_mat mt_mat::exp() const {
	mt_mat res(*this, mt_mat::Construct_Type_Create_As_Size);

	mt_mat_helper::mat_operation(res, vector<f64>(), *this, mt_mat_helper::Math_Op_Code_Exp);

	if (m_auto_derivative != NULL && m_auto_derivative->is_math_operation_recorded()) {
		res.attach(m_auto_derivative);
		m_auto_derivative->exp(res, *this);
	}

	return res;
}

void mt_mat::eigen(mt_mat& eigen_value, mt_mat& eigen_vector) const {
	basiclog_assert2(dim() == 2);
	basiclog_assert2(depth_channel() == mt_F32C1 || depth_channel() == mt_F64C1);

	if (is_min_abs_step_equal_element_size() || is_step_positive() || step()[0] < step()[1]) {
		clone().eigen(eigen_value, eigen_vector);
		return;
	}

	if (depth() == mt_F32) {
		private_math_operation::eigen<f32>(eigen_value, eigen_vector, *this);
	} else {
		private_math_operation::eigen<f64>(eigen_value, eigen_vector, *this);
	}
}

mt_mat mt_mat::reduce(mt_mat::Reduce_Type type, i32 reduce_dim) const {
	if (depth() == mt_F32) {
		return private_math_operation::reduce<f32>(*this, type, reduce_dim);
	} else {
		return private_math_operation::reduce<f64>(*this, type, reduce_dim);
	}
}