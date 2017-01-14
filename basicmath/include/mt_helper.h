#pragma once

#include "mt_scalar_t.h"

namespace basicmath {

	

	class mt_helper {
	public:

		template<class T>
		static bool is_same_symbol(const T& v1, const T& v2) {
			if ((v1 >= 0 && v2 >= 0)
				|| (v1 <= 0 && v2 <= 0)) {
					return true;
			}

			return false;
		}

		static i8 max_i8();
		static i8 min_i8();

		static u8 max_u8();
		static u8 min_u8();

		template<class T>
		static b8 valid_range(f64 value) {
			if (typeid(T) == typeid(f32) || typeid(T) == typeid(f64)) {
				return sys_true;
			} else {
				if (typeid(T) == typeid(i8)) {
					return value <= max_i8() && value >= min_i8();
				} else if (typeid(T) == typeid(u8)) {
					return value <= max_u8() && value >= min_u8();
				}

				return sys_true;
			}
		}

		static void set_float_eps(float eps);
		static float get_float_eps();
		static void set_double_eps(double eps);
		static double get_double_eps();

		static int compare_float(float a, float b);
		static int compare_double(double a, double b);

		template<class T>
		static int compare_value(const T& a, const T& b) {
			if (typeid(T) == typeid(float)) {
				return compare_float((float)a, (float)b);
			} else if (typeid(T) == typeid(double)) {
				return compare_double((double)a, (double)b);
			} else {
				return a - b;
			}
		}

		template<class T>
		static T compute_min(const T* values, int size) {
			T min_value = values[0];

			for (int i = 1; i < size; ++i) {
				if (values[i] < min_value) {
					min_value = values[i];
				}
			}

			return min_value;
		}

		template<class T>
		static T compute_min_in_abs(const T* values, int size) {
			T min_value = values[0];

			for (int i = 1; i < size; ++i) {
				if (abs(values[i]) < abs(min_value)) {
					min_value = values[i];
				}
			}

			return min_value;
		}

		template<class T>
		static T compute_abs_min(const T* values, int size) {
			T min_value = abs(values[0]);

			for (int i = 1; i < size; ++i) {
				if (abs(values[i]) < min_value) {
					min_value = abs(values[i]);
				}
			}

			return min_value;
		}

		template<class T>
		static void vec_from_scalar(vector<T>& dst, const mt_scalar_t<T>& src) {
			dst.resize(sizeof(src.value) / sizeof(src.value[0]));

			for (i32 i = 0; i < (i32)dst.size(); ++i) {
				dst[i] = src.value[i];
			}
		}

		template<class T>
		static vector<T> vec_from_array(i32 size, const T* data) {
			vector<T> dst;
			vec_from_array(dst, size, data);

			return dst;
		}

		template<class T>
		static void vec_from_array(vector<T>& dst, i32 size, const T* data) {
			dst.resize(size);

			for (i32 i = 0; i < size; ++i) {
				dst[i] = data[i];
			}
		}

		static i32 index_from_multi_index(const vector<i32>& indexs, i32 size, const i32* sizes) {
			basiclog_assert2((i32)indexs.size() == size);

			i32 res = indexs[size - 1];
			i32 cur_dim_element_number = 1;

			for (i32 i = size - 2; i >= 0; --i) {
				cur_dim_element_number *= sizes[i + 1];

				res += indexs[i] * cur_dim_element_number;
			}

			return res;
		}

		static void multi_index_from_index(vector<i32>& indexs, i32 index, i32 size, const i32* sizes) {
			indexs.resize(size);

			i32 dim_element_number = 1;

			for (i32 i = 1; i < size; ++i) {
				dim_element_number *= sizes[i];
			}

			for (i32 i = 0; i < size; ++i) {
				i32 cur_dim_index = index % dim_element_number;
				index = index - cur_dim_index * dim_element_number;
				dim_element_number /= sizes[i + 1];

				indexs[i] = cur_dim_index;
			}
		}

		static i32 factorial(i32 n) {
			int res = 1;
			for (int i = 1; i <= n; ++i) {
				res *= i;
			}

			return res;
		}

		static i32 combination(i32 n, i32 k) {
			basiclog_assert2(n >= k);
			return factorial(n) / (factorial(k) * factorial(n - k));
		}
	};

	template<class T>
	bool operator==(const vector<T>& vec1, const vector<T>& vec2) {
		if (vec1.size() != vec2.size()) {
			return false;
		}

		for (int i = 0; i < (int)vec1.size(); ++i) {
			if (mt_helper::compare_value<T>(vec1[i], vec2[i]) != 0) {
				return false;
			}
		}

		return true;
	}


}