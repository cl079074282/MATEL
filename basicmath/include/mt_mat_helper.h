#pragma once

#include "mt_mat.h"

namespace basicmath {

	class mt_mat_helper {
	public:

		enum Math_Op_Code {
			Math_Op_Code_Add,
			Math_Op_Code_Subtract,
			Math_Op_Code_Dot_Mul,
			Math_Op_Code_Dot_Div,
			Math_Op_Code_Pow,
			Math_Op_Code_Exp,
		};

		static wstring depth_str(int depth);
		static i32 depth_i32(const wstring& depth);

		static wstring depth_channel_str(int depth_channel);
		static i32 depth_channel_i32(const wstring& depth_channel);

		template<class T>
		static void data_operation(int code, int channel, u8* res, const u8* a, const u8* b) {
			T* res_t = (T*)res;
			const T* a_t = (const T*)a;
			const T* b_t = (const T*)b;

			for (int c = 0; c < channel; ++c) {
				switch (code) {
				case Math_Op_Code_Add:
					res_t[c] = a_t[c] + b_t[c];

					break;
				}
			}
		}

		template<class T>
		static void data_operation(int code, int channel, u8* res, const u8* a, const vector<double>& b) {
			T* res_t = (T*)res;
			const T* a_t = (const T*)a;

			for (int c = 0; c < channel; ++c) {
				switch (code) {
				case Math_Op_Code_Add:
					res_t[c] = a_t[c] + (T)b[c];

					break;
				}
			}
		}

		static void data_operation(int code, int depth_channel, basicsys::u8* res, const u8* a, const u8* b) {
			int channels = mt_get_channel(depth_channel);
			int depth = mt_get_depth(depth_channel);
			int depth_size = mt_get_depth_size(depth);

			switch (depth) {
			case mt_U8:
				data_operation<u8>(code, channels, res, a, b);
				break;
			case mt_S8:
				data_operation<i8>(code, channels, res, a, b);
				break;
			case mt_U16:
				data_operation<u16>(code, channels, res, a, b);
				break;
			case mt_S16:
				data_operation<i16>(code, channels, res, a, b);
				break;
			case mt_U32:
				data_operation<u32>(code, channels, res, a, b);
				break;
			case mt_S32:
				data_operation<i32>(code, channels, res, a, b);
				break;
			case mt_U64:
				data_operation<u64>(code, channels, res, a, b);
				break;
			case mt_S64:
				data_operation<i64>(code, channels, res, a, b);
				break;
			case mt_F32:
				data_operation<f32>(code, channels, res, a, b);
				break;
			case mt_F64:
				data_operation<f64>(code, channels, res, a, b);
				break;
			default:
				basiclog_unsupport2();
				break;
			}
		}

		static void data_operation(int code, int depth_channel, basicsys::u8* res, const u8* a, const vector<double>& b) {
			int channels = mt_get_channel(depth_channel);
			int depth = mt_get_depth(depth_channel);
			int depth_size = mt_get_depth_size(depth);

			switch (depth) {
			case mt_U8:
				data_operation<u8>(code, channels, res, a, b);
				break;
			case mt_S8:
				data_operation<i8>(code, channels, res, a, b);
				break;
			case mt_U16:
				data_operation<u16>(code, channels, res, a, b);
				break;
			case mt_S16:
				data_operation<i16>(code, channels, res, a, b);
				break;
			case mt_U32:
				data_operation<u32>(code, channels, res, a, b);
				break;
			case mt_S32:
				data_operation<i32>(code, channels, res, a, b);
				break;
			case mt_U64:
				data_operation<u64>(code, channels, res, a, b);
				break;
			case mt_S64:
				data_operation<i64>(code, channels, res, a, b);
				break;
			case mt_F32:
				data_operation<f32>(code, channels, res, a, b);
				break;
			case mt_F64:
				data_operation<f64>(code, channels, res, a, b);
				break;
			default:
				basiclog_unsupport2();
				break;
			}
		}

		template<class T>
		static void iteration_operation(mt_array_element_iterator& res, mt_array_element_const_iterator& a, mt_array_element_const_iterator& b, int channels, Math_Op_Code code) {
			for (;;) {
				const T* ptr_a = (const T*)a.data();
				const T* ptr_b = (const T*)b.data();
				T* ptr_res = (T*)res.data();

				if (ptr_res == NULL) {
					break;
				}

				for (int c = 0; c < channels; ++c) {
					switch (code) {
					case basicmath::mt_mat_helper::Math_Op_Code_Add:
						ptr_res[c] = ptr_a[c] + ptr_b[c];
						break;
					case basicmath::mt_mat_helper::Math_Op_Code_Subtract:
						ptr_res[c] = ptr_a[c] - ptr_b[c];
						break;
					case basicmath::mt_mat_helper::Math_Op_Code_Dot_Mul:
						ptr_res[c] = ptr_a[c] * ptr_b[c];
						break;
					case basicmath::mt_mat_helper::Math_Op_Code_Dot_Div:
						ptr_res[c] = ptr_a[c] / ptr_b[c];
						break;
					case basicmath::mt_mat_helper::Math_Op_Code_Pow:
						basiclog_unsupport2();
						break;
					case basicmath::mt_mat_helper::Math_Op_Code_Exp:
						basiclog_unsupport2();
						break;
					default:
						basiclog_unsupport2();
						break;
					}
				}
			}
		}

		template<class T>
		static void iteration_operation(mt_array_element_iterator& res, const vector<double>& b, mt_array_element_const_iterator& a, int channels, Math_Op_Code code) {
			for (;;) {
				const T* ptr_a = (const T*)a.data();
				T* ptr_res = (T*)res.data();

				if (ptr_res == NULL) {
					break;
				}

				for (int c = 0; c < channels; ++c) {
					switch (code) {
					case basicmath::mt_mat_helper::Math_Op_Code_Add:
						ptr_res[c] = (T)b[c] + ptr_a[c];
						break;
					case basicmath::mt_mat_helper::Math_Op_Code_Subtract:
						ptr_res[c] = (T)b[c] - ptr_a[c];
						break;
					case basicmath::mt_mat_helper::Math_Op_Code_Dot_Mul:
						ptr_res[c] = (T)b[c] * ptr_a[c];
						break;
					case basicmath::mt_mat_helper::Math_Op_Code_Dot_Div:
						ptr_res[c] = (T)b[c] / ptr_a[c];
						break;
					case basicmath::mt_mat_helper::Math_Op_Code_Pow:
						ptr_res[c] = (T)pow(ptr_a[c], b[c]);
						break;
					case basicmath::mt_mat_helper::Math_Op_Code_Exp:
						ptr_res[c] = (T)exp(ptr_a[c]);
						break;
					default:
						basiclog_unsupport2();
						break;
					}
				}
			}
		}

		static void mat_operation(mt_mat& res, const mt_mat& a, const mt_mat& b, Math_Op_Code code) {
			basiclog_assert2(a.is_same_size(b));

			mt_array_element_const_iterator iter_a(a);
			mt_array_element_const_iterator iter_b(b);
			mt_array_element_iterator res_iter(res);

			switch (a.depth()) {
			case mt_U8:
				iteration_operation<u8>(res_iter, iter_a, iter_b, a.channel(), code);
				break;
			case mt_S8:
				iteration_operation<i8>(res_iter, iter_a, iter_b, a.channel(), code);
				break;
			case mt_U16:
				iteration_operation<u16>(res_iter, iter_a, iter_b, a.channel(), code);
				break;
			case mt_S16:
				iteration_operation<i16>(res_iter, iter_a, iter_b, a.channel(), code);
				break;
			case mt_U32:
				iteration_operation<u32>(res_iter, iter_a, iter_b, a.channel(), code);
				break;
			case mt_S32:
				iteration_operation<i32>(res_iter, iter_a, iter_b, a.channel(), code);
				break;
			case mt_U64:
				iteration_operation<u64>(res_iter, iter_a, iter_b, a.channel(), code);
				break;
			case mt_S64:
				iteration_operation<i64>(res_iter, iter_a, iter_b, a.channel(), code);
				break;
			case mt_F32:
				iteration_operation<f32>(res_iter, iter_a, iter_b, a.channel(), code);
				break;
			case mt_F64:
				iteration_operation<f64>(res_iter, iter_a, iter_b, a.channel(), code);
				break;
			default:
				basiclog_unsupport2();
				break;
			}
		}

		static void mat_operation(mt_mat& res, const vector<double>& b, const mt_mat& a, Math_Op_Code code) {
			if (code != Math_Op_Code_Pow && code != Math_Op_Code_Exp) {
				basiclog_assert2(a.channel() == (i32)b.size());
			}
			
			mt_array_element_const_iterator iter_a(a);
			mt_array_element_iterator res_iter(res);

			switch (a.depth()) {
			case mt_U8:
				iteration_operation<u8>(res_iter, b, iter_a, a.channel(), code);
				break;
			case mt_S8:
				iteration_operation<i8>(res_iter, b, iter_a, a.channel(), code);
				break;
			case mt_U16:
				iteration_operation<u16>(res_iter, b, iter_a, a.channel(), code);
				break;
			case mt_S16:
				iteration_operation<i16>(res_iter, b, iter_a, a.channel(), code);
				break;
			case mt_U32:
				iteration_operation<u32>(res_iter, b, iter_a, a.channel(), code);
				break;
			case mt_S32:
				iteration_operation<i32>(res_iter, b, iter_a, a.channel(), code);
				break;
			case mt_U64:
				iteration_operation<u64>(res_iter, b, iter_a, a.channel(), code);
				break;
			case mt_S64:
				iteration_operation<i64>(res_iter, b, iter_a, a.channel(), code);
				break;
			case mt_F32:
				iteration_operation<f32>(res_iter, b, iter_a, a.channel(), code);
				break;
			case mt_F64:
				iteration_operation<f64>(res_iter, b, iter_a, a.channel(), code);
				break;
			default:
				basiclog_unsupport2();
				break;
			}
		}

		static void set_data(u8* ptr_data, int depth_channel, const vector<f64>& values) {
			if (values.empty()) {
				return;
			}

			set_data(ptr_data, depth_channel, &values[0], (i32)values.size());
		}

		/** 
		@param value_size if value_size is 0, all channel will be set to be values[0], otherwise the value_size should not be lower than channels.
		*/
		static void set_data(u8* ptr_data, int depth_channel, const f64* values, int value_size = 0) {
			int channels = mt_get_channel(depth_channel);
			int depth = mt_get_depth(depth_channel);
			int range = channels;

			if (value_size != 0) {
				basiclog_assert2(value_size >= (i32)channels);
				range = min(value_size, channels);
			}

			for (int c = 0; c < channels; ++c) {
				int value_index = value_size == 0 ? 0 : c;

				switch (depth) {
				case mt_U8:

					basiclog_assert2(mt_helper::valid_range<u8>(values[value_index]));

					ptr_data[c] = (u8)values[value_index];
					break;
				case mt_S8:
					{
						basiclog_assert2(mt_helper::valid_range<i8>(values[value_index]));

						i8* temp_ptr_src = (i8*)ptr_data;
						temp_ptr_src[c] = (i8)values[value_index];
					}

					break;
				case mt_U16:
					{
						basiclog_assert2(mt_helper::valid_range<u16>(values[value_index]));

						u16* temp_ptr_src = (u16*)ptr_data;
						temp_ptr_src[c] = (u16)values[value_index];
					}

					break;
				case mt_S16:
					{
						basiclog_assert2(mt_helper::valid_range<i16>(values[value_index]));

						i16* temp_ptr_src = (i16*)ptr_data;
						temp_ptr_src[c] = (i16)values[value_index];
					}

					break;
				case mt_U32:
					{
						basiclog_assert2(mt_helper::valid_range<u32>(values[value_index]));

						u32* temp_ptr_src = (u32*)ptr_data;
						temp_ptr_src[c] = (u32)values[value_index];
					}

					break;
				case mt_S32:
					{
						basiclog_assert2(mt_helper::valid_range<i32>(values[value_index]));

						i32* temp_ptr_src = (i32*)ptr_data;
						temp_ptr_src[c] = (i32)values[value_index];
					}

					break;
				case mt_U64:
					{
						basiclog_assert2(mt_helper::valid_range<u64>(values[value_index]));

						u64* temp_ptr_src = (u64*)ptr_data;
						temp_ptr_src[c] = (u64)values[value_index];
					}

					break;
				case mt_S64:
					{
						basiclog_assert2(mt_helper::valid_range<i64>(values[value_index]));

						i64* temp_ptr_src = (i64*)ptr_data;
						temp_ptr_src[c] = (i64)values[value_index];
					}

					break;
				case mt_F32:
					{
						float* temp_ptr_src = (float*)ptr_data;
						temp_ptr_src[c] = (float)values[value_index];
					}

					break;
				case mt_F64:
					{
						double* temp_ptr_src = (double*)ptr_data;
						temp_ptr_src[c] = values[value_index];
					}

					break;
				default:
					basiclog_unsupport2();
					break;
				}
			}
		}

		static void get_data(vector<double>& values, const u8* ptr_data, int depth_channel) {
			values.resize(mt_get_channel(depth_channel));

			get_data(&values[0], ptr_data, depth_channel);
		}

		static void get_data(basicsys::f64* values, const u8* ptr_data, int depth_channel) {
			int channels = mt_get_channel(depth_channel);
			int depth = mt_get_depth(depth_channel);

			for (int c = 0; c < channels; ++c) {
				switch (depth) {
				case mt_U8:
					values[c] = ptr_data[c];
					break;
				case mt_S8:
					{
						i8* temp_ptr_src = (i8*)ptr_data;
						values[c] = temp_ptr_src[c];
					}

					break;
				case mt_U16:
					{
						u16* temp_ptr_src = (u16*)ptr_data;
						values[c] = temp_ptr_src[c];
					}

					break;
				case mt_S16:
					{
						i16* temp_ptr_src = (i16*)ptr_data;
						values[c] = temp_ptr_src[c];
					}

					break;
				case mt_U32:
					{
						u32* temp_ptr_src = (u32*)ptr_data;
						values[c] = temp_ptr_src[c];
					}

					break;
				case mt_S32:
					{
						i32* temp_ptr_src = (i32*)ptr_data;
						values[c] = temp_ptr_src[c];
					}

					break;
				case mt_U64:
					{
						u64* temp_ptr_src = (u64*)ptr_data;
						values[c] = (f64)temp_ptr_src[c];
					}

					break;
				case mt_S64:
					{
						i64* temp_ptr_src = (i64*)ptr_data;
						values[c] = (f64)temp_ptr_src[c];
					}

					break;
				case mt_F32:
					{
						float* temp_ptr_src = (float*)ptr_data;
						values[c] = temp_ptr_src[c];
					}

					break;
				case mt_F64:
					{
						double* temp_ptr_src = (double*)ptr_data;
						values[c] = temp_ptr_src[c];
					}

					break;
				default:
					basiclog_unsupport2();
					break;
				}
			}
		}

		static void max_out(vector<mt_mat>& res, vector<mt_mat>& max_indexes, const vector<mt_mat>& src, int k);
		static void restore_max_out(vector<mt_mat>& src, int src_number, const vector<mt_mat>& max_res, vector<mt_mat>& max_indexes);

		static int get_mkl_conv_calculate_type(const mt_mat& src, const mt_mat& kernel);

		static int get_conv_result_size(int src_size, int kernel_size, int stride, mt_Conv_Boundary_Type boundary_type);
		static void get_conv_result_size(i32 dims, i32* sizes, const i32* src_sizes, const i32* kernel_sizes, const i32* stride_sizes, mt_Conv_Boundary_Type boundary_type);

		static int get_pooling_result_size(int src_size, int kernel_size, int stride);
		static void get_pooling_result_size(i32 dims, i32* res_sizes, const i32* src_sizes, const i32* kernel_sizes, const i32* stride_sizes);

		static mt_mat add(const vector<mt_mat>& elements);

		static mt_mat dot(const vector<mt_mat>& elements);

		static mt_mat conv(const vector<mt_mat>& srcs, const vector<mt_mat>& ketnels, mt_Conv_Boundary_Type boundary_type = mt_Conv_Boundary_Type_Valid, const int* conv_strides = NULL);

		static mt_mat merge_align_dim(const vector<mt_mat>& elements, i32 dim, b8 can_share_memory = sys_true);
		static mt_mat merge_align_channel(const vector<mt_mat>& channels, b8 can_share_memory = sys_true);

		static void save(const wstring& file_path, const mt_mat& mat, b8 text_file = sys_true);
		static void save(sys_buffer_writer* writer, const mt_mat& mat);
		static mt_mat load(const wstring& file_path, b8 text_file = sys_true);
		static mt_mat load(sys_buffer_reader* reader);



	//static 
	};
}