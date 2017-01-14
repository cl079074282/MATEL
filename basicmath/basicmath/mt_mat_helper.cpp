#include "stdafx.h"

#include "mt_mat_helper.h"

int mt_mat_helper::get_mkl_conv_calculate_type(const mt_mat& src, const mt_mat& kernel) {
	return VSL_CONV_MODE_DIRECT;
}

wstring mt_mat_helper::depth_str(int depth) {
	switch (depth) {
	case mt_U8:
		return wstring(L"mt_U8");
	case mt_S8:
		return wstring(L"mt_S8");
	case mt_U16:
		return wstring(L"mt_U16");
	case mt_S16:
		return wstring(L"mt_S16");
	case mt_U32:
		return wstring(L"mt_U32");
	case mt_S32:
		return wstring(L"mt_S32");
	case mt_U64:
		return wstring(L"mt_U64");
	case mt_S64:
		return wstring(L"mt_S64");
	case mt_F32:
		return wstring(L"mt_F32");
	case mt_F64:
		return wstring(L"mt_F64");
	case mt_User:
		return wstring(L"mt_User");
	}

	basiclog_assert2(false);
	return wstring(L"mt_Unknow");
}

i32 mt_mat_helper::depth_i32(const wstring& depth) {
	if (depth == L"mt_U8") {
		return mt_U8;
	} else if (depth == L"mt_S8") {
		return mt_S8;
	} else if (depth == L"mt_U16") {
		return mt_U16;
	} else if (depth == L"mt_S16") {
		return mt_S16;
	} else if (depth == L"mt_S32") {
		return mt_S32;
	} else if (depth == L"mt_U32") {
		return mt_U32;
	} else if (depth == L"mt_S64") {
		return mt_S64;
	} else if (depth == L"mt_U64") {
		return mt_U64;
	} else if (depth == L"mt_F32") {
		return mt_F32;
	} else if (depth == L"mt_F64") {
		return mt_F64;
	}

	return mt_User;
}

wstring mt_mat_helper::depth_channel_str(int depth_channel) {
	return sys_strcombine()<<depth_str(mt_get_depth(depth_channel))<<L"C"<<mt_get_channel(depth_channel);
}

i32 mt_mat_helper::depth_channel_i32(const wstring& depth_channel) {
	vector<wstring> elements;
	sys_strhelper::split(elements, depth_channel, L"C");

	i32 channel = 1;
	i32 depth = depth_i32(elements[0]);

	if ((i32)elements.size() > 1) {
		channel = _wtoi(elements[1].c_str());
	}

	return mt_make_depth_channel(depth, channel);
}

mt_mat mt_mat_helper::combine_mat_as_channel(vector<mt_mat>& channels) {
	int dims = channels.front().dim();
	int depth = channels.front().depth();

	i32* sizes = channels.front().size();

	mt_mat res(dims, sizes, mt_make_depth_channel(depth, (i32)channels.size()));

	sys_for(i, channels) {
		res.channel_at(i).set(channels[i]);
	}

	return res;
}

i32 mt_mat_helper::get_conv_result_size(int src_size, int kernel_size, int stride, mt_Conv_Boundary_Type boundary_type) {
	i32 res_size = 0;

	if (boundary_type == mt_Conv_Boundary_Type_Full) {
		res_size = src_size + kernel_size - 1;
	} else if (boundary_type == mt_Conv_Boundary_Type_Same) {
		res_size = src_size;
	} else if (boundary_type == mt_Conv_Boundary_Type_Same) {
		res_size = src_size - kernel_size + 1;
	} else {
		basiclog_unsupport2();
	}

	return (res_size - 1) / stride + 1;
}

void mt_mat_helper::get_conv_result_size(i32 dims, i32* res_sizes, const i32* src_sizes, const i32* kernel_sizes, const i32* stride_sizes, mt_Conv_Boundary_Type boundary_type) {
	for (i32 i = 0; i  < dims; ++i) {
		res_sizes[i] = get_conv_result_size(src_sizes[i], kernel_sizes[i], stride_sizes == NULL ? 1 : stride_sizes[i], boundary_type);
	}
}

int mt_mat_helper::get_pooling_result_size(int src_size, int kernel_size, int stride) {
	int res_size = 0;
	int cur_pos = 0;

	for (;;) {
		if (cur_pos >= src_size) {
			break;
		}

		if (cur_pos + kernel_size >= src_size) {
			++res_size;
			break;
		}

		cur_pos += stride;
		++res_size;	
	}

	return res_size;
}

void mt_mat_helper::get_pooling_result_size(i32 dims, i32* res_sizes, const i32* src_sizes, const i32* kernel_sizes, const i32* stride_sizes) {
	for (i32 i = 0; i  < dims; ++i) {
		res_sizes[i] = get_pooling_result_size(src_sizes[i], kernel_sizes[i], stride_sizes[i]);
	}
}

void mt_mat_helper::save(const wstring& file_path, const mt_mat& mat, b8 text_file) {
	if (text_file) {
		sys_string_file_buffer_writer buffer(file_path);
		save(&buffer, mat);
	} else {
		sys_byte_file_buffer_writer buffer(file_path);
		save(&buffer, mat);
	}
}

void mt_mat_helper::save(sys_buffer_writer* buffer, const mt_mat& mat) {
	sys_json_writer writer(buffer);

	writer<<L"basicmath_mat"<<mat;
}

mt_mat mt_mat_helper::load(const wstring& file_path, b8 text_file /* = sys_true */) {
	if (text_file) {
		sys_string_file_buffer_reader buffer(file_path);
		return load(&buffer);
	} else {
		sys_byte_file_buffer_reader buffer(file_path);
		return load(&buffer);
	}
}

mt_mat mt_mat_helper::load(sys_buffer_reader* buffer) {
	mt_mat res;

	sys_json_reader reader(buffer);

	reader[L"basicmath_mat"]>>res;

	return res;
}

void mt_mat_helper::add(mt_mat& res, const vector<mt_mat>& elements) {
	mt_auto_derivative* auto_derivative = NULL;

	for (i32 i = 0; i < (i32)elements.size(); ++i) {
		if (elements[i].m_auto_derivative != NULL) {
			if (auto_derivative != NULL && auto_derivative != elements[i].m_auto_derivative) {
				basiclog_assert_message2(sys_false, L"all mats must have only one auto_derivative!");
			}

			auto_derivative = elements[i].m_auto_derivative;
		}
	}

	res.set(elements[0]);

	for (i32 i = 1; i < (i32)elements.size(); ++i) {
		mt_mat_helper::mat_operation(res,  res, elements[i], mt_mat_helper::Math_Op_Code_Add);
	}

	if (auto_derivative != NULL) {
		res.attach(auto_derivative);

		for (i32 i = 0; i < (i32)elements.size(); ++i) {
			elements[i].m_auto_derivative = auto_derivative;
		}
	}
}

void mt_mat_helper::dot(mt_mat& res, const vector<mt_mat>& elements) {
	mt_auto_derivative* auto_derivative = NULL;

	for (i32 i = 0; i < (i32)elements.size(); ++i) {
		if (elements[i].m_auto_derivative != NULL) {
			if (auto_derivative != NULL && auto_derivative != elements[i].m_auto_derivative) {
				basiclog_assert_message2(sys_false, L"all mats must have only one auto_derivative!");
			}

			auto_derivative = elements[i].m_auto_derivative;
		}
	}

	res.set(elements[0]);

	for (i32 i = 1; i < (i32)elements.size(); ++i) {
		mt_mat_helper::mat_operation(res,  res, elements[i], mt_mat_helper::Math_Op_Code_Dot_Mul);
	}

	if (auto_derivative != NULL) {
		res.attach(auto_derivative);

		for (i32 i = 0; i < (i32)elements.size(); ++i) {
			elements[i].m_auto_derivative = auto_derivative;
		}
	}
}