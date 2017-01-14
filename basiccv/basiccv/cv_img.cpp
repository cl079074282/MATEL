#include "stdafx.h"
#include "cv_img.h"



mt_mat cv_img::load(const wstring& path, Load_Type type) {
	Mat cv_m = imread(basicsys_path_string(path), CV_LOAD_IMAGE_UNCHANGED);

	return from_opencv(imread(basicsys_path_string(path), (i32)type));
}

void cv_img::save(const wstring& path, const mt_mat& mat) {
	imwrite(basicsys_path_string(path), from_basiccv(mat, sys_false));
}

mt_mat cv_img::from_opencv(const Mat& mat) {
	if (mat.empty()) {
		return mt_mat();
	}

	i32 channel = mat.channels();

	i32 depth = mt_U8;

	switch (mat.depth()) {
	case CV_8U:
		depth = mt_U8;
		break;
	}

	i32 depth_channel = mt_make_depth_channel(depth, channel);

	if (mat.step.p != NULL) {
		return mt_mat(mat.dims, mat.size.p, depth_channel, mat.data, NULL).clone();
	} else {
		basicmath_mat_request_memory(i32, steps, mat.dims);

		for (i32 i = 0; i < mat.dims; ++i) {
			steps[i] = (i32)mat.step.p[i];
		}

		mt_mat res = mt_mat(mat.dims, mat.size.p, depth_channel, mat.data, steps).clone();

		basicmath_mat_release(steps);

		return res;
	}
}

Mat cv_img::from_basiccv(const mt_mat& mat, b8 deep_copy /* = sys_false */) {
	if (mat.empty()) {
		return Mat();
	}

	i32 channel = mat.channel();
	i32 depth = CV_8U;

	switch (mat.depth()) {
	case mt_U8:
		depth = CV_8U;
		break;
	}

	i32 depth_channel = CV_MAKETYPE(depth, channel);

	if (mat.is_step_positive() && mat.is_min_abs_step_equal_element_size()) {
		basicmath_mat_request_memory(u64, steps, mat.dim());

		for (i32 i = 0; i < mat.dim(); ++i) {
			steps[i] = (u64)mat.step()[i];
		}

		Mat res(mat.dim(), mat.size(), depth_channel, (void*)mat.data(), steps);

		if (deep_copy) {
			res = res.clone();
		}

		basicmath_mat_release(steps);

		return res;
	} else {
		basiclog_warning(basiclog_performance_warning, L"this will reduce the performance, please input the mat with positive steps and min abs step to be the channel size!");
		return cv_img::from_basiccv(mat.clone(), deep_copy);
	}
}

void cv_img::resize(basicmath::mt_mat& dst, const basicmath::mt_mat& src, const mt_size& dst_size, Inter_Type type /* = Inter_Type_Cubic */) {
	if (dst.get_auto_derivative() != NULL) {
		basiclog_warning2(L"dst mat join the auto derivative, resize may casue unexpectation result!");
	}

	if (&dst == &src) {
		mt_mat temp;
		resize(temp, src, dst_size, type);
		dst = temp;
	} else {
		dst.create(dst_size.m_height, dst_size.m_width, src.depth_channel());
		Mat cv_dst_mat = from_basiccv(dst, sys_false);
		Mat cv_src_mat = from_basiccv(src, sys_false);

		cv::resize(cv_src_mat, cv_dst_mat, Size(dst_size.m_width, dst_size.m_height), 0, 0, (i32)type);
	}
}