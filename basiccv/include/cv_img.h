#pragma once



namespace basiccv {

	class cv_img {
	public:

		enum Load_Type {
			Load_Unchanged = -1,
			Load_Grayscale = 0,
			Load_Color = 1,
		};

		static basicmath::mt_mat load(const wstring& path, Load_Type type = Load_Unchanged);
		static void save(const wstring& path, const basicmath::mt_mat& mat);

		enum Inter_Type {
			Inter_Type_Nearest = 0,
			Inter_Type_Linear = 1,
			Inter_Type_Cubic = 2,
		};

		static basicmath::mt_mat resize(const basicmath::mt_mat& src, const mt_size& dst_size, Inter_Type type = Inter_Type_Cubic);

		static basicmath::mt_mat from_opencv(const cv::Mat& mat);

		static cv::Mat from_basiccv(const basicmath::mt_mat& mat, b8 deep_copy = sys_false);
	};

}