#pragma once


namespace basicml {

	class ml_mnist {
	public:

		static void read(ml_data& training_data,
			ml_data& validation_data,
			const wstring& mnist_dir, 
			i32 depth_type = mt_F64);

		static void read(mt_mat& train_feature, 
			mt_mat& train_response, 
			mt_mat& validation_feature, 
			mt_mat& validation_response, 
			const wstring& mnist_dir,
			i32 depth_type = mt_F64);

		static void save_as_image(const wstring& save_dir, 
			const wstring& mnist_dir,
			i32 depth_type = mt_F64);

	private:

		static mt_mat read_feature(const wstring& path, i32 depth_type = mt_F64);
		static mt_mat read_response(const wstring& path, i32 depth_type = mt_F64);
	}; 

}