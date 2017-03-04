#include "stdafx.h"

#include "ml_mnist.h"
#include <fstream>
using namespace std;

void ml_mnist::read(ml_data& training_data, ml_data& validation_data, const wstring& mnist_dir, i32 depth_type /* = mt_F64 */) {
	mt_mat training_feature;
	mt_mat training_response;
	mt_mat validation_feature; 
	mt_mat validation_response;

	read(training_feature, training_response, validation_feature, validation_response, mnist_dir,depth_type);

	training_data.add(ml_data_element(training_feature, L"feature_input", ml_Data_Type_Numeric));
	training_data.add(ml_data_element(training_response, L"response_output", ml_Data_Type_Discrete));
	validation_data.add(ml_data_element(validation_feature, L"feature_input", ml_Data_Type_Numeric));
	validation_data.add(ml_data_element(validation_response, L"response_output", ml_Data_Type_Discrete));
}

void ml_mnist::read(mt_mat& training_feature, mt_mat& training_response, mt_mat& validation_feature, mt_mat& validation_response, const wstring& mnist_dir, i32 depth_type /* = mt_F64 */) {
	training_feature = read_feature(mnist_dir + L"train-images.idx3-ubyte", depth_type);
	training_response = read_response(mnist_dir + L"train-labels.idx1-ubyte", depth_type);
	validation_feature = read_feature(mnist_dir + L"t10k-images.idx3-ubyte", depth_type);
	validation_response = read_response(mnist_dir + L"t10k-labels.idx1-ubyte", depth_type);

	basiclog_info2(sys_strcombine()<<L"read training_feature "<<training_feature.size()[0]<<L" * "<<training_feature.size()[1]);
	basiclog_info2(sys_strcombine()<<L"read training_response "<<training_response.size()[0]<<L" * "<<training_response.size()[1]);
	basiclog_info2(sys_strcombine()<<L"read validation_feature "<<validation_feature.size()[0]<<L" * "<<validation_feature.size()[1]);
	basiclog_info2(sys_strcombine()<<L"read validation_response "<<validation_response.size()[0]<<L" * "<<validation_response.size()[1]);
}

mt_mat ml_mnist::read_feature(const wstring& path, i32 depth_type /* = mt_F64 */) {
	ifstream file(basicsys_path_string(path), ios::binary);
	if (file.is_open()){
		int magic_number = 0;
		int number_of_images = 0;
		int n_rows = 0;
		int n_cols = 0;
		file.read((char*) &magic_number, sizeof(magic_number));
		magic_number = mt_helper::reverse_i32(magic_number);
		file.read((char*) &number_of_images,sizeof(number_of_images));
		number_of_images = mt_helper::reverse_i32(number_of_images);
		file.read((char*) &n_rows, sizeof(n_rows));
		n_rows = mt_helper::reverse_i32(n_rows);
		file.read((char*) &n_cols, sizeof(n_cols));
		n_cols = mt_helper::reverse_i32(n_cols);
		int margin = 2;

		mt_mat res(number_of_images, n_rows * n_cols, depth_type);

		for(int i = 0; i < number_of_images; ++i){		
			u8* ptr_data = res.ptr<u8>(i, 0);

			for(int r = 0; r < n_rows; ++r){
				for(int c = 0; c < n_cols; ++c){
					i8 temp = 0;
					file.read((i8*) &temp, sizeof(temp));

					if (depth_type == mt_F64) {
						f64* ptr_temp = (f64*)ptr_data;
						*ptr_temp = temp;
					} else {
						f32* ptr_temp = (f32*)ptr_data;
						*ptr_temp = temp;
					}

					ptr_data += res.step()[1];
				}
			}
		}

		return res;
	} else {
		basiclog_error2(sys_strcombine()<<L"Open "<<path<<L" failed! Please check the file path!");
		return mt_mat();
	}
}

mt_mat ml_mnist::read_response(const wstring& path, i32 depth_type /* = mt_F64 */) {
	ifstream file(basicsys_path_string(path), ios::binary);
	if (file.is_open()){
		int magic_number = 0;
		int number_of_images = 0;
		int n_rows = 0;
		int n_cols = 0;
		file.read((i8*) &magic_number, sizeof(magic_number));
		magic_number = mt_helper::reverse_i32(magic_number);
		file.read((i8*) &number_of_images,sizeof(number_of_images));
		number_of_images = mt_helper::reverse_i32(number_of_images);
		mt_mat res(number_of_images, 1, depth_type);

		for(int i = 0; i < number_of_images; ++i){
			u8 temp = 0;
			file.read((i8*) &temp, sizeof(temp));
			res.set(temp, i, 0);
		}
	
		return res;
	} else {
		basiclog_error2(sys_strcombine()<<L"Open "<<path<<L" failed! Please check the file path!");
		return mt_mat();
	}
}