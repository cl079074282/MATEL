#include "stdafx.h"

#include "ml_helper.h"

namespace basicml {

	class private_ml_helper {
	public:

		template<class T>
		static mt_mat group_category(const mt_mat& response) {
			vector<i32> vec_categories;

			const u8* ptr_response_dim0 = response.data();

			for (i32 row = 0; row < response.size()[0]; ++row) {
				const T* ptr_response_dim1 = (const T*)ptr_response_dim0;

				for (i32 col = 0; col < response.size()[1]; ++col) {
					i32 current_category = (i32)ptr_response_dim1[col];

					int j = 0; 
					for (; j < (int)vec_categories.size(); ++j) {
						if (current_category == vec_categories[j]) {
							break;
						}
					}

					if (j == (int)vec_categories.size()) {
						vec_categories.push_back(current_category);
					}
				}

				ptr_response_dim0 += response.step()[0];
			}

			//small to large
			sort(vec_categories.begin(), vec_categories.end());

			mt_mat label_for_category = mt_mat(1, (i32)vec_categories.size(), response.depth_channel());

			for (int i = 0; i < (int)vec_categories.size(); ++i) {
				label_for_category.at<T>(0, i) = (T)vec_categories[i];
			}
		}

		template<class T>
		static i32 calculate_wrong_sequence_number_from_label(const mt_mat& predict_label, const mt_mat& groundtruth_label) {
			const u8* ptr_predict_dim0 = predict_label.data();
			const u8* ptr_groundtruth_dim0 = groundtruth_label.data();

			i32 different_number = 0;
			b8 sequence_different = sys_false;

			for (i32 row = 0; row < predict_label.size()[0]; ++row) {
				const T* ptr_predict_dim1 = (const T*)ptr_predict_dim0;
				const T* ptr_groundtruth_dim1 = (const T*)ptr_groundtruth_dim0;

				if (mt_helper::is_number((f64)ptr_groundtruth_dim1[0]) && !sequence_different) {
					i32 predict_index = mt_helper::index_of_max_value<T>(groundtruth_label.size()[1], ptr_predict_dim1);
					i32 groundtruth_index = mt_helper::index_of_max_value<T>(groundtruth_label.size()[1], ptr_groundtruth_dim1);

					if (predict_index != groundtruth_index) {
						sequence_different = sys_true;;
					}
				} 
				
				if (mt_helper::is_infinity((f64)ptr_groundtruth_dim1[0])) {

					if (sequence_different) {
						++different_number;
						sequence_different = sys_false;
					}
				}

				ptr_predict_dim0 += predict_label.step()[0];
				ptr_groundtruth_dim0 += groundtruth_label.step()[0];
			}

			return different_number;
		}

		template<class T>
		static i32 calculate_wrong_sequence_number_from_response(const mt_mat& predict_response, const mt_mat& groundtruth_response) {
			const u8* ptr_predict_dim0 = predict_response.data();
			const u8* ptr_groundtruth_dim0 = groundtruth_response.data();

			i32 different_number = 0;
			b8 sequence_different = sys_false;

			for (i32 row = 0; row < predict_response.size()[0]; ++row) {
				const T* ptr_predict_dim1 = (const T*)ptr_predict_dim0;
				const T* ptr_groundtruth_dim1 = (const T*)ptr_groundtruth_dim0;

				if (mt_helper::is_number((f64)ptr_groundtruth_dim1[0]) && !sequence_different) {
					if (mt_helper::compare_value(ptr_predict_dim1[0], ptr_groundtruth_dim1[0]) != 0) {
						++different_number;
					}
				}

				if (mt_helper::is_infinity((f64)ptr_groundtruth_dim1[0])) {
					if (sequence_different) {
						++different_number;
						sequence_different = sys_false;
					}
				}

				ptr_predict_dim0 += predict_response.step()[0];
				ptr_groundtruth_dim0 += groundtruth_response.step()[0];
			}

			return different_number;
		}

		template<class T>
		static i32 calculate_wrong_sample_number_from_label(const mt_mat& predict_label, const mt_mat& groundtruth_label) {
			const u8* ptr_predict_dim0 = predict_label.data();
			const u8* ptr_groundtruth_dim0 = groundtruth_label.data();

			i32 different_number = 0;

			for (i32 row = 0; row < predict_label.size()[0]; ++row) {
				const T* ptr_predict_dim1 = (const T*)ptr_predict_dim0;
				const T* ptr_groundtruth_dim1 = (const T*)ptr_groundtruth_dim0;
				
				if (mt_helper::is_number((f64)ptr_groundtruth_dim1[0])) {
					i32 predict_index = mt_helper::index_of_max_value<T>(groundtruth_label.size()[1], ptr_predict_dim1);
					i32 groundtruth_index = mt_helper::index_of_max_value<T>(groundtruth_label.size()[1], ptr_groundtruth_dim1);

					if (predict_index != groundtruth_index) {
						++different_number;
					}
				}

				ptr_predict_dim0 += predict_label.step()[0];
				ptr_groundtruth_dim0 += groundtruth_label.step()[0];
			}

			return different_number;
		}

		template<class T>
		static i32 calculate_wrong_sample_number_from_response(const mt_mat& predict_response, const mt_mat& groundtruth_response) {
			const u8* ptr_predict_dim0 = predict_response.data();
			const u8* ptr_groundtruth_dim0 = groundtruth_response.data();

			i32 different_number = 0;

			for (i32 row = 0; row < predict_response.size()[0]; ++row) {
				const T* ptr_predict_dim1 = (const T*)ptr_predict_dim0;
				const T* ptr_groundtruth_dim1 = (const T*)ptr_groundtruth_dim0;

				if (mt_helper::is_number((f64)ptr_groundtruth_dim1[0])) {
					if (mt_helper::compare_value(ptr_predict_dim1[0], ptr_groundtruth_dim1[0]) != 0) {
						++different_number;
					}
				}

				ptr_predict_dim0 += predict_response.step()[0];
				ptr_groundtruth_dim0 += groundtruth_response.step()[0];
			}

			return different_number;
		}

		template<class T>
		static i32 vector_label_index(i32 category, const mt_mat& label_for_category) {
			const T* ptr_dim1 = (const T*)label_for_category.data();

			for (i32 col = 0; col < label_for_category.size()[1]; ++col) {
				if ((i32)ptr_dim1[col] == category) {
					return col;
				}
			}

			basiclog_assert2(sys_false);
			return -1;
		}

		template<class T>
		static mt_mat vector_label_from_response(const mt_mat& response, const mt_mat& label_for_category) {
			mt_mat label(response.size()[0], response.size()[1] * label_for_category.size()[1], response.depth_channel());

			const u8* ptr_response_dim0 = response.data();
			u8* ptr_label_dim0 = label.data();

			for (i32 row = 0; row < response.size()[0]; ++row) {
				const T* ptr_response_dim1 = (const T*)ptr_response_dim0;
				T* ptr_label_dim1 = (T*)ptr_label_dim0;

				ptr_label_dim1[vector_label_index<T>((i32)ptr_response_dim1[0], label_for_category)] = T(1);

				ptr_response_dim0 += response.step()[0];
				ptr_label_dim0 += label.step()[0];
			}

			return label;
		}

		template<class T>
		static mt_mat response_from_vector_label(const mt_mat& label, const mt_mat& label_for_category) {
			mt_mat response(label.size()[0], label.size()[1] / label_for_category.size()[1], label.depth_channel());

			u8* ptr_response_dim0 = response.data();
			const u8* ptr_label_dim0 = label.data();
			
			for (i32 row = 0; row < label.size()[0]; ++row) {
				T* ptr_response_dim1 = (T*)ptr_response_dim0;
				const T* ptr_label_dim1 = (const T*)ptr_label_dim0;

				i32 index = mt_helper::index_of_max_value<T>(label_for_category.size()[1], ptr_label_dim1);
				*ptr_response_dim1 = label_for_category.at<T>(0, index, 0);

				ptr_response_dim0 += response.step()[0];
				ptr_label_dim0 += label.step()[0];
			}

			return response;
		}

		template<class T>
		static i32 calculate_valid_sequence_number(const mt_mat& label) {
			i32 valid_number = 0;
			const u8* ptr_label_dim0 = label.data();
			b8 seuqnce_first_sample = sys_true;

			for (i32 row = 0; row < label.size()[0]; ++row) {
				const T* ptr_label_dim1 = (const T*)ptr_label_dim0;
				
				if (seuqnce_first_sample) {
					if (mt_helper::is_number(ptr_label_dim1[0])) {
						++valid_number;
					}

					seuqnce_first_sample = sys_false;
				}

				if (mt_helper::is_infinity(ptr_label_dim1[0])) {
					seuqnce_first_sample = sys_true;
				}

				ptr_label_dim0 += label.step()[0];
			}

			return valid_number;
		}


		template<class T>
		static i32 calculate_valid_sample_number(const mt_mat& label) {
			i32 valid_number = 0;
			const u8* ptr_label_dim0 = label.data();

			for (i32 row = 0; row < label.size()[0]; ++row) {
				const T* ptr_label_dim1 = (const T*)ptr_label_dim0;

				if (mt_helper::is_number(ptr_label_dim1[0])) {
					++valid_number;
				}

				ptr_label_dim0 += label.step()[0];
			}

			return valid_number;
		}
	};

}

mt_mat ml_helper::group_category(const mt_mat& response) {
	basiclog_assert2(response.channel() == 1);

	if (response.depth() == mt_F32) {
		return private_ml_helper::group_category<f32>(response);
	} else if (response.depth() == mt_F64) {
		return private_ml_helper::group_category<f64>(response);
	} else {
		basiclog_unsupport2();
		return mt_mat();
	}
}

mt_mat ml_helper::label_from_response(const mt_mat& response, const mt_mat& label_for_category, const wstring& model_name) {
	if (model_name == ml_neural_network::name()) {
		if (response.depth() == mt_F32) {
			return private_ml_helper::vector_label_from_response<f32>(response, label_for_category);
		} else if (response.depth() == mt_F64) {
			return private_ml_helper::vector_label_from_response<f64>(response, label_for_category);
		} else {
			basiclog_unsupport2();
			return mt_mat();
		}
	}
}

mt_mat ml_helper::response_from_label(const mt_mat& label, const mt_mat& label_for_category, const wstring& model_name) {
	if (model_name == ml_neural_network::name()) {
		if (label.depth() == mt_F32) {
			return private_ml_helper::response_from_vector_label<f32>(label, label_for_category);
		} else if (label.depth() == mt_F64) {
			return private_ml_helper::response_from_vector_label<f64>(label, label_for_category);
		} else {
			basiclog_unsupport2();
			return mt_mat();
		}
	}
}

void ml_helper::label_from_response(map<wstring, mt_mat>& labels, const map<wstring, mt_mat>& responses, const map<wstring, mt_mat>& label_for_categories, const wstring& model_name) {
	for (map<wstring, mt_mat>::const_iterator iter = label_for_categories.begin(); iter != label_for_categories.end(); ++iter) {
		labels[iter->first] = label_from_response(responses.at(iter->first), iter->second, model_name);
	}
}

void ml_helper::response_from_label(map<wstring, mt_mat>& responses, const map<wstring, mt_mat>& labels, const map<wstring, mt_mat>& label_for_categories, const wstring& model_name) {
	for (map<wstring, mt_mat>::const_iterator iter = label_for_categories.begin(); iter != label_for_categories.end(); ++iter) {
		responses[iter->first] = response_from_label(labels.at(iter->first), iter->second, model_name);
	}
}

mt_mat ml_helper::align_sequence_data(const mt_mat& seuqnce_fixed_data, const mt_mat& seuqnce_src_data) {
	vector<mt_range> ranges;
	ml_helper::statistic_data_range(ranges, seuqnce_src_data, sys_true);

	i32 fixed_sequence_length = seuqnce_fixed_data.size()[0] / (i32)ranges.size();

	basicmath_mat_request_memory(i32, dst_sizes, seuqnce_fixed_data.dim());

	dst_sizes[0] = seuqnce_src_data.size()[0];

	for (i32 i = 1; i < seuqnce_fixed_data.dim(); ++i) {
		dst_sizes[i] = seuqnce_fixed_data.size()[1];
	}

	mt_mat dst(seuqnce_fixed_data.dim(), dst_sizes, seuqnce_fixed_data.depth_channel(), mt_scalar(mt_helper::infinity()));

	for (i32 i = 0; i < (i32)ranges.size(); ++i) {
		dst.sub(ranges[i]).set(seuqnce_fixed_data.sub(mt_range(i * fixed_sequence_length, i * fixed_sequence_length + ranges[i].size())));
	}

	return dst;
}

i32 ml_helper::calculate_wrong_sequence_number_from_label(const mt_mat& predict_label, const mt_mat& groundtruth_label) {
	basiclog_assert2(predict_label.is_same_size(groundtruth_label));

	if (predict_label.depth() == mt_F32) {
		return private_ml_helper::calculate_wrong_sequence_number_from_label<f32>(predict_label, groundtruth_label);
	} else if (predict_label.depth() == mt_F64) {
		return private_ml_helper::calculate_wrong_sequence_number_from_label<f64>(predict_label, groundtruth_label);
	} else {
		basiclog_unsupport2();
		return -1;
	}
}

i32 ml_helper::calculate_wrong_sequence_number_from_response(const mt_mat& predict_response, const mt_mat& groundtruth_response) {
	basiclog_assert2(predict_response.is_same_size(groundtruth_response));
	basiclog_assert2(predict_response.size()[1] == 1);

	if (predict_response.depth() == mt_F32) {
		return private_ml_helper::calculate_wrong_sequence_number_from_response<f32>(predict_response, groundtruth_response);
	} else if (predict_response.depth() == mt_F64) {
		return private_ml_helper::calculate_wrong_sequence_number_from_response<f64>(predict_response, groundtruth_response);
	} else {
		basiclog_unsupport2();
		return -1;
	}
}

i32 ml_helper::calculate_wrong_sample_number_from_label(const mt_mat& predict_label, const mt_mat& groundtruth_label) {
	basiclog_assert2(predict_label.is_same_size(groundtruth_label));

	if (predict_label.depth() == mt_F32) {
		return private_ml_helper::calculate_wrong_sample_number_from_label<f32>(predict_label, groundtruth_label);
	} else if (predict_label.depth() == mt_F64) {
		return private_ml_helper::calculate_wrong_sample_number_from_label<f64>(predict_label, groundtruth_label);
	} else {
		basiclog_unsupport2();
		return -1;
	}
}

i32 ml_helper::calculate_wrong_sample_number_from_response(const mt_mat& predict_response, const mt_mat& groundtruth_response) {
	basiclog_assert2(predict_response.is_same_size(groundtruth_response));
	basiclog_assert2(predict_response.size()[1] == 1);

	if (predict_response.depth() == mt_F32) {
		return private_ml_helper::calculate_wrong_sample_number_from_response<f32>(predict_response, groundtruth_response);
	} else if (predict_response.depth() == mt_F64) {
		return private_ml_helper::calculate_wrong_sample_number_from_response<f64>(predict_response, groundtruth_response);
	} else {
		basiclog_unsupport2();
		return -1;
	}
}

i32 ml_helper::calculate_valid_sequence_number(const mt_mat& response_or_label) {
	if (response_or_label.depth() == mt_F32) {
		return private_ml_helper::calculate_valid_sequence_number<f32>(response_or_label);
	} else if (response_or_label.depth() == mt_F64) {
		return private_ml_helper::calculate_valid_sequence_number<f64>(response_or_label);
	} else {
		basiclog_unsupport2();
		return -1;
	}
}

i32 ml_helper::calculate_valid_sample_number(const mt_mat& response_or_label) {
	if (response_or_label.depth() == mt_F32) {
		return private_ml_helper::calculate_valid_sample_number<f32>(response_or_label);
	} else if (response_or_label.depth() == mt_F64) {
		return private_ml_helper::calculate_valid_sample_number<f64>(response_or_label);
	} else {
		basiclog_unsupport2();
		return -1;
	}
}

void ml_helper::write(sys_json_writer& writer, const vector<ml_nn_layer*>& input_layers, const vector<ml_nn_layer*>& hidden_layers, const vector<ml_nn_layer*>& output_layers, const vector<ml_nn_layer*>& linked_layers, b8 write_learned_param /* = sys_true */) {
	writer<<L"{";

	for (i32 i = 0; i < (i32)input_layers.size(); ++i) {
		writer<<input_layers[i]->name();
		input_layers[i]->write(writer, write_learned_param);
	}

	for (i32 i = 0; i < (i32)hidden_layers.size(); ++i) {
		writer<<hidden_layers[i]->name();
		hidden_layers[i]->write(writer, write_learned_param);
	}

	for (i32 i = 0; i < (i32)output_layers.size(); ++i) {
		writer<<output_layers[i]->name();
		output_layers[i]->write(writer, write_learned_param);
	}

	for (i32 i = 0; i < (i32)linked_layers.size(); ++i) {
		writer<<linked_layers[i]->name();
		linked_layers[i]->write(writer, write_learned_param);
	}

	writer<<L"}";
}

void ml_helper::read(vector<ml_nn_layer*>& input_layers, vector<ml_nn_layer*>& hidden_layers, vector<ml_nn_layer*>& output_layers, vector<ml_nn_layer*>& linked_layers, sys_json_reader& reader) {
	sys_json_reader_iteartor iter = reader.begin();

	while (iter != reader.end()) {
		ml_nn_layer* layer = ml_nn_layer::read(reader);

		if (layer->to_input_data_layer() != NULL) {
			input_layers.push_back(layer);
		} else if (layer->to_output_data_layer() != NULL) {
			output_layers.push_back(layer);
		} else if (layer->to_data_layer() != NULL) {
			hidden_layers.push_back(layer);
		} else {
			linked_layers.push_back(layer);
		}
	}

	link_layer(linked_layers, input_layers, hidden_layers, output_layers);
}

void ml_helper::statistic_data_range(vector<vector<mt_range>>& ranges, const mt_mat& data, b8 seuqnce_data) {

}

void ml_helper::statistic_data_range(vector<mt_range>& ranges, const mt_mat& data, b8 sequence_data /* = sys_false */) {

}

void ml_helper::link_layer(vector<ml_nn_layer*>& linked_layers, const vector<ml_nn_layer*>& input_layers, const vector<ml_nn_layer*>& hidden_layers, const vector<ml_nn_layer*>& output_layers) {
	vector<ml_nn_layer*> data_layers = hidden_layers;
	data_layers.insert(data_layers.end(), input_layers.begin(), input_layers.end());
	data_layers.insert(data_layers.end(), output_layers.begin(), output_layers.end());
	
	for (i32 i = 0; i < (i32)linked_layers.size(); ++i) {
		const wstring& input_name = linked_layers[i]->to_linked_layer()->input_name();
		const wstring& output_name = linked_layers[i]->to_linked_layer()->output_name();

		for (i32 j = 0; j < (i32)data_layers.size(); ++j) {
			if (data_layers[j]->name() == input_name) {
				linked_layers[i]->to_linked_layer()->set_input(data_layers[j]->to_data_layer());
			} else if (data_layers[j]->name() == output_name) {
				linked_layers[i]->to_linked_layer()->set_output(data_layers[j]->to_data_layer());
			}
		}
	}
}