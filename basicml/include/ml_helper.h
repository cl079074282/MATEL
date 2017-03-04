#pragma once


namespace basicml {

	class ml_nn_layer;

	class ml_helper {
	public:

		/** Group the categories.
		@return 1 * n mt_mat for categories, the categories in mat will be sorted from small to large.
		*/
		static mt_mat group_category(const mt_mat& response);
		static mt_mat label_from_response(const mt_mat& response, const mt_mat& label_for_category, const wstring& model_name);
		static mt_mat response_from_label(const mt_mat& label, const mt_mat& label_for_category, const wstring& model_name);

		static void label_from_response(map<wstring, mt_mat>& labels, const map<wstring, mt_mat>& responses, const map<wstring, mt_mat>& label_for_categories, const wstring& model_name);
		static void response_from_label(map<wstring, mt_mat>& responses, const map<wstring, mt_mat>& labels, const map<wstring, mt_mat>& label_for_categories, const wstring& model_name);
		
		static mt_mat align_sequence_data(const mt_mat& seuqnce_fixed_data, const mt_mat& seuqnce_src_data);

		static i32 calculate_wrong_sequence_number_from_label(const mt_mat& predict_label, const mt_mat& groundtruth_label);
		static i32 calculate_wrong_sequence_number_from_response(const mt_mat& predict_response, const mt_mat& groundtruth_response);

		static i32 calculate_wrong_sample_number_from_label(const mt_mat& predict_label, const mt_mat& groundtruth_label);
		static i32 calculate_wrong_sample_number_from_response(const mt_mat& predict_response, const mt_mat& groundtruth_response);

		static i32 calculate_valid_sequence_number(const mt_mat& response_or_label);
		static i32 calculate_valid_sample_number(const mt_mat& response_or_label);

		static void write(sys_json_writer& writer, const vector<ml_nn_layer*>& input_layers, const vector<ml_nn_layer*>& hidden_layers, const vector<ml_nn_layer*>& output_layers, const vector<ml_nn_layer*>& linked_layers, b8 write_learned_param = sys_true);
		static void read(vector<ml_nn_layer*>& input_layers, vector<ml_nn_layer*>& hidden_layers, vector<ml_nn_layer*>& output_layers, vector<ml_nn_layer*>& linked_layers, sys_json_reader& reader);

		static void clone_layer(vector<ml_nn_layer*>& dst_input_layers, 
			vector<ml_nn_layer*>& dst_hidden_layers, 
			vector<ml_nn_layer*>& dst_output_layers, 
			vector<ml_nn_layer*>& dst_linked_layers, 
			const vector<ml_nn_layer*>& src_input_layers, 
			const vector<ml_nn_layer*>& src_hidden_layers, 
			const vector<ml_nn_layer*>& src_output_layers, 
			const vector<ml_nn_layer*>& src_linked_layers);

		/** Statistic data range considering the distributed mat and big mat, for each element of ranges, it should be saved in the continuous memory or file path.
		*/
		static void statistic_data_range(vector<vector<mt_range>>& ranges, const mt_mat& data, b8 sequence_data = sys_false);


		static void statistic_data_range(vector<mt_range>& ranges, const mt_mat& data, b8 sequence_data = sys_false);

		static i32 find_in_text(const wstring* texts, i32 size, const wstring& candidate, b8 assert_finded = sys_true);

	protected:

		static void link_layer(vector<ml_nn_layer*>& linked_layers, const vector<ml_nn_layer*>& input_layers, const vector<ml_nn_layer*>& hidden_layers, const vector<ml_nn_layer*>& output_layers);
	};



}