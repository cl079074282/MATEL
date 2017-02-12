#pragma once




namespace basicml {

	static const wstring ml_Data_Description_Feature = L"feature_";	//!< All the input feature descriptions must begin with feature_.
	static const wstring ml_Data_Description_Response = L"response_";	//!< All the response descriptions must begin with response_.
	static const wstring ml_Data_Description_Weight = L"weight";	//!< Weight description.

	class ml_data_element {
	public:

		ml_data_element(const mt_mat& mat, const wstring& description, ml_Data_Type data_type = ml_Data_Type_Numeric, b8 sequence_data = sys_false) {
			m_mat = mat;
			m_description = description;
			m_sequence_data = sequence_data;
			m_data_type = data_type;
		}

		ml_data_element(const mt_mat& mat, const wstring& description, const vector<wstring>& dimension_descriptions, ml_Data_Type data_type = ml_Data_Type_Numeric, b8 sequence_data = sys_false) {
			m_mat = mat;
			m_description = description;
			m_sequence_data = sequence_data;
			m_data_type = data_type;

			m_dimension_descriptions = dimension_descriptions;
		}

		mt_mat m_mat;
		wstring m_description;

		vector<wstring> m_dimension_descriptions;
		ml_Data_Type m_data_type;
		b8 m_sequence_data;
	};

	class ml_data {
	public:

		b8 add(const ml_data_element& element);

		const map<wstring, ml_data_element>& data() const {
			return m_datas;
		}

		virtual mt_mat get(const wstring& description, i32 index = -1) const;

		virtual mt_mat get(const wstring& description, const mt_range& range) const;

		virtual mt_mat get(const wstring& description, const vector<i32>& indexs) const;

		virtual void get(map<wstring, mt_mat>& data, i32 index) const;

		virtual void get(map<wstring, mt_mat>& datas, const mt_range& range) const;

		virtual mt_mat get(map<wstring, mt_mat>& datas, const vector<i32>& indexs) const;

		virtual void feature(map<wstring, mt_mat>& features, i32 index = -1) const;
		virtual void feature(map<wstring, mt_mat>& features, const mt_range& range) const;
		virtual void feature(map<wstring, mt_mat>& features, const vector<i32>& indexs) const;
		virtual void feature(map<wstring, mt_mat>& features, const vector<mt_range>& ranges, i32 suggest_sequence_length = 1) const;

		virtual void response(map<wstring, mt_mat>& labels, i32 index = -1) const;
		virtual void response(map<wstring, mt_mat>& labels, const mt_range& range) const;
		virtual void response(map<wstring, mt_mat>& labels, const vector<i32>& indexs) const;
		virtual void response(map<wstring, mt_mat>& labels, const vector<mt_range>& ranges, i32 suggest_sequence_length = 1) const;

		virtual void label(map<wstring, mt_mat>& labels, map<wstring, mt_mat>& label_for_categories, const wstring& model_name, i32 index = -1) const;
		virtual void label(map<wstring, mt_mat>& labels, map<wstring, mt_mat>& label_for_categories, const wstring& model_name, const mt_range& range) const;
		virtual void label(map<wstring, mt_mat>& labels, map<wstring, mt_mat>& label_for_categories, const wstring& model_name, const vector<i32>& indexs) const;
		virtual void label(map<wstring, mt_mat>& labels, map<wstring, mt_mat>& label_for_categories, const wstring& model_name, const vector<mt_range>& range, i32 suggest_sequence_length = 1) const;

		virtual void set_weight(const mt_mat& weight) {
			m_weight = weight;
		}

		virtual mt_mat weight() const {
			return m_weight;
		}

		virtual void statistic_category(map<wstring, mt_mat>& label_for_categories) const;

		virtual void split(vector<ml_data*>& res, const vector<f64>& ratios) const {}

		virtual i32 sequence_number()  const;
		virtual i32 sample_number() const;

		virtual ml_data resample_align_weight(b8 can_share_memory = sys_true) const;

		virtual ml_data generate_one_vs_all_data(b8 adjust_weight = sys_true, b8 can_share_memory = sys_true) const;

		b8 is_feature_sequence() const;
		b8 is_seuqnce(const wstring& description) const;

	protected:
		friend class ml_batcher;

		b8 check_feature_sequence_length(const ml_data_element& element);
		virtual b8 check_description(const wstring& description) const;

		map<wstring, ml_data_element> m_datas;
		mt_mat m_weight;
	};

	class ml_batcher {
	public:

		ml_batcher(const ml_data& data, i32 batch_size, b8 random, b8 can_ignore_less_number_batch = sys_true);

		i32 batch_number() const;
		virtual i32 feature(map<wstring, mt_mat>& features, i32 batch_index = -1);
		virtual void label(map<wstring, mt_mat>& labels, map<wstring, mt_mat>& label_for_categories, const wstring& model_name, i32 batch_index = -1);

	protected:

		void calculate_bucket(vector<vector<mt_range>>& ranges);
		i32 batch_range(vector<mt_range>& ranges, i32 batch_index);

		vector<mt_range> m_reshap_ranges;

		i32 m_cache_batch_index;
		vector<i32> m_cache_batch_indexs;

		const ml_data* m_data;
		i32 m_batch_size;

		vector<mt_range> m_feature_sequence_ranges;
		vector<mt_range> m_label_seuqnce_ranges;
		b8 m_can_ignore_less_number_batch;
		vector<i32> m_bucket_sizes;
	};
}