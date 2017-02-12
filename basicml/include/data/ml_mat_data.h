#pragma once


#include "ml_data.h"

namespace basicml {



	class ml_mat_data : public ml_data {
	public:

		ml_mat_data();

		void add(const ml_data_element& element);

		virtual mt_mat get(const wstring& description, i32 index = -1) const = 0;

		virtual mt_mat get(const wstring& description, const mt_range& range = mt_range(-1, -1)) const = 0;

		virtual mt_mat get(const wstring& description, const vector<i32>& indexs = vector<i32>()) const = 0;

		virtual void get(map<wstring, mt_mat>& data, i32 index) const = 0;

		virtual void get(map<wstring, mt_mat>& datas, const mt_range& range) const = 0;

		virtual mt_mat get(map<wstring, mt_mat>& datas, const vector<i32>& indexs = vector<i32>()) const = 0;

		virtual void feature(map<wstring, mt_mat>& features, i32 index = -1) const;
		virtual void feature(map<wstring, mt_mat>& features, const mt_range& range = mt_range(-1, -1)) const;
		virtual void feature(map<wstring, mt_mat>& features, const vector<i32>& indexs = vector<i32>()) const;

		virtual void label(map<wstring, mt_mat>& labels, map<wstring, mt_mat>& label_for_categories, const wstring& model_name, i32 index = -1) const;
		virtual void label(map<wstring, mt_mat>& labels, map<wstring, mt_mat>& label_for_categories, const wstring& model_name, const mt_range& range = mt_range(-1, -1)) const;
		virtual void label(map<wstring, mt_mat>& labels, map<wstring, mt_mat>& label_for_categories, const wstring& model_name, const vector<i32>& indexs = vector<i32>()) const;

		virtual void statistic_category(map<wstring, mt_mat>& label_for_categories) const {}

		virtual void split(vector<ml_data*>& res, const vector<f64>& ratios) const {}

		virtual i32 sequence_length(const wstring& input_layer_name = L"")  const = 0;

		virtual i32 number() const;

	protected:

		map<wstring, ml_data_element> m_datas;

		b8 check_description(const wstring& description) const;
	};

}