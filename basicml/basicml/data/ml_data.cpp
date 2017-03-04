#include "stdafx.h"

#include "ml_data.h"



b8 ml_data::add(const ml_data_element& element) {
	if (!check_description(element.m_description)) {
		return sys_false;
	}

	if (!check_feature_sequence_length(element)) {
		return sys_false;
	}

	m_datas.insert(make_pair(element.m_description, element));

	return sys_true;
}

b8 ml_data::has_sequence() const {
	for (map<wstring, ml_data_element>::const_iterator iter = m_datas.begin(); iter != m_datas.end(); ++iter) {
		if (iter->second.m_sequence_data) {
			return sys_true;
		}
	}

	return sys_false;
}

b8 ml_data::has_non_sequence() const {
	for (map<wstring, ml_data_element>::const_iterator iter = m_datas.begin(); iter != m_datas.end(); ++iter) {
		if (!iter->second.m_sequence_data) {
			return sys_true;
		}
	}

	return sys_false;
}

b8 ml_data::has_sequence_feature() const {
	for (map<wstring, ml_data_element>::const_iterator iter = m_datas.begin(); iter != m_datas.end(); ++iter) {
		if (iter->second.m_description.find(ml_Data_Description_Feature) != wstring::npos) {
			if (iter->second.m_sequence_data) {
				return sys_true;
			}
		}
	}

	return sys_false;
}

b8 ml_data::has_non_sequence_feature() const {
	for (map<wstring, ml_data_element>::const_iterator iter = m_datas.begin(); iter != m_datas.end(); ++iter) {
		if (iter->second.m_description.find(ml_Data_Description_Feature) != wstring::npos) {
			if (!iter->second.m_sequence_data) {
				return sys_true;
			}
		}
	}

	return sys_false;
}

b8 ml_data::has_sequence_response() const {
	for (map<wstring, ml_data_element>::const_iterator iter = m_datas.begin(); iter != m_datas.end(); ++iter) {
		if (iter->second.m_description.find(ml_Data_Description_Response) != wstring::npos) {
			if (iter->second.m_sequence_data) {
				return sys_true;
			}
		}
	}

	return sys_false;
}

b8 ml_data::has_non_sequence_response() const {
	for (map<wstring, ml_data_element>::const_iterator iter = m_datas.begin(); iter != m_datas.end(); ++iter) {
		if (iter->second.m_description.find(ml_Data_Description_Response) != wstring::npos) {
			if (!iter->second.m_sequence_data) {
				return sys_true;
			}
		}
	}

	return sys_false;
}

b8 ml_data::is_seuqnce(const wstring& description) const {
	basiclog_assert2(m_datas.find(description) != m_datas.end());

	return m_datas.find(description)->second.m_sequence_data;
}

void ml_data::response(map<wstring, mt_mat>& labels) const {
	for (map<wstring, ml_data_element>::const_iterator iter = m_datas.begin(); iter != m_datas.end(); ++iter) {
		if (iter->second.m_description.find(ml_Data_Description_Response) != wstring::npos) {
			labels[iter->first] = iter->second.m_mat;
		}
	}
}

void ml_data::feature(map<wstring, mt_mat>& features, const vector<i32>& sequence_indexs, const vector<i32>& non_sequence_indexs) const {

	for (map<wstring, ml_data_element>::const_iterator iter = m_datas.begin(); iter != m_datas.end(); ++iter) {
		if (iter->second.m_description.find(L"feature_") != wstring::npos) {
			vector<mt_mat> sub_mats;

			if (iter->second.m_sequence_data) {
				sub_mats.resize(sequence_indexs.size());

				for (i32 i = 0; i < (i32)sequence_indexs.size(); ++i) {
					sub_mats[i] = iter->second.m_mat.sub(sequence_indexs[i], sequence_indexs[i] + 1);
				}
			} else {
				sub_mats.resize(non_sequence_indexs.size());

				for (i32 i = 0; i < (i32)non_sequence_indexs.size(); ++i) {
					sub_mats[i] = iter->second.m_mat.sub(non_sequence_indexs[i], non_sequence_indexs[i] + 1);
				}
			}

			features[iter->first] = mt_mat_helper::merge_align_dim(sub_mats, 0);
		}
	}
}

mt_mat ml_data::feature(const wstring& feature_description, const vector<mt_range>& ranges, i32 suggest_sequence_length /* = 1 */) const {
	const ml_data_element& data_element = m_datas.find(feature_description)->second;

	vector<mt_mat> sub_mats;
	sub_mats.resize(ranges.size());

	i32 expand_sizes[] = {0, 0};

	for (i32 i = 0; i < (i32)ranges.size(); ++i) {
		sub_mats[i] = data_element.m_mat.sub(ranges[i]);
		expand_sizes[0] = suggest_sequence_length - ranges[i].size();
		sub_mats[i] = sub_mats[i].expand(2, NULL, expand_sizes, 0);
	}

	return mt_mat_helper::merge_align_dim(sub_mats, 0);
}

mt_mat ml_data::response(const wstring& response_description, const vector<mt_range>& ranges, i32 suggest_sequence_length /* = 1 */) const {
	const ml_data_element& data_element = m_datas.find(response_description)->second;

	vector<mt_mat> sub_mats;
	sub_mats.resize(ranges.size());

	i32 expand_sizes[] = {0, 0};

	for (i32 i = 0; i < (i32)ranges.size(); ++i) {
		sub_mats[i] = data_element.m_mat.sub(ranges[i]);
		expand_sizes[0] = suggest_sequence_length - ranges[i].size();
		sub_mats[i] = sub_mats[i].expand(2, NULL, expand_sizes, mt_Nan);
	}

	return mt_mat_helper::merge_align_dim(sub_mats, 0);
}

mt_mat ml_data::label(const wstring& response_description, map<wstring, mt_mat>& label_for_categories, const wstring& model_name, const vector<mt_range>& ranges, i32 suggest_sequence_length) const {
	mt_mat res = response(response_description, ranges, suggest_sequence_length);
	
	return ml_helper::label_from_response(res, label_for_categories.find(response_description)->second, model_name);
}

b8 ml_data::check_description(const wstring& description) const {
	if (description.find(ml_Data_Description_Feature) == wstring::npos
		&& description.find(ml_Data_Description_Response) == wstring::npos
		&& description != ml_Data_Description_Weight) {
			return sys_false;
	}

	for (map<wstring, ml_data_element>::const_iterator iter = m_datas.begin(); iter != m_datas.end(); ++iter) {
		if (description == iter->first) {
			return sys_false;
		}
	}

	return sys_true;
}

b8 ml_data::check_feature_sequence_length(const ml_data_element& element) {
	for (map<wstring, ml_data_element>::const_iterator iter = m_datas.begin(); iter != m_datas.end(); ++iter) {
		if (element.m_sequence_data == iter->second.m_sequence_data && iter->second.m_mat.size()[0] != element.m_mat.size()[0]) {
			basiclog_assert_message2(sys_false, L"sequence datas must be the same number, so is the non-sequence data!");
			return sys_false;
		}
	}

	return sys_true;
}

void ml_data::statistic_category(map<wstring, mt_mat>& label_for_categories) const {
	for (map<wstring, ml_data_element>::const_iterator iter = m_datas.begin(); iter != m_datas.end(); ++iter) {
		if (iter->second.m_description.find(ml_Data_Description_Response) != wstring::npos && iter->second.m_data_type.front() == ml_Data_Type_Discrete) {
			//classification task
			label_for_categories[iter->first] = ml_helper::group_category(iter->second.m_mat);
		}
	}
}

ml_data ml_data::resample_align_weight(b8 can_share_memory /* = sys_true */) const {
	return *this;
}

ml_batcher::ml_batcher(const ml_data& data, i32 batch_size, b8 random, b8 can_ignore_less_number_batch) {
	m_cache_batch_index = -1;

	m_data = &data;
	m_batch_size = batch_size;
	m_can_ignore_less_number_batch = can_ignore_less_number_batch;

	vector<vector<mt_range>> sequence_ranges;
	vector<vector<mt_range>> non_sequence_ranges;

	map<wstring, mt_mat> features;

	if (data.has_sequence_feature()) {
		for (map<wstring, ml_data_element>::const_iterator iter = data.m_datas.begin(); iter != data.m_datas.end(); ++iter) {
			if (iter->second.m_description.find(ml_Data_Description_Feature) != wstring::npos && iter->second.m_sequence_data) {
				ml_helper::statistic_data_range(sequence_ranges, iter->second.m_mat, sys_true);
			}
		}
	}

	if (data.has_non_sequence_response()) {
		for (map<wstring, ml_data_element>::const_iterator iter = data.m_datas.begin(); iter != data.m_datas.end(); ++iter) {
			if (iter->second.m_description.find(ml_Data_Description_Response) != wstring::npos && !iter->second.m_sequence_data) {
				ml_helper::statistic_data_range(non_sequence_ranges, iter->second.m_mat, sys_false);
			}
		}
	}

	reshape_range(m_sequence_ranges, sequence_ranges, random);
	reshape_range(m_non_sequence_ranges, non_sequence_ranges, random);

	if (!m_sequence_ranges.empty() && !m_non_sequence_ranges.empty()) {
		basiclog_assert2(m_sequence_ranges.size() == m_non_sequence_ranges.size());
	}
}

void ml_batcher::reshape_range(vector<mt_range>& ranges, vector<vector<mt_range>>& raw_ranges, b8 random) {
	if (random) {
		vector<i32> range_reshape_indexs;
		mt_random::randperm(range_reshape_indexs, (i32)raw_ranges.size());

		for (i32 i = 0; i < (i32)range_reshape_indexs.size(); ++i) {
			vector<mt_range>& range = raw_ranges[range_reshape_indexs[i]];

			vector<i32> reshape_indexs;
			mt_random::randperm(reshape_indexs, (i32)range.size());

			for (i32 j = 0; j < (i32)reshape_indexs.size(); ++j) {
				ranges.push_back(range[reshape_indexs[j]]);
			}
		}
	} else {
		for (i32 i = 0; i < (i32)raw_ranges.size(); ++i) {
			vector<mt_range>& range = raw_ranges[i];

			for (i32 j = 0; j < (i32)range.size(); ++j) {
				ranges.push_back(range[j]);
			}
		}
	}
}

i32 ml_batcher::batch_number() const {
	i32 range_number = m_sequence_ranges.empty() ? (i32)m_non_sequence_ranges.size() : (i32)m_sequence_ranges.size();

	if (m_can_ignore_less_number_batch) {
		return range_number / m_batch_size;
	} else {
		return (range_number - 1) / m_batch_size + 1;
	}
}

i32 ml_batcher::feature(map<wstring, mt_mat>& features, i32 batch_index /* = -1 */) {
	i32 suggest_sequence_length = 1;
	
	for (map<wstring, ml_data_element>::const_iterator iter = m_data->m_datas.begin(); iter != m_data->m_datas.end(); ++iter) {
		if (iter->second.m_description.find(ml_Data_Description_Feature) != wstring::npos) {
			vector<mt_range> ranges;

			if (iter->second.m_sequence_data) {
				suggest_sequence_length = bucket_batch_range(ranges, batch_index);

				features[iter->first] = m_data->feature(iter->first, ranges, suggest_sequence_length);
			} else {				
				normal_batch_range(ranges, batch_index);

				features[iter->first] = m_data->feature(iter->first, ranges);
			}
		}
	}

	return suggest_sequence_length;
}

void ml_batcher::label(map<wstring, mt_mat>& labels, map<wstring, mt_mat>& label_for_categories, const wstring& model_name, i32 batch_index /* = -1 */) {
	for (map<wstring, ml_data_element>::const_iterator iter = m_data->m_datas.begin(); iter != m_data->m_datas.end(); ++iter) {
		if (iter->second.m_description.find(ml_Data_Description_Response) != wstring::npos) {
			vector<mt_range> ranges;

			if (iter->second.m_sequence_data) {
				i32 suggest_sequence_length = bucket_batch_range(ranges, batch_index);

				labels[iter->first] = m_data->label(iter->first, label_for_categories, model_name, ranges, suggest_sequence_length);
			} else {				
				normal_batch_range(ranges, batch_index);

				labels[iter->first] = m_data->label(iter->first, label_for_categories, model_name, ranges);
			}
		}
	}
}

i32 ml_batcher::bucket_batch_range(vector<mt_range>& ranges, i32 batch_index) {
	i32 max_seuqnce_length = 0;
	for (i32 i = batch_index * m_batch_size; i < (batch_index + 1) * m_batch_size && i < (i32)m_sequence_ranges.size(); ++i) {
		ranges.push_back(m_sequence_ranges[i]);

		if (ranges[i].size() > max_seuqnce_length) {
			max_seuqnce_length = ranges[i].size();
		}
	}

	for (i32 i = 0; i < (i32)m_bucket_sizes.size(); ++i) {
		if (m_bucket_sizes[i] >= max_seuqnce_length) {
			max_seuqnce_length = m_bucket_sizes[i];
			break;
		}
	}

	return max_seuqnce_length;
}

void ml_batcher::normal_batch_range(vector<mt_range>& ranges, i32 batch_index) {
	for (i32 i = batch_index * m_batch_size; i < (batch_index + 1) * m_batch_size && i < (i32)m_non_sequence_ranges.size(); ++i) {
		ranges.push_back(m_non_sequence_ranges[i]);
	}
}

void ml_batcher::calculate_bucket(vector<vector<mt_range>>& ranges) {
	vector<f64> datas;

	for (i32 i = 0; i < (i32)ranges.size(); ++i) {
		for (i32 j = 0; j < (i32)ranges[i].size(); ++j) {
			datas.push_back(ranges[i][j].size());
		}
	}

	vector<mt_range_t<f64>> hist_ranges;
	vector<i32> numbers;
	mt_hist::hist(hist_ranges, numbers, datas, 16);

	vector<i32> sorted_indexs;
	mt_sort::sort<i32>(sorted_indexs, (i32)numbers.size(), &numbers[0], sys_false);

	for (i32 i = 0; i < 5; ++i) {
		m_bucket_sizes.push_back((i32)hist_ranges[sorted_indexs[i]].m_end);
	}
}