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

b8 ml_data::is_feature_sequence() const {
	for (map<wstring, ml_data_element>::const_iterator iter = m_datas.begin(); iter != m_datas.end(); ++iter) {
		if (iter->second.m_description.find(L"feature_") != wstring::npos) {
			return iter->second.m_sequence_data;
		}
	}

	basiclog_assert2(sys_false);
	return sys_false;
}

b8 ml_data::is_seuqnce(const wstring& description) const {
	basiclog_assert2(m_datas.find(description) != m_datas.end());

	return m_datas.find(description)->second.m_sequence_data;
}

void ml_data::feature(map<wstring, mt_mat>& features, const vector<i32>& indexs /* = vector<i32>() */) const {

	for (map<wstring, ml_data_element>::const_iterator iter = m_datas.begin(); iter != m_datas.end(); ++iter) {
		if (iter->second.m_description.find(L"feature_") != wstring::npos) {
			vector<mt_mat> sub_mats;
			sub_mats.resize(indexs.size());

			for (i32 i = 0; i < (i32)indexs.size(); ++i) {
				sub_mats[i] = iter->second.m_mat.sub(indexs[i], indexs[i] + 1);
			}

			features[iter->first] = mt_mat_helper::merge_align_dim(sub_mats, 0);
		}
	}
}

void ml_data::feature(map<wstring, mt_mat>& features, const vector<mt_range>& ranges, i32 suggest_sequence_length /* = 1 */) const {
	for (map<wstring, ml_data_element>::const_iterator iter = m_datas.begin(); iter != m_datas.end(); ++iter) {
		if (iter->second.m_description.find(L"feature_") != wstring::npos) {
			vector<mt_mat> sub_mats;
			sub_mats.resize(ranges.size());

			i32 expand_sizes[] = {0, 0};

			for (i32 i = 0; i < (i32)ranges.size(); ++i) {
				sub_mats[i] = iter->second.m_mat.sub(ranges[i]);
				expand_sizes[0] = suggest_sequence_length - ranges[i].size();
				expand_sizes[1] = sub_mats[i].size()[1];
				sub_mats[i] = sub_mats[i].expand(2, NULL, expand_sizes, 0);
			}

			features[iter->first] = mt_mat_helper::merge_align_dim(sub_mats, 0);
		}
	}
}

void ml_data::response(map<wstring, mt_mat>& features, const vector<mt_range>& ranges, i32 suggest_sequence_length /* = 1 */) const {
	for (map<wstring, ml_data_element>::const_iterator iter = m_datas.begin(); iter != m_datas.end(); ++iter) {
		if (iter->second.m_description.find(L"response_") != wstring::npos) {
			vector<mt_mat> sub_mats;
			sub_mats.resize(ranges.size());

			i32 expand_sizes[] = {0, 0};

			for (i32 i = 0; i < (i32)ranges.size(); ++i) {
				sub_mats[i] = iter->second.m_mat.sub(ranges[i]);
				expand_sizes[0] = suggest_sequence_length - ranges[i].size();
				expand_sizes[1] = sub_mats[i].size()[1];
				sub_mats[i] = sub_mats[i].expand(2, NULL, expand_sizes, mt_helper::nan());
			}

			features[iter->first] = mt_mat_helper::merge_align_dim(sub_mats, 0);
		}
	}
}

void ml_data::label(map<wstring, mt_mat>& labels, map<wstring, mt_mat>& label_for_categories, const wstring& model_name, const vector<mt_range>& ranges, i32 suggest_sequence_length) const {
	map<wstring, mt_mat> responses;
	response(responses, ranges, suggest_sequence_length);

	ml_helper::label_from_response(labels, responses, label_for_categories, model_name);
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
	b8 sequence_feature = is_feature_sequence();

	if (element.m_sequence_data != sequence_feature) {
		basiclog_assert2(L"features must be all sequence or non-sequence data!");
		return sys_false;
	}

	for (map<wstring, ml_data_element>::const_iterator iter = m_datas.begin(); iter != m_datas.end(); ++iter) {
		if (!iter->second.m_mat.is_same_size(element.m_mat)) {
			return sys_false;
		}
	}

	return sys_true;
}

void ml_data::statistic_category(map<wstring, mt_mat>& label_for_categories) const {
	for (map<wstring, ml_data_element>::const_iterator iter = m_datas.begin(); iter != m_datas.end(); ++iter) {
		if (iter->second.m_data_type == ml_Data_Type_Discrete) {
			//classification task
			label_for_categories[iter->first] = ml_helper::group_category(iter->second.m_mat);
		}
	}
}

ml_batcher::ml_batcher(const ml_data& data, i32 batch_size, b8 random, b8 can_ignore_less_number_batch) {
	m_cache_batch_index = -1;

	m_data = &data;
	m_batch_size = batch_size;
	m_can_ignore_less_number_batch = can_ignore_less_number_batch;

	vector<vector<mt_range>> feature_ranges;
	map<wstring, mt_mat> features;
	
	ml_helper::statistic_data_range(feature_ranges, features.begin()->second, m_data->is_feature_sequence());
	
	map<wstring, mt_mat> labels;
	vector<vector<mt_range>> non_seuqnce_label_ranges;

	if (!non_seuqnce_label_ranges.empty() && non_seuqnce_label_ranges.size() != feature_ranges.size()) {
		basiclog_assert2(L"you should make sure the feature and response to be saved in aligned!");
	}

	if (random) {
		vector<i32> range_reshape_indexs;
		mt_random::randperm(range_reshape_indexs, (i32)feature_ranges.size());

		for (i32 i = 0; i < (i32)range_reshape_indexs.size(); ++i) {
			vector<mt_range>& range = feature_ranges[range_reshape_indexs[i]];

			vector<i32> reshape_indexs;
			mt_random::randperm(reshape_indexs, (i32)range.size());

			for (i32 j = 0; j < (i32)reshape_indexs.size(); ++j) {
				m_reshap_ranges.push_back(range[reshape_indexs[j]]);
			}
		}
	} else {
		for (i32 i = 0; i < (i32)feature_ranges.size(); ++i) {
			vector<mt_range>& range = feature_ranges[i];

			for (i32 j = 0; j < (i32)range.size(); ++j) {
				m_reshap_ranges.push_back(range[j]);
			}
		}
	}
}

i32 ml_batcher::batch_number() const {
	if (m_can_ignore_less_number_batch) {
		return (i32)m_reshap_ranges.size() / m_batch_size;
	} else {
		return ((i32)m_reshap_ranges.size() - 1) / m_batch_size + 1;
	}
}

i32 ml_batcher::feature(map<wstring, mt_mat>& features, i32 batch_index /* = -1 */) {
	vector<mt_range> ranges;
	i32 suggest_sequence_length = batch_range(ranges, batch_index);

	m_data->feature(features, ranges, suggest_sequence_length);

	return suggest_sequence_length;
}

void ml_batcher::label(map<wstring, mt_mat>& labels, map<wstring, mt_mat>& label_for_categories, const wstring& model_name, i32 batch_index /* = -1 */) {
	vector<mt_range> ranges;
	i32 suggest_sequence_length = batch_range(ranges, batch_index);

	m_data->label(labels, label_for_categories, model_name, ranges, suggest_sequence_length);
}

i32 ml_batcher::batch_range(vector<mt_range>& ranges, i32 batch_index) {
	i32 max_seuqnce_length = 0;
	for (i32 i = batch_index * m_batch_size; i < (batch_index + 1) * m_batch_size && i < (i32)m_reshap_ranges.size(); ++i) {
		ranges.push_back(m_reshap_ranges[i]);

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