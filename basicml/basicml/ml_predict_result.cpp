#include "stdafx.h"

#include "ml_predict_result.h"



ml_predict_result::ml_predict_result(const ml_supervised_algorithm* model) {
	m_label_for_categories = &model->label_for_category();
	m_model_name = model->class_name();
}

void ml_predict_result::statistic_from_response(ml_statistic_info& info, const map<wstring, mt_mat>& ground_truth_responses) const {
	map<wstring, mt_mat> ground_truth_labels;
	ml_helper::label_from_response(ground_truth_labels, ground_truth_responses, *m_label_for_categories, m_model_name);

	statistic_from_label(info, ground_truth_labels);
}

void ml_predict_result::response(map<wstring, mt_mat>& responses) const {
	ml_helper::response_from_label(responses, m_labels, *m_label_for_categories, m_model_name);
}

void ml_predict_result::statistic_from_label(ml_statistic_info& info, const map<wstring, mt_mat>& ground_truth_labels) const {
	const map<wstring, mt_mat>& label_for_categories = *m_label_for_categories;
	
	for (map<wstring, mt_mat>::const_iterator iter = ground_truth_labels.begin(); iter != ground_truth_labels.end(); ++iter) {
		mt_mat predict_label = m_labels.at(iter->first);
		mt_mat groundtruth_label = ground_truth_labels.at(iter->first);
		i32 category_number = -1;
		i32 valid_sample_number = ml_helper::calculate_valid_sample_number(groundtruth_label);

		mt_mat loss = predict_label.loss(groundtruth_label, m_loss_types.at(iter->first)) / valid_sample_number;
		mt_scalar loss_scalar = loss.get(0, 0);

		info.m_losses[iter->first] = loss_scalar[0] * m_task_weights.at(iter->first);

		if (label_for_categories.find(iter->first) != label_for_categories.end()) {
			//classification task
			info.m_sequence_precisions[iter->first] = 1 - ml_helper::calculate_wrong_sequence_number_from_label(predict_label, groundtruth_label) / (f64)ml_helper::calculate_valid_sequence_number(groundtruth_label);
			info.m_precisions[iter->first] = 1 - ml_helper::calculate_wrong_sample_number_from_label(predict_label, groundtruth_label) / (f64)valid_sample_number;
		}
	}


}