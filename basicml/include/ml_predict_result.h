#pragma once

#include "ml_statistic_info.h"

namespace basicml {

	class ml_supervised_algorithm;

	class ml_predict_result {
	public:

		ml_predict_result() {

		}

		ml_predict_result(const wstring& name, const map<wstring, mt_mat>& label_for_categories = map<wstring, mt_mat>()) {
			m_model_name = name;
			m_label_for_categories = &label_for_categories;
		}

		ml_predict_result(const ml_supervised_algorithm* model);

		void set_model_name(const wstring& model_name) {
			m_model_name = model_name;
		}

		const wstring& model_name() const {
			return m_model_name;
		}

		void set_label_for_category(const map<wstring, mt_mat>& label_for_categories) {
			m_label_for_categories = &label_for_categories;
		}

		const map<wstring, mt_mat>* label_for_category() const {
			return m_label_for_categories;
		}

		void set_label(const map<wstring, mt_mat>& labels) {
			m_labels = labels;
		}

		void set_task_weight(const map<wstring, f64>& weights) {
			m_task_weights = weights;
		}

		void set_loss_type(const map<wstring, mt_Loss_Type>& loss_types) {
			m_loss_types = loss_types;
		}

		const map<wstring, mt_mat>& label() const {
			return m_labels;
		}

		void response(map<wstring, mt_mat>& responses) const;
		mt_mat one_response(const wstring& response_name) const;

		void probability(map<wstring, mt_mat>& probabilities, const map<wstring, i32>& categories) const;
		mt_mat one_probability(i32 category) const;

		void statistic_from_response(ml_statistic_info& info, const map<wstring, mt_mat>& ground_truth_responses) const;
		void statistic_from_label(ml_statistic_info& info, const map<wstring, mt_mat>& ground_truth_labels) const;

	protected:

		map<wstring, mt_mat> m_labels;
		map<wstring, f64> m_task_weights;
		map<wstring, mt_Loss_Type> m_loss_types;

		const map<wstring, mt_mat>* m_label_for_categories; 
		wstring m_model_name;
	}; 

}