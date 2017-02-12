/** @file ml_model.h
*/

#pragma once

#include "data/ml_data.h"
#include "ml_algorithm.h"
#include "ml_predict_result.h"
#include "ml_statistic_info.h"

namespace basicml {
	class ml_supervised_algorithm : public ml_algorithm {
	public:
		virtual ~ml_supervised_algorithm() {

		}

		virtual void train(const ml_data& training_data, const ml_data& validation_data = ml_data()) = 0;

		virtual void predict(ml_predict_result& res, const ml_data& features) const = 0;
		virtual void evaluate(ml_statistic_info& statistic_info, const ml_data& data) const = 0;

		/** Set label for category map.
		If you do not invoke this method before train the model
		*/
		void set_label_for_category(const map<wstring, mt_mat>& label_for_catefories) {
			m_label_for_categories = label_for_catefories;
		}

		virtual const map<wstring, mt_mat>& label_for_category() const {
			return m_label_for_categories;
		}

		static wstring& name() {
			static wstring sname(L"ml_supervised_algorithm");
			return sname;
		}

		virtual ml_supervised_algorithm* to_supervised_model() {return this;}
		virtual const ml_supervised_algorithm* to_supervised_model() const {return this;}

	protected:

		map<wstring, mt_mat> m_label_for_categories;
	};
}