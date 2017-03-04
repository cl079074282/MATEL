#pragma once

#include "ml_nn_linked_layer.h"

namespace basicml {

	static const wstring ml_Drop_Type_Descriptions[] = {L"null", L"out", L"connect"};
	static const wstring ml_Inference_Type_Descriptions[] = {L"average", L"drawn"};

	class ml_learning_param_updater;

	class ml_nn_inner_product_linked_layer : public ml_nn_linked_layer {
	public:

		enum Drop_Type {
			Drop_Type_Null,
			Drop_Type_Out,
			Drop_Type_Connect,
		};

		enum Inference_Type {
			Inference_By_Average,
			Inference_Type_Drawn,
		};

		ml_nn_inner_product_linked_layer() {

		}

		ml_nn_inner_product_linked_layer(const wstring& layer_name, ml_nn_data_layer* input_layer, ml_nn_data_layer* output_layer, ml_learning_param_updater* weight_updater = NULL, ml_learning_param_updater* bias_updater = NULL, Drop_Type drop_type = Drop_Type_Null, f64 drop_ratio = 0.5, Inference_Type inference_type = Inference_By_Average, i32 drawn_number = 1)
			: ml_nn_linked_layer(layer_name, input_layer, output_layer, weight_updater, bias_updater) {

			m_drop_type = drop_type;
			m_drop_ratio = drop_ratio;
			m_inference_type = inference_type;
			m_drawn_number = drawn_number;
		}

		void init_need_learn_params(int data_type);

		void feedforward(const ml_nn_layer_learning_params& pars);
		void update_learning_param(const vector<mt_mat>& losses, const ml_nn_layer_learning_params& pars);
		ml_nn_layer* clone() const;

		virtual void write(sys_json_writer& writer, b8 save_learned_param = sys_true) const;
		static ml_nn_inner_product_linked_layer* read(const sys_json_reader& reader);

		virtual void copy_learned_param(const ml_nn_linked_layer* other);
		
		/** 

		For such pre-training processing like DBN or auto-encoder, we may use this method to create the linked layer with the learned parameters.

		@note the momentum will be reset to zero.
		*/
		virtual void copy_learned_param(const mt_mat& weight, const mt_mat& bias);

		virtual ml_nn_inner_product_linked_layer* to_inner_product_linked_layer() {return this;}
		virtual const ml_nn_inner_product_linked_layer* to_inner_product_linked_layer() const {return this;}

		void set_drop_type(Drop_Type type) {
			m_drop_type = type;
		}

		int get_drop_type() const {
			return m_drop_type;
		}

		void set_inference_type(int type) {
			m_inference_type = type;
		}

		int get_inference_type() const {
			return m_inference_type;
		}

		void set_drawn_number(int number) {
			m_drawn_number = number;
		}

		int get_drawn_number() const {
			return m_drawn_number;
		} 

		void set_drop_ratio(double ratio) {
			m_drop_ratio = ratio;
		}

		double get_drop_ratio() const {
			return m_drop_ratio;
		}

	protected:

		mt_mat m_weight;
		mt_mat m_bias;

		Drop_Type m_drop_type;
		int m_inference_type;
		int m_drawn_number;
		double m_drop_ratio;
	};

	
}