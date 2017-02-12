#pragma once

#include "ml_nn_data_layer.h"

namespace basicml {
	class ml_file_storage;
	class ml_file_node;

	class ml_nn_input_data_layer : public ml_nn_data_layer {
	public:

		ml_nn_input_data_layer() {
		}

		ml_nn_input_data_layer(const wstring& layer_name, i32 unit_number)		
			: ml_nn_data_layer(layer_name, unit_number, mt_Activate_Type_Linear) {

		}

		ml_nn_input_data_layer(const wstring& layer_name, i32 width, i32 height, i32 channels)		
			: ml_nn_data_layer(layer_name, width, height, channels, mt_Activate_Type_Linear) {

		}

		void feedforward_by_input(const mt_mat& input, const ml_nn_layer_learning_params& pars);

		ml_nn_layer* clone() const;
		virtual ml_nn_input_data_layer* to_input_data_layer() {return this;}
		virtual const ml_nn_input_data_layer* to_input_data_layer() const {return this;}

		virtual void write(sys_json_writer& writer, b8 write_learned_param = sys_true) const;
		static ml_nn_input_data_layer* read(const sys_json_reader& reader);

	protected:

		
	};
}