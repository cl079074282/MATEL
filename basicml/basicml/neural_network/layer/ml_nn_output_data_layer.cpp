#include "stdafx.h"

#include "ml_nn_output_data_layer.h"
#include "ml_nn_linked_layer.h"

mt_mat ml_nn_output_data_layer::label() const {
	mt_mat res = mt_mat_helper::merge_align_channel(m_activated_signals);

	i32 rows = res.size()[0];
	i32 cols = res.element_number() / rows;

	return res.reshape(rows, cols);
}

ml_nn_layer* ml_nn_output_data_layer::clone() const {
	return new ml_nn_output_data_layer();
}

void ml_nn_output_data_layer::write(sys_json_writer& writer, b8 write_learned_param /* = sys_true */) const {

}

ml_nn_output_data_layer* ml_nn_output_data_layer::read(const sys_json_reader& reader) {

}