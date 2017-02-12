#pragma once


namespace basicml {
	class ml_nn_layer_learning_params {
	public:
		ml_nn_layer_learning_params(b8 inference, i32 index, i32 total_iteration_number, b8 batch_norm_statistic = sys_false) {
			m_inference_stage = inference;
			m_iteration_index = index;
			m_batch_norm_statistic = batch_norm_statistic;
			m_total_iteration_number = total_iteration_number;
			m_batch_norm_statistic = sys_false;
			m_sequence_length = 1;
		}

		b8 m_batch_norm_statistic;

		i32 m_total_iteration_number;
		b8 m_inference_stage;
		i32 m_iteration_index;
		i32 m_sequence_length;
	};
}

