#pragma once

#include "ml_nn_linked_layer.h"

namespace basicml {
	class ml_nn_lstm_gate_info {
	public:

		Mat m_in_to_gate_weight;
		Mat m_cout_to_gate_weight;

		Mat m_cell_to_gate_weight;
		Mat m_gate_bias;

		Mat m_d_in_to_gate_weight;
		Mat m_d_cout_to_gate_weight;
		Mat m_d_cell_to_gate_weight;
		Mat m_d_gate_bias;

		Mat m_v_in_to_gate_weight;
		Mat m_v_cout_to_gate_weight;
		Mat m_v_cell_to_gate_weight;
		Mat m_v_gate_bias;

		Mat m_gate_activate_cache;
		Mat m_gate_delta_cache;
		int m_gate_activate_type;
		vector<double> m_gate_activate_params;
		bool m_enable_gate;
		bool m_enable_peep_connection;
	};

	class ml_nn_lstm_info {
	public:

		ml_nn_lstm_gate_info m_igate;
		ml_nn_lstm_gate_info m_fgate;
		ml_nn_lstm_gate_info m_ogate;

		Mat m_in_to_cin_weight;
		Mat m_cout_to_cin_weight;
		Mat m_cout_to_out_weight;
		Mat m_cin_bias;

		Mat m_d_in_to_cin_weight;
		Mat m_d_cout_to_cin_weight;
		Mat m_d_cout_to_out_weight;
		Mat m_d_cin_bias;

		Mat m_v_in_to_cin_weight;
		Mat m_v_cout_to_cin_weight;
		Mat m_v_cout_to_out_weight;
		Mat m_v_cin_bias;

		Mat m_d_cout_to_cin_weight_temp;
		Mat m_d_cout_to_gate_weight_temp;
		Mat m_d_cell_to_gate_weight_temp;

		Mat m_cin_activation_without_gate_cache;
		Mat m_cin_activation_cache;
		Mat m_cin_delta_cache;
		Mat m_cout_signal_cache;
		Mat m_cout_activation_without_gate_cache;
		Mat m_cout_activation_cache;
		Mat m_out_activation_without_bias_cache;		

		Mat m_cout_epsilon_cache;
		Mat m_cout_delta_cache;
		Mat m_cell_delta_cache;

		int m_cin_activate_type;
		vector<double> m_cin_activate_params;

		int m_cout_activate_type;
		vector<double> m_cout_activate_params;

		int m_block_number;
		int m_cell_pre_block_number;
	};

	class ml_nn_lstm_inner_product_linked_layer : public ml_nn_linked_layer {
	public:

		virtual void feedforward(const ml_nn_layer_learning_params& pars);
		virtual void backpropagation(const ml_nn_layer_learning_params& pars);

		void enable_input_gate(bool enable) {
			m_info.m_igate.m_enable_gate = enable;
		}

		void enable_forget_gate(bool enable) {
			m_info.m_fgate.m_enable_gate = enable;
		}

		void enable_output_gate(bool enable) {
			m_info.m_ogate.m_enable_gate = enable;
		}

		void enable_input_gate_peep_connection(bool enable) {
			m_info.m_igate.m_enable_peep_connection = enable;
		}

		void enable_forget_gate_peep_connection(bool enable) {
			m_info.m_fgate.m_enable_peep_connection = enable;
		}

		void enable_output_gate_peep_connection(bool enable) {
			m_info.m_ogate.m_enable_peep_connection = enable;
		}

		static void feedforward(ml_nn_lstm_info& info, const Mat& input_data, const vector<Range>& seq_ranges, bool forward);
		static void backpropagation(ml_nn_lstm_info& info, const Mat& input_data, const Mat& next_delta, const vector<Range>& seq_ranges, bool forward);

	protected:

		
		static void calculate_nonfirst_gate_activation(Mat& activation, Mat& temp_block, const Mat& prev_cout_activation, const Mat& prev_cout_signal, const ml_nn_lstm_gate_info& gate, bool first, int block_number, int cell_pre_block_number);
		static void multiply_gate_activation(Mat& res, const Mat& cell, Mat& gate_activation, int block_number, int cell_pre_block_number);
		static void cell_multiply_cell_gate_weight(Mat& res, const Mat& cell, const Mat& cell_gate_weight, int block_number, int cell_pre_block_number);
		static void gate_multiply_cell_gate_weight(Mat& res, const Mat& gate, const Mat& cell_gate_weight, int block_number, int cell_pre_block_number);
		static void gate_multiply_cell(Mat& cell_to_gate_weight, const Mat& gate, const Mat& cell, int block_number, int cell_pre_block_number);
		
		

		ml_nn_lstm_info m_info;
		Mat m_out_bias;
	};
}
