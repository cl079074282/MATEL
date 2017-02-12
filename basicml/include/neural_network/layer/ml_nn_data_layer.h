#pragma once

#include "ml_nn_layer.h"
#include "ml_nn_layer_learning_params.h"

namespace basicml {

	

class ml_nn_data_layer : public ml_nn_layer {
public:

	ml_nn_data_layer() 
		: m_feedforward_count(0)
		, m_activate_type(mt_Activate_Type_Linear)
		, m_channels(1) {
	}

	ml_nn_data_layer(const wstring& layer_name, i32 unit_number, mt_Activate_Type activate_type = mt_Activate_Type_Linear, const vector<f64>& activate_params = vector<f64>()) 
		: m_feedforward_count(0)
		, m_activate_type(activate_type)
		, m_channels(1)
		, m_activate_params(activate_params) {
			m_data_sizes.push_back(unit_number);
	}

	ml_nn_data_layer(const wstring& layer_name, i32 width, i32 height, i32 channels, mt_Activate_Type activate_type = mt_Activate_Type_Linear, const vector<f64>& activate_params = vector<f64>()) 
		: m_feedforward_count(0)
		, m_activate_type(activate_type)
		, m_channels(channels)
		, m_activate_params(activate_params) {
			m_data_sizes.push_back(width);
			m_data_sizes.push_back(height);
	}

	virtual void feedforward_drop_drawn_singal(const vector<mt_mat>& drawn_singals, const ml_nn_layer_learning_params& pars);


	virtual void feedforward_singal(const mt_mat& ff_singal, const ml_nn_layer_learning_params& pars);
	virtual void feedforward_singal(const vector<mt_mat>& ff_singals, const ml_nn_layer_learning_params& pars);

	virtual void update_learning_param(const vector<mt_mat>& losses, const ml_nn_layer_learning_params& pars) {};

	const vector<ml_nn_linked_layer*>& get_prev_linked_layers() const {
		return m_prev_linked_layers;
	}

	const vector<ml_nn_linked_layer*>& get_next_linked_layers() const {
		return m_next_linked_layers;
	}

	const vector<mt_mat>& get_activated_output() const {
		return m_activated_signals;
	}

	const mt_mat& get_front_activated_output() const {
		return m_activated_signals.front();
	}

	void set_activate_func_type(mt_Activate_Type activate_func_type, const vector<double>& activate_pars = vector<double>()) {
		m_activate_type = activate_func_type;
		m_activate_params = activate_pars;
	}

	void set_size(int unit_number) {
		m_data_sizes.push_back(unit_number);
	}

	void set_size(int rows, int cols) {
		m_data_sizes.resize(2);
		m_data_sizes[0] = rows;
		m_data_sizes[1] = cols;
	}

	void set_size(int planes, int rows, int cols) {
		m_data_sizes.resize(3);
		m_data_sizes[0] = planes;
		m_data_sizes[1] = rows;
		m_data_sizes[2] = cols;
	}

	void set_size(const vector<int>& sizes) {
		m_data_sizes = sizes;
	}

	int feature_size() const {
		return channel_feature_size() * m_channels;
	}

	int channel_feature_size() const {
		int res = 1;
		for (int i = 0; i < (int)m_data_sizes.size(); ++i) {
			res *= m_data_sizes[i];
		}

		return res;
	}

	const vector<int>& size() const {
		return m_data_sizes;
	}

	void set_channel(int channel) {
		m_channels = channel;
	}

	int channel() const {
		return m_channels;
	}

	mt_Activate_Type activate_type() const {
		return m_activate_type;
	}

	const vector<double>& activate_params() const {
		return m_activate_params;
	}

	virtual void compute_default_setting();

	ml_nn_layer* clone() const;
	virtual ml_nn_data_layer* to_data_layer() {return this;}
	virtual const ml_nn_data_layer* to_data_layer() const {return this;}

	virtual void write(sys_json_writer& writer, b8 write_learned_param = sys_true) const;
	static ml_nn_data_layer* read(const sys_json_reader& reader);

protected:

	friend class ml_nn_linked_layer;	

	void try_activate(const ml_nn_layer_learning_params& pars);

	int m_feedforward_count;

	vector<vector<mt_mat>> m_input_singals;
	vector<mt_mat> m_activated_signals;
	vector<mt_mat> m_drawn_input_singals;

	vector<ml_nn_linked_layer*> m_prev_linked_layers;
	vector<ml_nn_linked_layer*> m_next_linked_layers;

	mt_Activate_Type m_activate_type;
	vector<f64> m_activate_params;

	vector<i32> m_data_sizes;
	i32 m_channels;
};

}

