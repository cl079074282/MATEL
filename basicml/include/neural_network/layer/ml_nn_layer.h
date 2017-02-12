#pragma once



namespace basicml {
class ml_nn_layer_config;
class ml_nn_linked_layer;
class ml_nn_data_layer;
class ml_nn_output_data_layer;
class ml_nn_input_data_layer;
class ml_nn_maxout_linked_layer;
class ml_nn_pooling_linked_layer;
class ml_nn_convolution_linked_layer;
class ml_nn_inner_product_linked_layer;
class ml_nn_combination_linked_layer;
class ml_nn_batch_norm_linked_layer;

class ml_nn_layer {
public:

	ml_nn_layer() {

	}

	virtual ~ml_nn_layer() {}

	virtual void compute_default_setting() {}

	virtual ml_nn_layer* clone() const = 0;

	virtual ml_nn_data_layer* to_data_layer() {return NULL;}
	virtual const ml_nn_data_layer* to_data_layer() const {return NULL;}

	virtual ml_nn_linked_layer* to_linked_layer() {return NULL;}
	virtual const ml_nn_linked_layer* to_linked_layer() const {return NULL;}

	virtual ml_nn_maxout_linked_layer* to_maxout_linked_layer() {return NULL;}
	virtual const ml_nn_maxout_linked_layer* to_maxout_linked_layer() const {return NULL;}

	virtual ml_nn_pooling_linked_layer* to_pooling_linked_layer() {return NULL;}
	virtual const ml_nn_pooling_linked_layer* to_pooling_linked_layer() const {return NULL;}

	virtual ml_nn_convolution_linked_layer* to_convolution_linked_layer() {return NULL;}
	virtual const ml_nn_convolution_linked_layer* to_convolution_linked_layer() const {return NULL;}

	virtual ml_nn_inner_product_linked_layer* to_inner_product_linked_layer() {return NULL;}
	virtual const ml_nn_inner_product_linked_layer* to_inner_product_linked_layer() const {return NULL;}

	virtual ml_nn_batch_norm_linked_layer* to_batch_norm_linked_layer() {return NULL;}
	virtual const ml_nn_batch_norm_linked_layer* to_batch_norm_linked_layer() const {return NULL;}

	virtual ml_nn_combination_linked_layer* to_combination_linked_layer() {return NULL;}
	virtual const ml_nn_combination_linked_layer* to_combination_linked_layer() const {return NULL;}

	virtual ml_nn_input_data_layer* to_input_data_layer() {return NULL;}
	virtual const ml_nn_input_data_layer* to_input_data_layer() const {return NULL;}

	virtual ml_nn_output_data_layer* to_output_data_layer() {return NULL;}
	virtual const ml_nn_output_data_layer* to_output_data_layer() const {return NULL;}

	virtual void write(sys_json_writer& writer, b8 write_learned_param = sys_true) const = 0;
	static ml_nn_layer* read(const sys_json_reader& reader);

	void set_name(const wstring& name) {
		m_layer_name = name;
	}

	const wstring& name() const {
		return m_layer_name;
	}

	static const wstring& layer_type() {
		static wstring type = L"ml_nn_layer";
		return type;
	}

protected:



	wstring m_layer_name;
};


}