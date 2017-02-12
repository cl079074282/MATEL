#include "test.h"

void ml_neural_network_test::run(vector<wstring>& argvs) {
	ml_neural_network nn;

	ml_nn_input_data_layer* input_layer = new ml_nn_input_data_layer(L"input", 100);
	ml_nn_data_layer* hidden_layer = new ml_nn_data_layer(L"hidden", 100, mt_Activate_Type_Sigmoid);
	ml_nn_output_data_layer* output_layer = new ml_nn_output_data_layer(L"output", 10, mt_Activate_Type_Softmax, mt_Loss_Type_Logarithmic);
	ml_nn_inner_product_linked_layer* linked_1_layer = new ml_nn_inner_product_linked_layer(L"linked_1", input_layer, hidden_layer);
	ml_nn_inner_product_linked_layer* linked_2_layer = new ml_nn_inner_product_linked_layer(L"linked_1", hidden_layer, output_layer);

	nn.add_layer(input_layer);
	nn.add_layer(hidden_layer);
	nn.add_layer(output_layer);

	nn.add_layer(linked_1_layer);
	nn.add_layer(linked_2_layer);

	nn.setup();



}