#include "test.h"
using namespace std;

int main() {
	sys_exe_config exe_config(new log_ide_console_logger(), sys_tester::Test_Type_Slience);

	vector<wstring> argvs;
	ml_neural_network_test().run(argvs);

	return 0;
}