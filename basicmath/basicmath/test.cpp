#include "test.h"
#include <iostream>
using namespace std;

int main() {
	sys_exe_config exe_config(new log_ide_console_logger(), sys_tester::Test_Type_Slience);

	vector<wstring> argvs;
	mt_mat_test().run(argvs);

	return 0;
}