#include "test.h"
#include <iostream>
#include <limits>
using namespace std;

int main() {
	sys_exe_config exe_config(new log_ide_console_logger(), sys_tester::Test_Type_Slience);

	vector<wstring> argvs;
	mt_mat_test().run(argvs);

	f32 val = -(f32)mt_Infinity;

	if (val < 0 && mt_helper::is_infinity(val)) {
		basiclog_info2(L"haha");
	}

	return 0;
}