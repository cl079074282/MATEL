
#include "test.h"

int main() {
	sys_exe_config exe_config(new log_ide_console_logger(), sys_tester::Test_Type_Slience);

	vector<wstring> argvs;

	__int64 a = 0x0fffffffffffffff;

	basiclog_info2(a);

	sys_strcombine_test().run(argvs);
	sys_strhelper_test().run(argvs);
	sys_json_test().run(argvs);
	sys_buffer_test().run(argvs);

	i16 i = 1;

	u8* ptr_data = (u8*)&i;

	basiclog_info2(ptr_data[0]);



	return 0;
}