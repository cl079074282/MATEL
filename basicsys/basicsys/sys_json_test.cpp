#include "test.h"

static void write_json_context(sys_json_writer& json_writer) {
	json_writer<<L"name"<<L"xuhailiang";

	json_writer<<L"age"<<L"19";

	json_writer<<L"citys"<<L"[";

	json_writer<<L"����";
	json_writer<<L"nanjing";
	json_writer<<L"beijing";

	json_writer<<L"]";

	json_writer<<L"friends"<<L"[";

	json_writer<<L"{";

	json_writer<<L"name"<<L"chenyidong";
	json_writer<<L"age"<<L"26";
	json_writer<<L"citys"<<L"[";

	json_writer<<L"luan";
	json_writer<<L"nanjing";
	json_writer<<L"shenzhen";

	json_writer<<L"]";

	json_writer<<L"}";

	json_writer<<L"{";

	json_writer<<L"name"<<L"xuwei";
	json_writer<<L"age"<<L"26";
	json_writer<<L"citys"<<L"[";

	json_writer<<L"hefei";
	json_writer<<L"shanghai";

	json_writer<<L"]";

	json_writer<<L"}";

	json_writer<<L"]";

	json_writer<<L"empty_map"<<L"{"<<L"}";

	vector<i32> numbers;

	numbers.resize(200, 0);

	vector<vector<i32>> numbers2;
	numbers2.push_back(numbers);
	numbers2.push_back(numbers);

	vector<vector<vector<i32>>> numbers3;
	numbers3.push_back(numbers2);
	numbers3.push_back(numbers2);

	json_writer<<L"number_array"<<numbers3;

	vector<wstring> strings;
	strings.push_back(L"haha");
	strings.push_back(L"");
	strings.push_back(L"world");

	vector<vector<wstring>> strings2;
	strings2.push_back(strings);

	vector<vector<vector<wstring>>> strings3;
	strings3.push_back(strings2);

	json_writer<<L"string_array"<<strings3;
}

static void read_json(sys_json_reader& reader) {
	sys_test_equal((wstring)reader[L"name"], L"xuhailiang");
	sys_test_equal((wstring)reader[L"age"], L"19");
	sys_test_equal((wstring)reader[L"citys"][0], L"����");
	sys_test_equal((wstring)reader[L"citys"][1], L"nanjing");
	sys_test_equal((wstring)reader[L"citys"][2], L"beijing");

	sys_test_equal((wstring)reader[L"friends"][0][L"name"], L"chenyidong");
	sys_test_equal((wstring)reader[L"friends"][0][L"citys"][0], L"luan");
	sys_test_equal((wstring)reader[L"friends"][0][L"citys"][1], L"nanjing");
	sys_test_equal((wstring)reader[L"friends"][0][L"citys"][2], L"shenzhen");
	sys_test_equal((wstring)reader[L"friends"][0][L"age"], L"26");

	sys_test_equal((wstring)reader[L"friends"][1][L"name"], L"xuwei");
	sys_test_equal((wstring)reader[L"friends"][1][L"citys"][0], L"hefei");
	sys_test_equal((wstring)reader[L"friends"][1][L"citys"][1], L"shanghai");
	sys_test_equal((wstring)reader[L"friends"][1][L"age"], L"26");

	i32 size = reader[L"empty_map"].size();

	vector<vector<vector<i32>>> numbers;
	reader[L"number_array"]>>numbers;

	basiclog_info2(numbers);

	vector<vector<vector<wstring>>> strings;
	reader[L"string_array"]>>strings;

	basiclog_info2(strings);
}

static void test_string_buffer_json() {
	wstring str;
	sys_string_buffer_writer string_file_writer(str);
	sys_json_writer json_writer(&string_file_writer);

	write_json_context(json_writer);

	basiclog_info2(str);

	sys_string_buffer_reader string_buffer_reader(str);
	sys_json_reader json_reader(&string_buffer_reader, L"");

	read_json(json_reader);
}


static void test_string_file_buffer_json() {
	sys_string_file_buffer_writer string_file_writer(L"test.txt");
	sys_json_writer json_writer(&string_file_writer);

	write_json_context(json_writer);

	string_file_writer.close();

	sys_string_file_buffer_reader string_file_reader(L"test.txt");
	sys_json_reader json_reader(&string_file_reader, L"");

	read_json(json_reader);
}


static void test_byte_buffer_json() {
	vector<u8> buffer;
	sys_byte_buffer_writer writer(buffer);
	sys_json_writer json_writer(&writer);

	write_json_context(json_writer);

	//basiclog_info2(str);

	sys_byte_buffer_reader reader(&buffer[0], &buffer[0] + (i32)buffer.size());
	sys_json_reader json_reader(&reader, L"");

	read_json(json_reader);
}


static void test_byte_file_buffer_json() {
	sys_byte_file_buffer_writer writer(L"test.btxt");
	sys_json_writer json_writer(&writer);

	write_json_context(json_writer);

	writer.close();

	sys_byte_file_buffer_reader reader(L"test.btxt");
	sys_json_reader json_reader(&reader, L"");

	read_json(json_reader);
}

void sys_json_test::run(vector<wstring>& argvs) {
	test_string_buffer_json();
	test_string_file_buffer_json();

	test_byte_buffer_json();	
	test_byte_file_buffer_json();
}