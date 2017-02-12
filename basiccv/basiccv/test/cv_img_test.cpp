#include "test.h"


static void test_load_save() {
	mt_mat img = cv_img::load(L"G:/study_project/basiccv/resource/1.jpg");
	cv_img::save(L"G:/study_project/basiccv/resource/save_1.jpg", img);
}

static void test_resize() {
	mt_mat img = cv_img::load(L"G:/study_project/basiccv/resource/1.jpg");
	mt_mat small_img = cv_img::resize(img, mt_size(100, 100), cv_img::Inter_Type_Cubic);

	cv_img::save(L"G:/study_project/basiccv/resource/small_1.jpg", small_img);
}

void cv_img_test::run(vector<wstring>& argvs) {
	test_load_save();
	test_resize();
}