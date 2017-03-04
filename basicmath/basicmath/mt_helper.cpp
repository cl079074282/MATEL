#include "stdafx.h"

static float mt_float_eps = 0.000001f;
static double mt_double_eps = 0.00000000001;

b8 mt_helper::is_nan(f64 val) {
	return _isnan(val) != 0;
}

b8 mt_helper::is_infinity(f64 val) {
	if (is_nan(val)) {
		return sys_false;
	}

	return _finite(val) == 0;
}

b8 mt_helper::is_number(f64 val) {
	return !is_nan(val) && !is_infinity(val);
}

void mt_helper::set_float_eps(float eps) {
	mt_float_eps = eps;
}

float mt_helper::get_float_eps() {
	return mt_float_eps;
}

void mt_helper::set_double_eps(double eps) {
	mt_double_eps = eps;
}

double mt_helper::get_double_eps() {
	return mt_double_eps;
}

int mt_helper::compare_float(float a, float b) {
	float value = a - b;

	if (abs(value) < mt_float_eps) {
		return 0;
	}

	float max_abs = max(abs(a), abs(b));

	if (abs(value) / max_abs < mt_float_eps) {
		return 0;
	}

	if (value > 0) {
		return 1;
	} else {
		return -1;
	}
}

int mt_helper::compare_double(double a, double b) {
	double value = a - b;

	if (abs(value) < mt_double_eps) {
		return 0;
	}

	double max_abs = max(abs(a), abs(b));

	if (abs(value) / max_abs < mt_double_eps) {
		return 0;
	}

	if (value > 0) {
		return 1;
	} else {
		return -1;
	}
}

i32 mt_helper::reverse_i32(i32 val) {
	unsigned char ch1, ch2, ch3, ch4;
	ch1 = val & 255;
	ch2 = (val >> 8) & 255;
	ch3 = (val >> 16) & 255;
	ch4 = (val >> 24) & 255;
	return((i32) ch1 << 24) + ((i32)ch2 << 16) + ((i32)ch3 << 8) + ch4;
}