#include "stdafx.h"

#include "mt_helper.h"
#include <limits>
using namespace std;

static float mt_float_eps = 0.000001f;
static double mt_double_eps = 0.00000000001;

f64 mt_helper::nan() {
	return std::numeric_limits<f64>::quiet_NaN();
}

f64 mt_helper::infinity() {
	return std::numeric_limits<f64>::infinity();
}

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

f64 mt_helper::e64() {
	return exp(1);
}

f32 mt_helper::e32() {
	return expf(1.0);
}

i8 mt_helper::max_i8() {
	return 127;
}

i8 mt_helper::min_i8() {
	return -128;
}

u8 mt_helper::max_u8() {
	return 255;
}

u8 mt_helper::min_u8() {
	return 0;
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