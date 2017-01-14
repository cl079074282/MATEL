#include "stdafx.h"

#include "mt_helper.h"

static float mt_float_eps = 0.000001f;
static double mt_double_eps = 0.00000000001;

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