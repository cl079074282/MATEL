#pragma once



#define BASICMATH_MKL





#if defined BASICMATH_MKL
#include <mkl.h>
#elif defined 
#endif

#include <basicsys.h>
using namespace basicsys;

#include <basiclog.h>
using namespace basiclog;

#include "mt_data_type.h"

#include "mt_range_t.h"
#include "mt_rect_t.h"
#include "mt_scalar_t.h"
#include <stdio.h>
#include <stdarg.h>

#include <vector>
#include <map>
using namespace std;

using namespace basicmath;

#include "mt_point_t.h"
#include "mt_size_t.h"
#include "mt_rect_t.h"
#include "mt_helper.h"
#include "mt_mat.h"
#include "mt_mat_t.h"

#include "mt_mat_helper.h"
#include "mt_array_iteration.h"


