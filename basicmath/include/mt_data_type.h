#pragma once

#include <limits>

namespace basicmath {
	enum mt_Depth {
		mt_U8,
		mt_S8,
		mt_U16,
		mt_S16,
		mt_U32,
		mt_S32,
		mt_U64,
		mt_S64,
		mt_F32,
		mt_F64,

		mt_User,
	};

	enum mt_Memory_Type {
		mt_Memory_Type_CPU,
		mt_Memory_Type_GPU,
		mt_Memory_Type_Distributed_CPU,
		mt_Memory_Type_Distributed_GPU,
	};

	static const int mt_Mat_Normal_Support_Dim = 4;
	static const int mt_Depth_Size[] = {1, 1, 2, 2, 4, 4, 8, 8, 4, 8};

	static int mt_Depth_Mask = 0x0000ffff;
	static int mt_Channel_Mask = 0xffff0000;

	static int mt_get_depth(int depth_channel) {
		return depth_channel & mt_Depth_Mask;
	}

	static int mt_get_channel(int device_channel) {
		return ((device_channel & mt_Channel_Mask) >> 16) + 1;
	}

	static int mt_make_depth_channel(int depth, int channel) {
		return (channel - 1) << 16 | depth;
	}

	static int mt_get_depth_size(int depth) {
		return mt_Depth_Size[depth];
	}

	static int mt_get_depth_channel_size(int depth_channel) {
		return mt_get_depth_size(mt_get_depth(depth_channel)) * mt_get_channel(depth_channel);
	}

	static const int mt_U8C1 = mt_make_depth_channel(mt_U8, 1);
	static const int mt_S8C1 = mt_make_depth_channel(mt_S8, 1);
	static const int mt_U16C1 = mt_make_depth_channel(mt_U16, 1);
	static const int mt_S16C1 = mt_make_depth_channel(mt_S16, 1);
	static const int mt_U32C1 = mt_make_depth_channel(mt_U32, 1);
	static const int mt_S32C1 = mt_make_depth_channel(mt_S32, 1);
	static const int mt_U64C1 = mt_make_depth_channel(mt_U64, 1);
	static const int mt_S64C1 = mt_make_depth_channel(mt_S64, 1);
	static const int mt_F32C1 = mt_make_depth_channel(mt_F32, 1);
	static const int mt_F64C1 = mt_make_depth_channel(mt_F64, 1);

	static const int mt_U8C3 = mt_make_depth_channel(mt_U8, 3);
	static const int mt_S8C3 = mt_make_depth_channel(mt_S8, 3);
	static const int mt_U16C3 = mt_make_depth_channel(mt_U16, 3);
	static const int mt_S16C3 = mt_make_depth_channel(mt_S16, 3);
	static const int mt_U32C3 = mt_make_depth_channel(mt_U32, 3);
	static const int mt_S32C3 = mt_make_depth_channel(mt_S32, 3);
	static const int mt_U64C3 = mt_make_depth_channel(mt_U64, 3);
	static const int mt_S64C3 = mt_make_depth_channel(mt_S64, 3);
	static const int mt_F32C3 = mt_make_depth_channel(mt_F32, 3);
	static const int mt_F64C3 = mt_make_depth_channel(mt_F64, 3);

	enum mt_Conv_Boundary_Type {
		mt_Conv_Boundary_Type_Valid,
		mt_Conv_Boundary_Type_Full,
		mt_Conv_Boundary_Type_Same,
	};

	static const wstring mt_Conv_Boundary_Type_Descriptions[] = {L"valid", L"full", L"same"};

	enum mt_Pooling_Type{
		mt_Pooling_Type_Mean,
		mt_Pooling_Type_Max,
		mt_Pooling_Type_Min,
		mt_Pooling_Type_Sum,
		mt_Pooling_Type_First_Value,		
	};

	static const wstring mt_Pooling_Type_Descriptions[] = {L"mean", L"max", L"min", L"first_value"};

	enum mt_Dist_Type {
		mt_Dist_Type_L1,
		mt_Dist_Type_L2,
	};

	static const f64 mt_PI = 3.1415926535898;
	static const f64 mt_E = exp(1);
	static const f64 mt_Nan = std::numeric_limits<f64>::quiet_NaN();
	static const f64 mt_Infinity = std::numeric_limits<f64>::infinity();
	static const i8 mt_S8_Max = 127;
	static const i8 mt_S8_Min = -128;
	static const u8 mt_U8_Max = 255;
	static const u8 mt_U8_Min = 0;

	enum mt_Activate_Type {
		mt_Activate_Type_Sigmoid,
		mt_Activate_Type_Linear,
		mt_Activate_Type_Tanh,
		mt_Activate_Type_Softmax,
		mt_Activate_Type_Relu,
	};

	enum mt_Loss_Type {
		mt_Loss_Type_0_1,
		mt_Loss_Type_Quadratic,
		mt_Loss_Type_Logarithmic,
	};
}