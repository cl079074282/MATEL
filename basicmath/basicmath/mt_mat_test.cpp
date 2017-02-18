#include "test.h"



static void test_mat_create() {
	mt_mat a(3, 3, mt_U8C1, mt_scalar(0));
	basiclog_info2(a);

	sys_test_equal(a, mt_mat(3, 3, mt_U8C1).set(0));

	a = mt_mat(3, 3, mt_U8C1).set_incremental(0);

	basiclog_info2(a);
	sys_test_equal(a, mt_mat_t<u8>(3, 3, 1).read(0, 1, 2, 3, 4, 5, 6, 7, 8));

	mt_mat b = a;

	basiclog_info2(b);

	basiclog_info2(mt_mat_t<u8>(2, 2, 2, 1).read(0, 1, 2, 3, 4, 5, 6, 7));

	vector<int> vec;
	vec.push_back(1);

	basiclog_info2(mt_mat_t<i32>::read(vec));
}

static void test_mat_add() {
	mt_mat a = mt_mat_t<u8>(3, 3, 1).read(0, 1, 2, 3, 4, 5, 6, 7, 8);
	mt_mat b = mt_mat_t<u8>(3, 3, 1).read(1, 1, 1, 1, 1, 1, 1, 1, 1);

	mt_mat c = a + b;
	sys_test_equal(c, mt_mat_t<u8>(3, 3, 1).read(1, 2, 3, 4, 5, 6, 7, 8, 9));

	a += 1;

	sys_test_equal(a, c);

	a = mt_mat_t<u8>(2, 2, 1).read(0, 1, 2, 3);
	b = a + a.flip(0);

	sys_test_equal(b, mt_mat_t<u8>(2, 2, 1).read(2, 4, 2, 4));

	a = mt_mat_t<f32>(3, 3, 1).read(0.0, 1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0);
	mt_mat negative_a = -a;

	basiclog_info2(negative_a);
	sys_test_equal(negative_a, mt_mat_t<f32>(3, 3, 1).read(0.0, 1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0) * -1);
}

static void test_mat_sub() {
	basiclog_info2(L"test_mat_sub");

	mt_mat a = mt_mat(3, 3, mt_U8C1).set_incremental(0);
	basiclog_info2(a);

	sys_test_equal(a.row(0), mt_mat_t<u8>(1, 3, 1).read(0, 1, 2));

	basiclog_info2(a.row(1));
	sys_test_equal(a.row(1), mt_mat_t<u8>(1, 3, 1).read(3, 4, 5));
	sys_test_equal(a.row(2), mt_mat_t<u8>(1, 3, 1).read(6, 7, 8));

	basiclog_info2(a.col(0));
	basiclog_info2(mt_mat_t<u8>(3, 1, 1).read(0, 3, 6));

	sys_test_equal(a.col(0), mt_mat_t<u8>(3, 1, 1).read(0, 3, 6));
	sys_test_equal(a.col(1), mt_mat_t<u8>(3, 1, 1).read(1, 4, 7));
	sys_test_equal(a.col(2), mt_mat_t<u8>(3, 1, 1).read(2, 5, 8));

	basiclog_info2(a.t().row(0));
	basiclog_info2(mt_mat_t<u8>(1, 3, 1).read(0, 3, 6));
	sys_test_equal(a.t().row(0), mt_mat_t<u8>(1, 3, 1).read(0, 3, 6));
	sys_test_equal(a.t().row(1), mt_mat_t<u8>(1, 3, 1).read(1, 4, 7));
	sys_test_equal(a.t().row(2), mt_mat_t<u8>(1, 3, 1).read(2, 5, 8));


	sys_test_equal(a.t().col(0), mt_mat_t<u8>(3, 1, 1).read(0, 1, 2));
	sys_test_equal(a.t().col(1), mt_mat_t<u8>(3, 1, 1).read(3, 4, 5));
	sys_test_equal(a.t().col(2), mt_mat_t<u8>(3, 1, 1).read(6, 7, 8));

	sys_test_equal(a.front(1), mt_mat_t<u8>(1, 3, 1).read(0, 1, 2));
	sys_test_equal(a.front(2), mt_mat_t<u8>(2, 3, 1).read(0, 1, 2, 3, 4, 5));
	sys_test_equal(a.front(3), mt_mat_t<u8>(3, 3, 1).read(0, 1, 2, 3, 4, 5, 6, 7, 8));

	sys_test_equal(a.back(1), mt_mat_t<u8>(1, 3, 1).read(6, 7, 8));
	sys_test_equal(a.back(2), mt_mat_t<u8>(2, 3, 1).read(3, 4, 5, 6, 7, 8));
	sys_test_equal(a.back(3), mt_mat_t<u8>(3, 3, 1).read(0, 1, 2, 3, 4, 5, 6, 7, 8));

	sys_test_equal(a.index(1), mt_mat_t<u8>(1, 3, 1).read(3, 4, 5));
	sys_test_equal(a.index(1, 1), mt_mat_t<u8>(3, 1, 1).read(1, 4, 7));


	
	mt_mat b = mt_mat(4, 4, mt_U8C1).set_incremental(0);
	basiclog_info2(b);

	mt_range ranges[2];
	ranges[0].m_start = 1;
	ranges[0].m_end = 3;

	ranges[1] = ranges[0];

	basiclog_info2(b.sub(2, ranges));
	basiclog_info2(b.sub(2, ranges).t().front(1));



	mt_mat c = mt_mat(3, 4, 4, mt_U8C1).set_incremental(0);
	basiclog_info2(c);

	basiclog_info2(c.flip(1));
	basiclog_info2(c.flip(1).swap_dim(1, 2).index(0).decrease_dim(0).row(1).t());

	//BASICLOG_INFO2(c.sub(1, 0));

	//BASICLOG_INFO2(c.sub(1, 0).decrease_dim(0).t());

	basiclog_info2(c.swap_dim(0, 2));
	basiclog_info2(c.swap_dim(0, 2).index(1).decrease_dim(0).flip(0).flip(1));
}

static void test_mat_flip() {
	basiclog_info2(L"test_mat_flip");

	mt_mat a = mt_mat(3, 3, mt_U8C1).set_incremental(0);

	basiclog_info2(a.flip(0).flip(1));
	basiclog_info2(a.flip(0).flip(1).t());

	basiclog_info2(a.flip(0).flip(1).index(1).t());


}

static void test_dim_channel() {
	mt_mat a = mt_mat(3, 3, mt_U8C1).set_incremental(0);

	basiclog_info2(a.last_dim_as_channel());

	basiclog_info2(a.last_dim_as_channel().channel_as_last_dim());
	sys_test_equal(a.last_dim_as_channel().channel_as_last_dim(), a);

	a = a.t().clone();

	basiclog_info2(a);
	basiclog_info2(a.last_dim_as_channel());

	basiclog_info2(a.last_dim_as_channel().channel_as_last_dim());
	sys_test_equal(a.last_dim_as_channel().channel_as_last_dim(), a);

	a = a.sub(mt_range(1, 3), 0).sub(mt_range(1, 3), 1);
	basiclog_info2(a.last_dim_as_channel());

	basiclog_info2(a.last_dim_as_channel().channel_as_last_dim());
	sys_test_equal(a.last_dim_as_channel().channel_as_last_dim(), a);

	a = a.flip(0);
	basiclog_info2(a);
	basiclog_info2(a.last_dim_as_channel().channel_as_last_dim());
	sys_test_equal(a.last_dim_as_channel().channel_as_last_dim(), a);
}

static void test_mat_t() {
	mt_mat a = mt_mat(4, 4, mt_U8C1).set_incremental(0);
	mt_mat sub_a = a.sub(1, 3, 0).sub(1, 3, 1);

	sys_test_equal(sub_a.t(), mt_mat_t<u8>(2, 2, 1).read(5, 9, 6, 10));

	mt_mat b = mt_mat(4, 4, mt_U8C1).set_incremental(0);
	
	
	mt_mat c = b.t();

	basiclog_info2(c);

	mt_mat sub_c = c.sub(1, 3, 0).sub(1, 3, 1);
	basiclog_info2(sub_c);

	sys_test_equal(sub_c, mt_mat_t<u8>(2, 2, 1).read(5, 9, 6, 10));
}

static void test_get_index() {
	mt_mat a = mt_mat(4, 4, mt_U8C1).set_incremental(0);
	basiclog_info2(a);

	vector<int> index;
	index.push_back(3);
	index.push_back(3);

	basiclog_info2(a.get_index(a.ptr<u8>(index, 0)));

	sys_test_equal(index, a.get_index(a.ptr<u8>(index, 0)));

	//a = a.flip(0);
	a = a.sub(mt_range(1, 3), 0).sub(mt_range(1, 3), 1);

	basiclog_info2(a);

	index[0] = 1;
	index[1] = 0;

	a = a.flip(0);

	basiclog_info2(a.get_index(a.ptr<u8>(index, 0)));

	sys_test_equal(index, a.get_index(a.ptr<u8>(index, 0)));
}

static void test_auto_derivative() {
	mt_mat a = mt_mat(4, 4, mt_U8C1).set_incremental(0);
	mt_mat b = mt_mat(2, 2, mt_U8C1).set_incremental(0);

	//mt_mat sub_a = a.sub(mt_range(1, 3), 0).sub(mt_range(1, 3), 1);
	mt_auto_derivative auto_derivative;
	a.attach(&auto_derivative);

	mt_mat sub_a = a.sub(mt_range(1, 3), 0).sub(mt_range(1, 3), 1);

	mt_mat c = sub_a + b;
	mt_mat d = c + sub_a;
	mt_mat e = c + d;

	mt_mat derivate_c_to_a = auto_derivative.derivate(a, e);

	basiclog_info2(derivate_c_to_a);

	//mt_mat gt_derivate_c_to_a = mt_mat().create_as(a).set(0);
	//gt_derivate_c_to_a.sub(1, 3, 0).sub(1, 3, 1).set(2);

	//basiclog_info2(gt_derivate_c_to_a);
	//sys_test_equal(derivate_c_to_a, gt_derivate_c_to_a);
}

static void test_mul() {
	//test normal mul

	mt_mat a = mt_mat(2, 2, mt_F32).set_incremental(0);
	mt_mat b = a.clone();

	mt_mat res = mt_mat_t<f32>(2, 2, 1).read(2.0, 3.0, 6.0, 11.0);

	basiclog_info2(a);
	basiclog_info2(b);

	mt_mat c = a.mul(b);

	basiclog_info2(c);
	basiclog_info2(res);

	sys_test_equal(c, res);

	//test mul on t
	a = mt_mat(2, 2, mt_F32).set_incremental(0);

	//test mul on t and sub and flip
	mt_mat e = mt_mat(3, 4, mt_F32).set_incremental(0);
	mt_mat f = e.sub(1, 3, 0).sub(1, 3, 1).t();

	basiclog_info2(f);

	basiclog_info2(a.flip_all_dim());
	c = a.flip_all_dim().mul(a.flip_all_dim());

	basiclog_info2(c);
	sys_test_equal(c, mt_mat_t<f32>(2, 2, 1).read(11.0, 6.0, 3.0, 2.0));

	mt_mat mat_331 = mt_mat(3, 3, mt_F32).set_incremental(0, true);
	mt_mat mat_333 = mt_mat(3, 3, mt_F32C3).set_incremental(0, true);
	basiclog_info2(mat_333.channel_at(1));

	basiclog_info2(mat_331.mul(mat_331));
	basiclog_info2(mat_333.channel_at(1).mul(mat_333.channel_at(2)));

	sys_test_equal(mat_333.channel_at(1).mul(mat_333.channel_at(2)), mat_331.mul(mat_331));

	
}

static void test_conv() {
	mt_mat mat_33 = mt_mat(3, 3, mt_F32).set_incremental(0);
	mt_mat mat_22 = mt_mat(2, 2, mt_F32).set_incremental(0);

	sys_test_equal(mat_33.conv(mat_22), mt_mat_t<f32>(2, 2, 1).read(5.0, 11.0, 23.0, 29.0));

	basiclog_info2(mat_33.t().conv(mat_22.t()));

	sys_test_equal(mat_33.t().conv(mat_22.t()), mt_mat_t<f32>(2, 2, 1).read(5.0, 23.0, 11.0, 29.0));

	mt_mat a = mt_mat(1, 3, mt_F32).set_incremental(0);
	mt_mat b = mt_mat(1, 2, mt_F32).set_incremental(0);

	basiclog_info2(a.decrease_dim(0).conv(b.decrease_dim(0), mt_Conv_Boundary_Type_Full));
	sys_test_equal(a.decrease_dim(0).conv(b.decrease_dim(0), mt_Conv_Boundary_Type_Full), mt_mat_t<f32>(1, 4, 1).read(0.000000, 0.000000, 1.000000, 2.000000).decrease_dim(0));
}

static void test_convert() {
	mt_mat a = mt_mat(3, 3, mt_U8).set_incremental(0);
	
	basiclog_info2(a.convert(mt_F32));

	sys_test_equal(a.convert(mt_F32), mt_mat(3, 3, mt_F32).set_incremental(0));

	mt_mat int_mat = mt_mat(3, 3, mt_S32).set_incremental(0);
}

static void test_mat_channel() {
	mt_mat mat_33 = mt_mat(3, 3, mt_F32C3).set_incremental(0, false);
	basiclog_info2(mat_33);

	mt_mat mat_channel = mat_33.channel_at(1);
	basiclog_info2(mat_channel);

	vector<mt_mat> channels;
	mat_33.split(channels);

	basiclog_info2(channels[0]);
	basiclog_info2(channels[1]);

	sys_test_equal(mt_mat_helper::merge_align_channel(channels), mat_33);
}

static void test_reshape() {
	mt_mat a = mt_mat(4, 4, mt_U8).set_incremental(0);
	mt_mat b = mt_mat(2, 8, mt_U8).set_incremental(0);

	sys_test_equal(a.reshape(2, 8), b);
	sys_test_equal(a.reshape(1, 16), b.reshape(1, 16));
}

static void test_repeat() {
	mt_mat a = mt_mat(2, 2, mt_U8).set_incremental(0);
	mt_mat b = mt_mat_t<u8>(2, 4, 1).read(0, 1, 0, 1, 2, 3, 2, 3);

	sys_test_equal(a.repeat(2, 1), b);

	basiclog_info2(a.repeat(2, 1));
	basiclog_info2(b);

	mt_mat c = a.increase_dim(0);
	basiclog_info2(c.repeat(2, 0));
}

static void test_save_mat() {
	mt_mat a = mt_mat(20, 20, mt_U16C3).set_incremental(0, sys_false);

	mt_mat_helper::save(L"test.txt", a);

	mt_mat b = mt_mat_helper::load(L"test.txt");

	sys_test_equal(a, b);

	mt_mat_helper::save(L"test.b", a, sys_false);

	b = mt_mat_helper::load(L"test.b", sys_false);

	sys_test_equal(a, b);


	//basiclog_info2(b);
}

static void test_pooling() {
	mt_mat a = mt_mat(4, 4, 4, mt_U16C3).set_incremental(0, sys_false);
	basiclog_info2(a);
	
	//test max pooling 
	i32 kernel_sizes[] = {2, 2, 2};
	i32 strides[] = {2, 2, 2};
	mt_mat b = a.pooling(mt_mat(), mt_Pooling_Type_Max, 3, kernel_sizes, strides);
	mt_mat gt_b = mt_mat_t<u16>(2, 2, 2, 3).read(63, 64, 65, 69, 70, 71, 87, 88, 89, 93, 94, 95, 159, 160, 161, 165, 166, 167, 183, 184, 185, 189, 190, 191);

	sys_test_equal(b, gt_b);

	basiclog_info2(b);

	mt_mat sub_a = a.sub(1, 3, 0).sub(1, 3, 1).sub(1, 3, 2);
	//basiclog_info2(sub_a);
	b = sub_a.pooling(mt_mat(), mt_Pooling_Type_Max, 3, kernel_sizes, strides);

	gt_b = mt_mat_t<u16>(1, 1, 1, 3).read(126, 127, 128);
	sys_test_equal(b, gt_b);


	i32 kernel_sizes_3[] = {3, 3, 3};
	i32 strides_3[] = {2, 2, 2};

	b = a.pooling(mt_mat(), mt_Pooling_Type_Max, 3, kernel_sizes_3, strides_3);
	basiclog_info2(b);

	a = mt_mat(4, 4, mt_F32C3).set_incremental(0, sys_false);
	basiclog_info2(a);
	b = a.pooling(mt_mat(), mt_Pooling_Type_Mean, 2, kernel_sizes, strides);
	basiclog_info2(b);

	gt_b = mt_mat_t<float>(2, 2, 3).read(7.500000, 8.5000000, 9.5000000, 13.5000000, 14.5000000, 17.8750000, 31.5000000, 32.5000000, 37.9687500, 37.5000000, 38.5000000, 48.9921875);

	sys_test_equal(b, gt_b);
} 

static void test_sub_stride() {
	mt_mat a = mt_mat(8, 8, mt_S32).set_incremental(0, sys_false);
	basiclog_info2(a);

	i32 strides[] = {2, 2};

	basiclog_info2(a.sub_stride(2, strides));

	sys_test_equal(a.sub(1, 7, 0).sub(1, 7, 1).t().flip_all_dim().sub_stride(2, strides), mt_mat_t<i32>(3, 3, 1).read(54, 38, 22, 52, 36, 20, 50, 34, 18));

	//basiclog_info2(a.sub(1, 7, 0).sub(1, 7, 1));
	//basiclog_info2(a.sub(1, 7, 0).sub(1, 7, 1).t());
	//basiclog_info2(a.sub(1, 7, 0).sub(1, 7, 1).t().flip_all_dim());
	//basiclog_info2(a.sub(1, 7, 0).sub(1, 7, 1).t().flip_all_dim().sub_stride(2, strides));
}

static void test_at_ptr() {
	mt_mat a = mt_mat(8, 8, mt_S32).set_incremental(0, sys_false);
	basiclog_info2(a);

	i32 value = a.at<i32>(1, 0);
	
	
	sys_test_equal(a.at<i32>(1, 0), 8);
	sys_test_equal(a.at<i32>(1, 0, 0), 8);

	sys_test_equal(*a.ptr<i32>(1, 0), 8);
	sys_test_equal(*a.ptr<i32>(1, 0, 0), 8);
}

static void test_eigen() {
	basiclog_debug2(L"test eigen:");
	// example from https://en.wikipedia.org/wiki/Jacobi_eigenvalue_algorithm
	// result:
	// e1 = 2585.253662
	// v0 = (0.0291933, -0.3287122, 0.7914113, -0.5145529)
	// e2 = 37.101521
	// v1 = (-0.1791862, 0.7419176, -0.1002286, -0.6382831)
	// e3 = 1.478052
	// v2 = (-0.5820814, 0.3704996, 0.5095764, 0.5140461)
	// e4 = 0.166633
	// v3 = (0.7926043, 0.4519261, 0.3224201, 0.2521646)

	mt_mat a = mt_mat_t<f32>(4, 4, 1).read(4.0f, -30.0, 60.0, -35.0, -30.0, 300.0, -675.0, 420.0, 60.0, -675.0, 1620.0, -1050.0, -35.0, 420.0, -1050.0, 700.0);
	mt_mat result_ev = mt_mat_t<f32>(1, 4, 1).read(2585.253662f, 37.101521, 1.478052, 0.166633);
	mt_mat result_ec = mt_mat_t<f32>(4, 4, 1).read(0.0291933f, -0.3287122, 0.7914113, -0.5145529, -0.1791862, 0.7419176, -0.1002286, -0.6382831, -0.5820814, 0.3704996, 0.5095764, 0.5140461, 0.7926043, 0.4519261, 0.3224201, 0.2521646);

	mt_mat ev, ec;
	a.eigen(ev, ec);

	sys_test_equal(ev.dim(), 2);
	sys_test_equal(ev.depth_channel(), a.depth_channel());
	sys_test_equal(ev.element_number(), a.size()[0]);
	sys_test_equal(ec.dim(), 2);
	sys_test_equal(ec.depth_channel(), a.depth_channel());
	sys_test_equal(ec.element_number(), a.element_number());
	sys_test_equal(ev, result_ev);
	sys_test_equal(ec, result_ec);
	
	basiclog_debug2(L"\nresult eigen values: " << ev);
	basiclog_debug2(L"\nresult eigen vectors: " << ec);
}

void mt_mat_test::run(vector<wstring>& argvs) {
	test_mat_create();
	test_mat_sub();
	test_mat_t();
	test_mat_flip();
	test_dim_channel();
	test_get_index();
	test_auto_derivative();
	test_mul();
	test_conv();
	test_mat_channel();
	test_convert();
	test_mat_add();
	test_repeat();
	test_reshape();
	test_save_mat();
	test_pooling();
	test_sub_stride();
	test_at_ptr();
	test_eigen();
}