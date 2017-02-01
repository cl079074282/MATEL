#include "stdafx.h"
#include "mt_mat.h"
#include "mt_mat_helper.h"
#include "mt_auto_derivative.h"
#include "mt_mat_cache.h"

mt_mat::mt_mat() 
	: m_data(NULL)
	, m_modified_number(NULL)
	, m_dynamic_size_steps(NULL)
	, m_dynamic_size_step_size(0)
	, m_reference(NULL)
	, m_dims(0)
	, m_auto_derivative(NULL)
	, m_depth_channel(mt_User)
	, m_shared_memory(NULL){

}

mt_mat::mt_mat(int cols, int depth, const mt_scalar init_value) 
	: m_data(NULL)
	, m_modified_number(NULL)
	, m_dynamic_size_steps(NULL)
	, m_dynamic_size_step_size(0)
	, m_reference(NULL)
	, m_dims(0)
	, m_auto_derivative(NULL)
	, m_depth_channel(mt_User)
	, m_shared_memory(NULL) {

		int sizes[] = {cols};
		*this = s_mat_cache.get(1, sizes, depth);
		set(init_value);
}

mt_mat::mt_mat(int row, int col, int depth, const mt_scalar init_value) 
	: m_data(NULL)
	, m_modified_number(NULL)
	, m_dynamic_size_steps(NULL)
	, m_dynamic_size_step_size(0)
	, m_reference(NULL)
	, m_dims(0)
	, m_auto_derivative(NULL)
	, m_depth_channel(mt_User)
	, m_shared_memory(NULL) {

		int sizes[] = {row, col};
		*this = s_mat_cache.get(2, sizes, depth);
		set(init_value);
}

mt_mat::mt_mat(int plane, int row, int col, int depth, const mt_scalar init_value) 
	: m_data(NULL)
	, m_modified_number(NULL)
	, m_dynamic_size_steps(NULL)
	, m_dynamic_size_step_size(0)
	, m_reference(NULL)
	, m_dims(0)
	, m_auto_derivative(NULL)
	, m_depth_channel(mt_User)
	, m_shared_memory(NULL) {
		int sizes[] = {plane, row, col};
		*this = s_mat_cache.get(3, sizes, depth);

		set(init_value);
}

mt_mat::mt_mat(int dims, const int* sizes, int depth, const mt_scalar init_value) 
	: m_data(NULL)
	, m_modified_number(NULL)
	, m_dynamic_size_steps(NULL)
	, m_dynamic_size_step_size(0)
	, m_reference(NULL)
	, m_dims(0)
	, m_auto_derivative(NULL)
	, m_depth_channel(mt_User)
	, m_shared_memory(NULL) {
		*this = s_mat_cache.get(dims, sizes, depth);
		set(init_value);
}

mt_mat::mt_mat(const vector<i32>& sizes, int depth, const mt_scalar init_value) 
	: m_data(NULL)
	, m_modified_number(NULL)
	, m_dynamic_size_steps(NULL)
	, m_dynamic_size_step_size(0)
	, m_reference(NULL)
	, m_dims(0)
	, m_auto_derivative(NULL)
	, m_depth_channel(mt_User)
	, m_shared_memory(NULL) {
		*this = s_mat_cache.get(sizes, depth);
		set(init_value);
}

mt_mat::mt_mat(int cols, int depth, u8* data, const int* steps) 
	: m_data(NULL)
	, m_modified_number(NULL)
	, m_dynamic_size_steps(NULL)
	, m_dynamic_size_step_size(0)
	, m_reference(NULL)
	, m_dims(0)
	, m_auto_derivative(NULL)
	, m_depth_channel(mt_User)
	, m_shared_memory(NULL) {
		int sizes[] = {cols};
		mt_mat::mt_mat(1, sizes, depth, data, steps);
}

mt_mat::mt_mat(int row, int col, int depth, u8* data, const int* steps) 
	: m_data(NULL)
	, m_modified_number(NULL)
	, m_dynamic_size_steps(NULL)
	, m_dynamic_size_step_size(0)
	, m_reference(NULL)
	, m_dims(0)
	, m_auto_derivative(NULL)
	, m_depth_channel(mt_User)
	, m_shared_memory(NULL) {
		int sizes[] = {row, col};
		mt_mat::mt_mat(2, sizes, depth, data, steps);
}

mt_mat::mt_mat(int plane, int row, int col, int depth, u8* data, const int* steps) 
	: m_data(NULL)
	, m_modified_number(NULL)
	, m_dynamic_size_steps(NULL)
	, m_dynamic_size_step_size(0)
	, m_reference(NULL)
	, m_dims(0)
	, m_auto_derivative(NULL)
	, m_depth_channel(mt_User)
	, m_shared_memory(NULL) {
		int sizes[] = {plane, row, col};
		mt_mat::mt_mat(3, sizes, depth, data, steps);
}

mt_mat::mt_mat(i32 dims, const i32* sizes, i32 depth_channel, u8* data, const i32* steps) 
	: m_data(NULL)
	, m_modified_number(NULL)
	, m_dynamic_size_steps(NULL)
	, m_dynamic_size_step_size(0)
	, m_reference(NULL)
	, m_dims(0)
	, m_auto_derivative(NULL)
	, m_depth_channel(mt_User)
	, m_shared_memory(NULL) {
		m_dims = dims;
		m_depth_channel = depth_channel;

		if (m_dims > 4) {
			m_dynamic_size_step_size = m_dims;
			m_dynamic_size_steps = new int[m_dynamic_size_step_size * 2];
		}

		for (int i = 0; i < dims; ++i) {
			size()[i] = sizes[i];
		}

		m_data = data;
		m_shared_memory = NULL;
		m_reference = NULL;

		if (steps == NULL) {
			fill_auto_step();
		} else {
			for (int i = 0; i < m_dims; ++i) {
				step()[i] = steps[i];
			}
		}
}

mt_mat::mt_mat(const vector<i32>& sizes, i32 depth_channel, u8* data, const vector<i32>& steps) 
	: m_data(NULL)
	, m_modified_number(NULL)
	, m_dynamic_size_steps(NULL)
	, m_dynamic_size_step_size(0)
	, m_reference(NULL)
	, m_dims(0)
	, m_auto_derivative(NULL)
	, m_depth_channel(mt_User)
	, m_shared_memory(NULL) {
		mt_mat::mt_mat((i32)sizes.size(), &sizes[0], depth_channel, data, steps.empty() ? NULL : &steps[0]);
}

mt_mat::mt_mat(const mt_mat& other, Construct_Type type, const mt_scalar init_value)
	: m_data(NULL)
	, m_modified_number(NULL)
	, m_reference(NULL)
	, m_dims(0)
	, m_auto_derivative(NULL)
	, m_dynamic_size_step_size(0)
	, m_dynamic_size_steps(NULL)
	, m_depth_channel(mt_User)
	, m_shared_memory(NULL) {

		if (type == Construct_Type_Operator_Equal) {
			*this = other;
		} else {
			*this = s_mat_cache.get(other.dim(), other.size(), other.depth_channel());
			set(init_value);
		}
}

mt_mat::mt_mat(const mt_mat& other, i32 dims, const mt_range* ranges) 
	: m_data(NULL)
	, m_modified_number(NULL)
	, m_reference(NULL)
	, m_dims(0)
	, m_auto_derivative(NULL)
	, m_dynamic_size_step_size(0)
	, m_dynamic_size_steps(NULL)
	, m_depth_channel(mt_User)
	, m_shared_memory(NULL) {
		*this = other.sub(dims, ranges);
}

mt_mat::mt_mat(const mt_mat& other, const vector<mt_range>& ranges) 
	: m_data(NULL)
	, m_modified_number(NULL)
	, m_reference(NULL)
	, m_dims(0)
	, m_auto_derivative(NULL)
	, m_dynamic_size_step_size(0)
	, m_dynamic_size_steps(NULL)
	, m_depth_channel(mt_User)
	, m_shared_memory(NULL) {
		*this = other.sub(ranges);
}

mt_mat::mt_mat(const mt_mat& other, const mt_range& range, int dim) 
	: m_data(NULL)
	, m_modified_number(NULL)
	, m_reference(NULL)
	, m_dims(0)
	, m_auto_derivative(NULL)
	, m_dynamic_size_step_size(0)
	, m_dynamic_size_steps(NULL)
	, m_depth_channel(mt_User)
	, m_shared_memory(NULL) {
		*this = other.sub(range, dim);
}

mt_mat::mt_mat(const mt_mat& other, const mt_rect& roi) 
	: m_data(NULL)
	, m_modified_number(NULL)
	, m_reference(NULL)
	, m_dims(0)
	, m_auto_derivative(NULL)
	, m_dynamic_size_step_size(0)
	, m_dynamic_size_steps(NULL)
	, m_depth_channel(mt_User)
	, m_shared_memory(NULL) {
		*this = other.sub(roi);
}

mt_mat::~mt_mat() {
	try_deallocate();

	if (m_dynamic_size_steps != 0) {
		delete[] m_dynamic_size_steps;
	}
}

mt_mat mt_mat::derivative(const mt_mat& other) const {
	return m_auto_derivative->derivate(other, *this);
}

b8 mt_mat::empty() const {
	return m_data == NULL;
}

mt_mat& mt_mat::set_eye(b8 left_high) {
	basiclog_assert2(dim() == 2);
	on_vaule_changed();

}

bool mt_mat::operator==(const mt_mat& other) const {
	if (other.depth_channel() != depth_channel()) {
		return false;
	}

	for (int c = 0; c < channel(); ++c) {
		if (size()[c] != other.size()[c]) {
			return false;
		}
	}

	if (depth() == mt_F32 || depth() == mt_F64) {
		mt_array_element_const_iterator other_iter(other);
		mt_array_element_const_iterator cur_iter(*this);

		for (;;) {
			const u8* ptr_src = other_iter.data();
			const u8* ptr_dst = cur_iter.data();

			if (ptr_src == NULL) {
				break;
			}

			if (depth() == mt_F32) {
				const float* ptr_float_src = (const float*)ptr_src;
				const float* ptr_float_dst = (const float*)ptr_dst;

				for (int c = 0; c < channel(); ++c) {
					if (mt_helper::compare_float(ptr_float_src[c], ptr_float_dst[c]) != 0) {
						return false;
					}
				}
			} else {
				const double* ptr_float_src = (const double*)ptr_src;
				const double* ptr_float_dst = (const double*)ptr_dst;

				for (int c = 0; c < channel(); ++c) {
					if (mt_helper::compare_double(ptr_float_src[c], ptr_float_dst[c]) != 0) {
						return false;
					}
				}
			}
		}

		return true;
	} else {
		return mt_array_iteration::array_cmp(other.data(), data(), dim(), size(), other.step(), step(), element_size()) == 0;
	}
}

bool mt_mat::operator!=(const mt_mat& other) const {
	return !(*this == other);
}

bool mt_mat::is_memory_shared(const mt_mat& other) const {
	return m_shared_memory == other.m_shared_memory;
}

bool mt_mat::is_same(const mt_mat& other) const {
	if (m_shared_memory != other.m_shared_memory) {
		return false;
	}

	if (m_data != other.m_data) {
		return false;
	}

	if (m_dims != other.m_dims) {
		return false;
	}

	for (int i = 0; i < m_dims; ++i) {
		if (size()[i] != other.size()[i] || step()[i] != other.step()[i]) {
			return false;
		}
	}

	return true;
}

mt_mat& mt_mat::set_incremental(double value, b8 same_value_for_multi_channel) {
	on_vaule_changed();
	mt_array_memory_block_iterator iter(*this);

	basicmath_mat_request_memory(f64, channel_buffer, channel());

	for (;;) {
		u8* ptr_src = iter.data();

		if (ptr_src == NULL) {
			break;
		}

		for (int i = 0; i < iter.block_element_number(); ++i) {
			if (same_value_for_multi_channel) {
				mt_mat_helper::set_data(ptr_src, m_depth_channel, &value, 0);
				++value;
			} else {
				for (int c = 0; c < channel(); ++c) {
					channel_buffer[c] = value++;
				}

				mt_mat_helper::set_data(ptr_src, m_depth_channel, channel_buffer, channel());
			}

			ptr_src += iter.block_element_step();
		}
	}

	basicmath_mat_release(channel_buffer);
	return *this;
}

mt_mat& mt_mat::set(const mt_mat& other) {
	on_vaule_changed();
	basiclog_assert2(is_same_size(other));
	
	if (depth() == other.depth()) {
		mt_array_iteration::array_copy(data(), other.data(), other.dim(), other.size(), step(), other.step(), other.element_size());
	} else {
		mt_array_element_const_iterator other_iter(other);
		mt_array_element_iterator cur_iter(*this);

		vector<double> values;

		for (;;) {
			const u8* ptr_other = other_iter.data();
			u8* ptr_cur = cur_iter.data();

			if (ptr_other == NULL) {
				break;
			}

			mt_mat_helper::get_data(values, ptr_other, depth_channel());
			mt_mat_helper::set_data(ptr_cur, depth_channel(), &values[0], (i32)values.size());
		}
	}

	return *this;
}

mt_mat& mt_mat::set(double value) {
	on_vaule_changed();
	mt_array_memory_block_iterator iter(*this);

	for (;;) {
		u8* ptr_src = iter.memory_start();

		if (ptr_src == NULL) {
			break;
		}

		if (mt_get_depth_size(depth()) == 1) {
			u8 temp_value = (u8)value;
			memset(ptr_src, temp_value, iter.block_element_number());
		} else {
			for (int i = 0; i < iter.block_element_number(); ++i) {
				mt_mat_helper::set_data(ptr_src, m_depth_channel, &value, 0);
				ptr_src += abs(iter.block_element_step());
			}
		}
	}

	++*m_modified_number;
	return *this;
}

mt_mat& mt_mat::set(const mt_scalar& value) {
	vector<double> temp_value;
	temp_value.resize(4);
	temp_value[0] = value[0];
	temp_value[1] = value[1];
	temp_value[2] = value[2];
	temp_value[3] = value[3];

	return set(temp_value);
}

mt_mat& mt_mat::set(const vector<double>& value) {
	on_vaule_changed();
	mt_array_memory_block_iterator iter(*this);

	for (;;) {
		u8* ptr_src = iter.memory_start();

		if (ptr_src == NULL) {
			break;
		}

		for (int i = 0; i < iter.block_element_number(); ++i) {
			mt_mat_helper::set_data(ptr_src, m_depth_channel, &value[0], (int)value.size());
			ptr_src += abs(iter.block_element_step());
		}
	}

	++*m_modified_number;
	return *this;
}

mt_mat& mt_mat::set(double value, basicsys::i32 row, basicsys::i32 col) {
	int dims = 2;
	int indexs[] = {row, col};

	return set(value, dims, indexs);
}

mt_mat& mt_mat::set(double value, basicsys::i32 plane, basicsys::i32 row, basicsys::i32 col) {
	int dims = 3;
	int indexs[] = {plane, row, col};

	return set(value, dims, indexs);
}

mt_mat& mt_mat::set(double value, basicsys::i32 dims, const basicsys::i32* indexs) {
	u8* ptr_data = ptr<u8>(dims, indexs, 0);
	mt_mat_helper::set_data(ptr_data, depth_channel(), &value, 0);

	return *this;
}

mt_mat& mt_mat::set(const mt_scalar& value, basicsys::i32 row, basicsys::i32 col) {
	int dims = 2;
	int indexs[] = {row, col};

	return set(value, dims, indexs);
}

mt_mat& mt_mat::set(const mt_scalar& value, basicsys::i32 plane, basicsys::i32 row, basicsys::i32 col) {
	int dims = 3;
	int indexs[] = {plane, row, col};

	return set(value, dims, indexs);
}

mt_mat& mt_mat::set(const mt_scalar& value, basicsys::i32 dims, const basicsys::i32* indexs) {
	on_vaule_changed();

	double values[] = {value[0], value[1], value[2], value[3]};

	u8* ptr_data = ptr<u8>(dims, indexs, 0);
	mt_mat_helper::set_data(ptr_data, depth_channel(), values, 4);

	return *this;
}

mt_mat& mt_mat::set(const double* values, basicsys::i32 dims, const basicsys::i32* indexs) {
	on_vaule_changed();

	u8* ptr_data = ptr<u8>(dims, indexs, 0);
	mt_mat_helper::set_data(ptr_data, depth_channel(), values, dims);

	return *this;
}

void mt_mat::get(double* values, basicsys::i32 dims, const basicsys::i32* indexs) const {
	const u8* ptr_data = ptr<u8>(dims, indexs, 0);
	mt_mat_helper::get_data(values, ptr_data, depth_channel());
}

mt_mat mt_mat::clone() const {
	mt_mat res(m_dims, size(), m_depth_channel);
	res.set(*this);
	return res;
}

mt_mat mt_mat::convert(int depth) const {
	mt_mat res(dim(), size(), mt_make_depth_channel(depth, channel()));

	mt_array_element_const_iterator cur_iter(*this);
	mt_array_element_iterator other_iter(res);

	vector<double> values;

	for (;;) {
		const u8* ptr_cur = cur_iter.data();
		u8* ptr_other = other_iter.data();

		if (ptr_cur == NULL) {
			break;
		}

		mt_mat_helper::get_data(values, ptr_cur, depth_channel());
		mt_mat_helper::set_data(ptr_other, res.depth_channel(), &values[0], (i32)values.size());
	}

	return res;
}

void mt_mat::operator =(const mt_mat& other) {
	if (this == &other) {
		return;
	}

	try_deallocate();

	if (other.m_dims > 4 && m_dynamic_size_step_size < other.m_dims) {
		if (m_dynamic_size_steps != NULL) {
			delete[] m_dynamic_size_steps;
		}

		m_dynamic_size_step_size = other.m_dims;
	}

	m_dims = other.m_dims;

	for (int i = 0; i < other.m_dims; ++i) {
		step()[i] = other.step()[i];
		size()[i] = other.size()[i];
	}

	m_depth_channel = other.m_depth_channel;
	m_data = other.m_data;
	m_modified_number = other.m_modified_number;
	m_shared_memory = other.m_shared_memory;
	m_reference = other.m_reference;
	m_auto_derivative = other.m_auto_derivative;
	m_reference = other.m_reference;

	if (NULL != m_reference) {
		++*m_reference;
	}
}

bool mt_mat::is_same_size(const mt_mat& other) const {
	if (m_dims == other.m_dims && channel() == other.channel()) {
		bool same_size = true;
		for (int i = 0; i < m_dims; ++i) {
			if (size()[i] != other.size()[i]) {
				same_size = false;
				break;
			}
		}

		if (same_size) {
			return true;
		}
	}

	return false;
}

mt_mat& mt_mat::create_imp(int dims, const int* sizes, int depth_channel) {
	m_depth_channel = depth_channel;

	try_deallocate();

	if (dims > 4 && m_dynamic_size_step_size < dims) {

		if (m_dynamic_size_steps != NULL) {
			delete[] m_dynamic_size_steps;
		}

		m_dynamic_size_step_size = dims;
		m_dynamic_size_steps = new int[m_dynamic_size_step_size * 2];
	}

	m_dims = dims;

	int data_size = element_size();
	for (int i = 0; i < m_dims; ++i) {
		size()[i] = sizes[i];
		data_size *= sizes[i];
	}

	m_data = new u8[data_size];
	m_shared_memory = m_data;
	m_reference = new int;
	*m_reference = 1;
	m_modified_number = new int;
	*m_modified_number = 0;

	fill_auto_step();

	return *this;
}

int mt_mat::element_size() const {
	return mt_get_depth_channel_size(m_depth_channel);
}

int mt_mat::element_channel_size() const {
	return mt_get_depth_size(depth());
}

u8* mt_mat::memory_data() {
	u8* ptr_data = data();

	for (int i = 0; i < dim(); ++i) {
		if (step()[i] < 0) {
			ptr_data += (size()[i] - 1) * step()[i];
		}
	}

	return ptr_data;
}

const u8* mt_mat::memory_data() const {
	return (const_cast<mt_mat*>(this))->memory_data();
}

int mt_mat::element_number() const {
	int res = 1;

	for (int i = 0; i < m_dims; ++i) {
		res *= size()[i];
	}

	return res;
}

bool mt_mat::is_min_abs_step_equal_element_size() const {
	return element_size() == mt_helper::compute_abs_min(step(), dim());
}

bool mt_mat::is_continuous() const {
	return mt_array_iteration::get_continuous_dim(dim(), size(), step(), element_channel_size()) == 0;
}

void mt_mat::try_deallocate() {
	if (m_reference != NULL) {
		--*m_reference;

		if (*m_reference == 0) {
			delete m_reference;
			delete[] m_shared_memory;

			basiclog_info2(L"release memory");
		}
	}
}

mt_mat mt_mat::increase_dim(int added_dim) const {
	int splited_sizes[2];

	if (added_dim == m_dims) {
		added_dim -= 1;
		splited_sizes[0] = size()[m_dims - 1];
		splited_sizes[1] = 1;
	} else {
		splited_sizes[0] = 1;
		splited_sizes[1] = size()[added_dim];
	}

	return mt_mat::split_dim(added_dim, 2, splited_sizes);
}

mt_mat mt_mat::decrease_dim(int deleted_dim) const {
	basiclog_assert_message2(size()[deleted_dim] == 1, L"Only the dimension of size is 1 can be deleted!");

	if (deleted_dim == 0) {
		return combine_dim(0, 2);
	} else {
		return combine_dim(deleted_dim - 1, 2);
	}
}

mt_mat mt_mat::reshape(i32 rows, i32 cols) const {
	i32 sizes[] = {rows, cols};
	return reshape(2, sizes);
}

mt_mat mt_mat::reshape(i32 planes, i32 rows, i32 cols) const {
	i32 sizes[] = {planes, rows, cols};
	return reshape(3, sizes);
}

mt_mat mt_mat::reshape(const vector<int>& sizes) const {
	return reshape((i32)sizes.size(), &sizes[0]);
}

mt_mat mt_mat::reshape(int dims, const int* sizes) const {
	if (!is_continuous()) {
		basiclog_warning(basiclog_performance_warning, L"current mat is not continuous, hence we need to clone one mat to reshape it!");
		return clone().reshape(dims, sizes);
	}

	i32 current_total_size = mt_helper::mutiply<i32>(size(), size() + dim());
	i32 reshape_total_size = mt_helper::mutiply<i32>(sizes, sizes + dims);

	basiclog_assert2(current_total_size * reshape_total_size);

	mt_mat res;
	res.m_dims = dims;

	if (res.m_dims > 4 && m_dynamic_size_step_size < res.m_dims) {
		res.m_dynamic_size_step_size = res.m_dims;
		res.m_dynamic_size_steps = new int[m_dynamic_size_step_size * 2];
	}

	for (int i = 0; i < res.m_dims; ++i) {
		res.size()[i] = sizes[i];
	}

	++*m_reference;
	res.m_reference = m_reference;

	res.m_data = m_data;
	res.m_depth_channel = m_depth_channel;
	res.m_shared_memory = m_shared_memory;

	res.fill_auto_step();

	return res;
}

mt_mat mt_mat::split_dim(int dim, int splited_dims, int* splited_sizes) const {
	mt_mat res;
	res.m_dims = m_dims + splited_dims - 1;

	if (res.m_dims > mt_Mat_Normal_Support_Dim) {
		res.m_dynamic_size_step_size = res.m_dims;
		res.m_dynamic_size_steps = new int[m_dynamic_size_step_size * 2];
	}

	for (int i = 0; i < splited_dims; ++i) {
		res.size()[i + dim] = splited_sizes[i];
	}

	res.step()[dim + splited_dims - 1] = step()[dim];
	for (int i = splited_dims - 2; i >= 0; --i) {
		res.step()[i + dim] = res.step()[i + dim + 1] * res.size()[i + dim + 1];
	}

	for (int i = 0; i < dim; ++i) {
		res.size()[i] = size()[i];
		res.step()[i] = step()[i];
	}

	for (int i = res.m_dims - 1; i > dim + splited_dims - 1; --i) {
		res.size()[i] = size()[i - splited_dims + 1];
		res.step()[i] = step()[i - splited_dims + 1];
	}

	++*m_reference;
	res.m_reference = m_reference;

	res.m_depth_channel = m_depth_channel;
	res.m_data = m_data;
	res.m_shared_memory = m_shared_memory;

	return res;
}

mt_mat mt_mat::combine_dim(int combined_dim_start, int combined_dim_count) const {
	bool order_dim = true;

	for (int i = 0; i < combined_dim_count - 1; ++i) {
		if (step()[combined_dim_start + i] < step()[i + combined_dim_start + 1]) {
			order_dim = false;
			break;
		}
	}

	if (!order_dim) {
		basiclog_warning2(L"combine dim for a mat with unordered dim, this will reduce the performance! You should better input an ordered mat");
		return this->clone().combine_dim(combined_dim_start, combined_dim_count);
	}

	mt_mat res;

	res.m_dims = m_dims - combined_dim_count + 1;

	for (int i = 0; i < combined_dim_start; ++i) {
		res.size()[i] = size()[i];
		res.step()[i] = step()[i];
	}

	res.size()[combined_dim_start] = size()[combined_dim_start];

	for (int i = 1; i < combined_dim_count; ++i) {
		res.size()[combined_dim_start] *= size()[combined_dim_start + i];
	}

	res.step()[combined_dim_start] = step()[combined_dim_start + combined_dim_count - 1];

	for (int i = combined_dim_start + 1; i < res.m_dims; ++i) {
		res.size()[i] = size()[i + combined_dim_count - 1];
		res.step()[i] = step()[i + combined_dim_count - 1];
	}

	++*m_reference;
	res.m_depth_channel = m_depth_channel;
	res.m_reference = m_reference;
	res.m_data = m_data;
	res.m_shared_memory = m_shared_memory;

	return res;
}

mt_mat mt_mat::repeat(i32 nsize, i32 dim) const {
	basicmath_mat_request_memory(i32, nsizes, this->dim());

	for (i32 i = 0; i < this->dim(); ++i) {
		nsizes[i] = 1;
	}

	nsizes[dim] = nsize;
	
	return repeat(this->dim(), nsizes);

	basicmath_mat_release(nsizes);
}

mt_mat mt_mat::repeat(const vector<i32>& nsizes) const {
	return repeat((i32)nsizes.size(), &nsizes[0]);
}

mt_mat mt_mat::repeat(i32 dims, const i32* nsizes) const {
	basiclog_assert2(nsizes != NULL);
	basiclog_assert2(dims == dim());

	basicmath_mat_request_memory(i32, dst_sizes, dim());

	b8 size_changed = sys_false;

	for (i32 i = 0; i < dim(); ++i) {
		dst_sizes[i] = size()[i] * nsizes[i];

		if (nsizes[i] > 1) {
			size_changed = sys_true;
		}
	}

	mt_mat res(dim(), dst_sizes, depth_channel());

	if (size_changed) {
		mt_array_index_iterator iter(dim(), nsizes);

		vector<mt_range> start_ranges;
		start_ranges.resize(dim());

		while (iter.next()) {

			for (i32 i = 0; i < dim(); ++i) {
				start_ranges[i].m_start = iter.position()[i] * size()[i];
				start_ranges[i].m_end = start_ranges[i].m_start + size()[i];
			}

			mt_mat copy_dst = res.sub(start_ranges);
			copy_dst.set(*this);
		}
	} else {
		res.set(*this);
	}

	basicmath_mat_release(dst_sizes);

	return res;
}

mt_mat mt_mat::t() const {
	basiclog_assert2(m_dims == 2);

	return swap_dim(0, 1);
}

mt_mat mt_mat::swap_dim(int dim_a, int dim_b) const {
	mt_mat res = *this;

	swap(res.size()[dim_a], res.size()[dim_b]);
	swap(res.step()[dim_a], res.step()[dim_b]);

	return res;
}

void mt_mat::fill_auto_step() {
	step()[m_dims - 1] = element_size();

	for (int i = m_dims - 2; i >= 0; --i) {
		step()[i] = step()[i + 1] * size()[i + 1];
	}
}

void mt_mat::on_vaule_changed() {
	basiclog_assert2(m_auto_derivative == NULL || !m_auto_derivative->is_math_operation_recorded());
}

mt_mat mt_mat::flip(int dim) const {
	mt_mat res = *this;

	res.m_data += (res.size()[dim] - 1) * res.step()[dim];
	res.step()[dim] = -res.step()[dim];

	return res;
}

mt_mat mt_mat::flip(const vector<int>& dim_indexs) const {
	return flip((int)dim_indexs.size(), &dim_indexs[0]);
}

mt_mat mt_mat::flip(int size, const int* dims) const {
	mt_mat res = *this;

	for (int i = 0; i < size; ++i) {
		res.m_data += (res.size()[i] - 1) * res.step()[i];
		res.step()[i] = -res.step()[i];
	}

	return res;
}

mt_mat mt_mat::flip(const basicsys::b8* flip_flags) const {
	mt_mat res = *this;

	for (int i = 0; i < res.m_dims; ++i) {
		if (flip_flags[i]) {
			res.m_data += (res.size()[i] - 1) * res.step()[i];
			res.step()[i] = -res.step()[i];
		}
	}

	return res;
}

mt_mat mt_mat::flip_all_dim() const {
	mt_mat res = *this;

	for (int i = 0; i < res.m_dims; ++i) {
			res.m_data += (res.size()[i] - 1) * res.step()[i];
			res.step()[i] = -res.step()[i];
	}

	return res;
}

mt_mat mt_mat::channel_as_last_dim() const {
	mt_mat res;
	res.m_dims = m_dims + 1;

	if (res.m_dims > mt_Mat_Normal_Support_Dim) {
		res.m_dynamic_size_step_size = res.m_dims;
		res.m_dynamic_size_steps = new int[m_dynamic_size_step_size * 2];
	}

	for (int i = 0; i < m_dims; ++i) {
		res.size()[i] = size()[i];
		res.step()[i] = step()[i];
	}

	res.size()[res.m_dims - 1] = channel();
	res.step()[res.m_dims - 1] = mt_get_depth_size(depth());

	++*m_reference;
	res.m_depth_channel = mt_make_depth_channel(depth(), 1);
	res.m_reference = m_reference;
	res.m_data = m_data;
	res.m_shared_memory = m_shared_memory;

	return res;
}

mt_mat mt_mat::last_dim_as_channel() const {
	basiclog_assert_message2(m_dims > 1, L"dimension must be exceed 1, you can use increase_dim() method!");
	basiclog_assert_message2(channel() == 1, L"channel in the the last dimension must be 1!");
	basiclog_assert_message2(step()[m_dims - 1] > 0, L"step in the last dimension must be positive, maybe you used the flip() method!");
	basiclog_assert_message2(mt_array_iteration::get_continuous_dim(m_dims, size(), step(), element_channel_size()) < m_dims, L"data in the last dimension must be continuous, maybe you used the t() or swap() method!");

	mt_mat res;
	res.m_dims = m_dims - 1;

	for (int i = 0; i < res.m_dims; ++i) {
		res.size()[i] = size()[i];
		res.step()[i] = step()[i];
	}

	res.m_depth_channel = mt_make_depth_channel(depth(), size()[m_dims - 1]);

	++*m_reference;
	res.m_reference = m_reference;
	res.m_data = m_data;
	res.m_shared_memory = m_shared_memory;

	return res;
}

void mt_mat::split(vector<mt_mat>& channels, b8 can_share_memory) const {
	channels.resize(channel());

	if (channel() == 1 && can_share_memory) {
		channels[0] = *this;
	} else {
		sys_for(i, channels) {
			channels[i] = channel_at(i).clone();
		}
	}
}

mt_mat mt_mat::channel_at(int channel) const {
	mt_mat res = *this;

	res.m_depth_channel = mt_make_depth_channel(depth(), 1);
	res.m_data += channel * element_channel_size();

	return res;
}

mt_mat mt_mat::row(int row) const {
	basiclog_assert(L"mt_mat", m_dims == 2);

	return index(row, 0);
}

mt_mat mt_mat::col(int col) const {
	basiclog_assert(L"mt_mat", m_dims == 2);

	return index(col, 1);
}

mt_mat mt_mat::plane(int plane) const {
	basiclog_assert(L"mt_mat", m_dims == 3);

	return index(plane, 2);
}

mt_mat mt_mat::front(int number, int dim /* = 0 */) const {
	return sub(mt_range(0, number), dim);
}

mt_mat mt_mat::back(int number, int dim /* = 0 */) const {
	return sub(mt_range(size()[dim] - number, size()[dim]), dim);
}

mt_mat mt_mat::index(int index, int dim) const {
	return sub(mt_range(index, index + 1), dim);
}

mt_mat mt_mat::sub(int start_index, int stop_index, int dim /* = 0 */) const {
	return sub(mt_range(start_index, stop_index), dim);
}

mt_mat mt_mat::sub(const vector<mt_range>& ranges) const {
	return sub((i32)ranges.size(), &ranges[0]);
}

mt_mat mt_mat::sub(i32 dims, const mt_range* ranges) const {
	if (dims == 0) {
		return *this;
	}

	mt_mat res;

	basiclog_assert2(dims == m_dims && ranges != NULL);

	res.m_dims = m_dims;
	res.m_depth_channel = m_depth_channel;
	res.m_modified_number = m_modified_number;
	res.m_shared_memory = m_shared_memory;
	res.m_reference = m_reference;
	res.m_auto_derivative = m_auto_derivative;
	++*res.m_reference;

	if (res.m_dims > 4) {
		res.m_dynamic_size_step_size = res.m_dims;
		res.m_dynamic_size_steps = new int[res.m_dynamic_size_step_size * 2];
	}

	res.m_data = m_data;

	for (int i = 0; i < res.m_dims; ++i) {
		basiclog_assert(L"mt_mat", ranges[i].is_valid() && (ranges[i].m_end <= size()[i]));

		res.m_data += ranges[i].m_start * step()[i];
		res.step()[i] = step()[i];

		if (ranges[i].m_end == -1) {
			res.size()[i] = size()[i] - ranges[i].m_start;
		} else {
			res.size()[i] = ranges[i].size();
		}
	}		

	if (m_auto_derivative != NULL) {
		res.attach(m_auto_derivative);

		vector<mt_range> vec_ranges;
		mt_helper::vec_from_array(vec_ranges, dims, ranges);

		m_auto_derivative->sub(res, *this, vec_ranges);
	}

	return res;
}

mt_mat mt_mat::sub(const mt_rect& roi) const {
	basiclog_assert2(dim() == 2);

	mt_range ranges[2];
	ranges[0].m_start = roi.m_top;
	ranges[0].m_end = roi.m_top + roi.m_height;
	ranges[1].m_start = roi.m_left;
	ranges[1].m_end = roi.m_left + roi.m_width;

	return sub(2, ranges);
}

mt_mat mt_mat::sub(const mt_range& range, int dim) const {
	basicmath_mat_request_memory(mt_range, ranges, m_dims);

	for (i32 i = 0; i < m_dims; ++i) {
		if (i == dim) {
			ranges[i] = range;
		} else {
			ranges[i].m_start = 0;
			ranges[i].m_end = size()[i];
		}
	}

	basicmath_mat_release(ranges);
	return sub(m_dims, ranges);

	/*basiclog_assert(L"mt_mat", range.is_valid() && (range.m_end <= this->size()[dim]));

	mt_mat res;

	res.m_dims = m_dims;
	res.m_depth_channel = m_depth_channel;
	res.m_modified_number = m_modified_number;
	res.m_shared_memory = m_shared_memory;
	res.m_reference = m_reference;
	res.m_auto_derivative = m_auto_derivative;
	++*res.m_reference;

	if (res.m_dims > 4) {
	res.m_dynamic_size_step_size = res.m_dims;
	res.m_dynamic_size_steps = new int[res.m_dynamic_size_step_size * 2];
	}

	res.m_data = m_data;

	for (int i = 0; i < m_dims; ++i) {
	if (dim != i) {
	res.size()[i] = size()[i];
	} else {
	res.m_data += range.m_start * step()[i];

	if (range.m_end == -1) {
	res.size()[i] = size()[i] - range.m_start;
	} else {
	res.size()[i] = range.size();
	}
	}

	res.step()[i] = step()[i];
	}

	if (m_auto_derivative != NULL) {
	res.attach(m_auto_derivative);

	vector<mt_range> ranges;

	for (i32 i = 0; i )
	}

	return res;*/
}

vector<int> mt_mat::get_index(const u8* ptr_data) const {
	vector<int> index;
	get_index(index, ptr_data);

	return index;
}

void mt_mat::get_index(vector<int>& index, const u8* ptr_data) const {
	index.clear();
	index.assign(m_dims, -1);

	bool same_symbol = true;
	u8* memory_start_data = m_data;

	int positive_number = 0;

	for (int i = 0; i < m_dims; ++i) {
		if (step()[i] > 0) {
			++positive_number;
		} else if (positive_number > 0) {
			break;
		}
	}

	if (positive_number == 0 || positive_number == m_dims) {
		int offset = (int)(ptr_data - memory_start_data);
		basicmath_mat_request_memory(b8, flags, m_dims);
		mt_array_iteration::array_assign<b8>((u8*)flags, sys_false, m_dims);

		int count = 0;

		for (;;) {
			int max_abs_step = 0;
			int max_index = -1;

			for (int i = 0; i < m_dims; ++i) {
				if (flags[i] == 0 && abs(step()[i]) > max_abs_step) {
					max_abs_step = abs(step()[i]);
					max_index = i;
				}
			}

			index[max_index] = offset / step()[max_index];
			offset -= index[max_index] * step()[max_index];
			flags[max_index] = sys_true;

			++count;

			if (count == m_dims) {
				break;
			}
		}

		basicmath_mat_release(flags);
	} else {
		basicmath_mat_request_memory(b8, flip_flags, m_dims);

		for (i32 i = 0; i < m_dims; ++i) {
			flip_flags[i] = step()[i] > 0 ? sys_false : sys_true;
		}

		mt_mat temp_with_positive_step = flip(flip_flags);
		temp_with_positive_step.get_index(index, ptr_data);

		for (i32 i = 0; i < m_dims; ++i) {
			if (step()[i] != temp_with_positive_step.step()[i]) {
				index[i] = size()[i] - 1 - index[i];
			}
		}

		basicmath_mat_release(flip_flags);
	}
}

bool mt_mat::is_step_positive() const {
	for (i32 i = 0; i < m_dims; ++i) {
		if (step()[i] < 0) {
			return false;
		}
	}

	return true;
}

bool mt_mat::is_step_negative() const {
	for (i32 i = 0; i < m_dims; ++i) {
		if (step()[i] > 0) {
			return false;
		}
	}

	return true;
}