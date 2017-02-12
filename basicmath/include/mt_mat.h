#pragma once

#define basicmath_mat_request_memory(data_type, name, need_size)	\
	data_type fixed##name[mt_Mat_Normal_Support_Dim];	\
	data_type* name = fixed##name; \
	if (need_size > mt_Mat_Normal_Support_Dim) { \
	name = new data_type[need_size];	\
	}

#define basicmath_mat_release(name)	\
	if (name != fixed##name) { \
	delete[] name; \
	}

namespace basicmath{

	class mt_auto_derivative;	

class mt_mat {
	public:
		mt_mat();

		mt_mat(i32 cols, i32 depth_channel, const mt_scalar& init_value = mt_scalar());
		mt_mat(i32 rows, i32 cols, i32 depth_channel, const mt_scalar& init_value = mt_scalar());
		mt_mat(i32 planes, i32 rows, i32 cols, i32 depth_channel, const mt_scalar& init_value = mt_scalar());
		mt_mat(i32 dims, const i32* sizes, i32 depth_channel, const mt_scalar& init_value = mt_scalar());
		mt_mat(const vector<i32>& sizes, i32 depth_channel, const mt_scalar& init_value = mt_scalar());

		mt_mat(i32 cols, i32 depth_channel, u8* data, const i32* steps);
		mt_mat(i32 rows, i32 cols, i32 depth_channel, u8* data, const i32* steps);
		mt_mat(i32 planes, i32 rows, i32 cols, i32 depth_channel, u8* data, const i32* steps);
		mt_mat(i32 dims, const i32* sizes, i32 depth_channel, u8* data, const i32* steps);
		mt_mat(const vector<i32>& sizes, i32 depth_channel, u8* data, const vector<i32>& steps);

		enum Construct_Type {
			Construct_Type_Operator_Equal,
			Construct_Type_Create_As_Size,
		};

		mt_mat(const mt_mat& other, Construct_Type type = Construct_Type_Operator_Equal, const mt_scalar init_value = mt_scalar());
		mt_mat(const mt_mat& other, i32 dims, const mt_range* ranges);
		mt_mat(const mt_mat& other, const vector<mt_range>& ranges);
		mt_mat(const mt_mat& other, const mt_range& range, i32 dim = 0);
		mt_mat(const mt_mat& other, const mt_rect& roi);

		~mt_mat();

		void attach(mt_auto_derivative* auto_derivative) {
			m_auto_derivative = auto_derivative;
		}

		void detach() {
			m_auto_derivative = NULL;
		}

		mt_auto_derivative* auto_derivative() const {
			return m_auto_derivative;
		}

		i32 reference_number() const {
			if (m_reference == NULL) {
				return 0;
			}
		
			return *m_reference;
		}
			 
		mt_mat derivative(const mt_mat& other) const;
		vector<mt_mat> derivative(const vector<mt_mat>& others) const;
		void derivative(vector<mt_mat>& reses, const vector<mt_mat>& others) const;
		
		b8 zero() const;
		b8 empty() const;


		/**
		@param left_high is sys_true means the following mat

		1 0 0 
		0 1 0
		0 0 1

		while left_high is sys_false means the following mat
		0 0 1
		0 1 0
		1 0 0
		*/
		mt_mat& set_eye(b8 left_high = sys_true);

		mt_mat& set(const mt_mat& other); 

		/**
		@note all channels of an element will be set to the same value.
		*/
		mt_mat& set(double value);
		mt_mat& set_incremental(double start_value, b8 same_value_for_multi_channel = sys_true);

		/**
		For a m*n*l matrix, if l <= 4, we can use Scalar to set the value.
		*/
		mt_mat& set(const mt_scalar& value);
		mt_mat& set(const vector<basicsys::f64>& value);

		mt_mat& set(double value, basicsys::i32 row, basicsys::i32 col);
		mt_mat& set(double value, basicsys::i32 plane, basicsys::i32 row, basicsys::i32 col);
		mt_mat& set(double value, basicsys::i32 dim, const basicsys::i32* indexs);

		mt_mat& set(const mt_scalar& value, basicsys::i32 row, basicsys::i32 col);
		mt_mat& set(const mt_scalar& value, basicsys::i32 plane, basicsys::i32 row, basicsys::i32 col);
		mt_mat& set(const mt_scalar& value, basicsys::i32 dim, const basicsys::i32* indexs);

		mt_mat& set(const vector<double>& values, basicsys::i32 row, basicsys::i32 col);
		mt_mat& set(const vector<double>& values, basicsys::i32 plane, basicsys::i32 row, basicsys::i32 col);
		mt_mat& set(const vector<double>& values, basicsys::i32 dims, const basicsys::i32* indexs);

		mt_mat& set(const double* values, basicsys::i32 row, basicsys::i32 col);
		mt_mat& set(const double* values, basicsys::i32 plane, basicsys::i32 row, basicsys::i32 col);
		mt_mat& set(const double* values, basicsys::i32 dims, const basicsys::i32* indexs);

		mt_scalar get(basicsys::i32 row, basicsys::i32 col) const;
		mt_scalar get(basicsys::i32 plane, basicsys::i32 row, basicsys::i32 col) const;
		mt_scalar get(basicsys::i32 dim, const basicsys::i32* indexs) const;

		void get(vector<double>& values, basicsys::i32 row, basicsys::i32 col) const;
		void get(vector<double>& values, basicsys::i32 plane, basicsys::i32 row, basicsys::i32 col) const;
		void get(vector<double>& values, basicsys::i32 dim, const basicsys::i32* indexs) const;

		void get(double* values, basicsys::i32 row, basicsys::i32 col) const;
		void get(double* values, basicsys::i32 plane, basicsys::i32 row, basicsys::i32 col) const;
		void get(double* values, basicsys::i32 dims, const basicsys::i32* indexs) const;

		void operator=(const mt_mat& other);
		bool operator==(const mt_mat& other) const;
		bool operator!=(const mt_mat& other) const;
		b8 memory_shared(const mt_mat& other) const;
		b8 same(const mt_mat& other) const;
		b8 same_size(const mt_mat& other) const;
		

		mt_mat clone() const;
		mt_mat convert(int depth) const;
		
		mt_mat row(int row) const;
		mt_mat col(int col) const;
		mt_mat plane(int plane) const;
		mt_mat index(int index, int dim = 0) const;

		mt_mat row_range(i32 start_index, i32 end_index);
		mt_mat col_range(i32 start_index, i32 end_index);
		mt_mat row_range(const mt_range& range) const;
		mt_mat col_range(const mt_range& range) const;

		mt_mat front(int number, int dim = 0) const;
		mt_mat back(int number, int dim = 0) const;

		mt_mat sub(int start_index, int stop_index, int dim = 0) const;
		mt_mat sub(const mt_range& range, int dim = 0) const;
		mt_mat sub(const vector<mt_range>& ranges) const;
		mt_mat sub(i32 dims, const mt_range* ranges) const;
		mt_mat sub(const mt_rect& roi) const;
		
		mt_mat increase_dim(int added_dim) const;
		mt_mat decrease_dim(int deleted_dim) const;

		mt_mat reshape(i32 rows, i32 cols) const;
		mt_mat reshape(i32 planes, i32 rows, i32 cols) const;
		mt_mat reshape(const vector<int>& sizes) const;
		mt_mat reshape(int dims, const int* sizes) const;
		mt_mat split_dim(int dim, int splited_dims, int* splited_sizes) const;
		mt_mat combine_dim(int combined_dim_start, int combined_dim_count) const;

		mt_mat repeat(i32 nsize, i32 dim = 0) const;
		mt_mat repeat(const vector<i32>& nsizes) const;
		mt_mat repeat(i32 dims, const i32* nsizes) const;
		

		mt_mat t() const;
		mt_mat swap_dim(int dim_a, int dim_b) const;
		mt_mat resort_dim(int* resort_dim) const;

		mt_mat flip(int dim) const;
		mt_mat flip(const vector<basicsys::i32>& dim_indexs) const;
		mt_mat flip(int size, const int* dims) const;
		mt_mat flip(const basicsys::b8* flip_flags) const;
		mt_mat flip_all_dim() const;

		mt_mat channel_as_last_dim() const;
		mt_mat last_dim_as_channel() const;

		void split(vector<mt_mat>& channels, b8 can_share_memory = sys_true) const;
		mt_mat channel_at(int channel) const;

		int depth() const {
			return mt_get_depth(m_depth_channel);
		}

		int dim() const {
			return m_dims;
		}

		int channel() const {
			return mt_get_channel(m_depth_channel);
		}

		int depth_channel() const {
			return m_depth_channel;
		}

		u8* data() {
			return m_data;
		}

		const u8* data() const {
			return m_data;
		}

		u8* memory_data();
		const u8* memory_data() const;

		const int* size() const {
			return (m_dims > 4 ? m_dynamic_size_steps : m_fixed_size_steps);
		}

		const int* step() const {
			return size() + m_dims;
		}

		/**
		@note An element may contains multiple channels.
		*/
		int element_number() const;
		int element_size() const;
		int element_channel_size() const;

		/** Determine the minimal abs step whether equals the element channel size. If the mat comes from channel_at() of other mat with more than 1 channel, this method will 
			return false.  
		*/
		b8 min_abs_step_equal_element_size() const;

		/** Determine the memory whether is continuous in accessing stage (t() method may return a non-continuous mat even the memory is still continuous). Steps can be all positive or all negative.
		*/
		b8 continuous() const;
		b8 step_positive() const;
		b8 step_negative() const;

		bool valid_index(const vector<basicsys::i32>& indexs) const {
			basiclog_assert2(!indexs.empty());

			return valid_index((basicsys::i32)indexs.size(), &indexs[0]);
		}

		bool valid_index(int size, const int* indexs) const {
			for (int i = 0; i < size; ++i) {
				if (indexs[i] >= this->size()[i] || indexs[i] < 0) {
					return false;
				}
			}

			return true;
		}

		template<class T>
		T& at(i32 index, i32 channel) {
			on_vaule_changed();
			return at<T>(1, &index, channel);
		}

		template<class T>
		const T& at(i32 index, i32 channel) const {
			return at<T>(1, &index, channel);
		}

		template<class T>
		T& at(i32 index1, i32 index2, i32 channel) {
			on_vaule_changed();
			i32 indexes[] = {index1, index2};
			return at<T>(2, indexes, channel);
		}

		template<class T>
		const T& at(i32 index1, i32 index2, i32 channel) const {
			i32 indexes[] = {index1, index2};
			return at<T>(2, indexes, channel);
		}

		template<class T>
		T& at(i32 index1, i32 index2, i32 index3, i32 channel) {
			on_vaule_changed();
			i32 indexes[] = {index1, index2, index3};
			return at<T>(3, indexes, channel);
		}

		template<class T>
		const T& at(i32 index1, i32 index2, i32 index3, i32 channel) const {
			i32 indexes[] = {index1, index2, index3};
			return at<T>(3, indexes, channel);
		}

		template<class T>
		T& at(const vector<i32>& index, i32 channel) {
			on_vaule_changed();
			return *ptr<T>((i32)index.size(), &index[0], channel);
		}

		template<class T>
		const T& at(const vector<i32>& index, i32 channel) const {
			return *ptr<T>((int)index.size(), &index[0], channel);
		}

		template<class T>
		T& at(i32 size, const i32* indexs, i32 channel) {
			on_vaule_changed();
			return *ptr<T>(size, indexs, channel);
		}

		template<class T>
		const T& at(i32 size, const int* indexs, i32 channel) const {
			return *ptr<T>(size, indexs, channel);
		}

		template<class T>
		T* ptr(i32 index1, i32 channel) {
			on_vaule_changed();
			return ptr<T>(1, &index1, channel);
		}

		template<class T>
		const T* ptr(i32 index1, i32 channel) const {
			return ptr<T>(1, &index1, channel);
		}

		template<class T>
		T* ptr(i32 index1, i32 index2, i32 channel) {
			on_vaule_changed();
			i32 indexes[] = {index1, index2};
			return ptr<T>(2, indexes, channel);
		}

		template<class T>
		const T* ptr(i32 index1, i32 index2, i32 channel) const {
			i32 indexes[] = {index1, index2};
			return ptr<T>(2, indexes, channel);
		}

		template<class T>
		T* ptr(i32 index1, i32 index2, i32 index3, i32 channel) {
			on_vaule_changed();
			i32 indexes[] = {index1, index2, index3};
			return ptr<T>(3, indexes, channel);
		}

		template<class T>
		const T* ptr(i32 index1, i32 index2, i32 index3, i32 channel) const {
			i32 indexes[] = {index1, index2, index3};
			return ptr<T>(3, indexes, channel);
		}

		template<class T>
		T* ptr(const vector<int>& index, int channel) {
			on_vaule_changed();
			return ptr<T>((int)index.size(), &index[0], channel);
		}

		template<class T>
		const T* ptr(const vector<int>& index, int channel) const {
			return ptr<T>((int)index.size(), &index[0], channel);
		}

		template<class T>
		T* ptr(int size, const int* indexs, int channel) {
			on_vaule_changed();
			basiclog_assert2(indexs != NULL);
			basiclog_assert2(size <= m_dims);
			basiclog_assert2(valid_index(size, indexs));

			u8* ptr_data = m_data;

			for (int dim = 0; dim < size; ++dim) {
				ptr_data += indexs[dim] * step()[dim];
			}

			return (T*)ptr_data;
		}

		template<class T>
		const T* ptr(int size, const int* indexs, int channel) const {
			return const_cast<mt_mat*>(this)->ptr<T>(size, indexs, channel);
		}

		vector<int> get_index(const u8* ptr_data) const;
		void get_index(vector<int>& indexs, const u8* ptr_data) const;

		int* size() {
			return (m_dims > 4 ? m_dynamic_size_steps : m_fixed_size_steps);
		}

		int* step() {
			return size() + m_dims;
		}

		mt_mat& operator+=(double value);
		mt_mat& operator-=(double value);
		mt_mat& operator*=(double value);
		mt_mat& operator/=(double value);

		mt_mat& operator+=(const mt_scalar& value);
		mt_mat& operator-=(const mt_scalar& value);
		mt_mat& operator*=(const mt_scalar& value);
		mt_mat& operator/=(const mt_scalar& value);

		mt_mat& operator+=(const vector<double>& value);
		mt_mat& operator-=(const vector<double>& value);
		mt_mat& operator*=(const vector<double>& value);
		mt_mat& operator/=(const vector<double>& value);

		mt_mat& operator+=(const mt_mat& other);
		mt_mat& operator-=(const mt_mat& other);
		mt_mat& operator*=(const mt_mat& other);
		mt_mat& operator/=(const mt_mat& other);

		mt_mat operator+(double value) const;
		mt_mat operator-(double value) const;
		mt_mat operator*(double value) const;
		mt_mat operator/(double value) const;

		mt_mat operator+(const mt_scalar& value) const;
		mt_mat operator-(const mt_scalar& value) const;
		mt_mat operator*(const mt_scalar& value) const;
		mt_mat operator/(const mt_scalar& value) const;

		mt_mat operator+(const vector<double>& value) const;
		mt_mat operator-(const vector<double>& value) const;
		mt_mat operator*(const vector<double>& value) const;
		mt_mat operator/(const vector<double>& value) const;

		mt_mat operator+(const mt_mat& value) const;
		mt_mat operator-(const mt_mat& value) const;
		mt_mat operator*(const mt_mat& value) const;
		mt_mat operator/(const mt_mat& value) const;

		mt_mat mul(const mt_mat& other) const;
		
		/**
		pow(value, number)
		*/
		mt_mat pow(f64 number) const;
		mt_mat& self_pow(f64 number);

		mt_mat exp() const;
		mt_mat& self_exp();

		mt_mat pooling(mt_mat& mask_mat, mt_Pooling_Type pooling_type, i32 size, const basicsys::i32* kernel_sizes, const basicsys::i32* strides) const;
		mt_mat unpooling(const int* src_size, const mt_mat& mask_mat, mt_Pooling_Type pooling_type, i32 size, const int* kernel_sizes, const int* strides) const;

		mt_mat expand(i32 size, const i32* side_sizes_1, const i32* side_size_2, const mt_scalar& filled_channel_value = mt_scalar(0)) const;
		mt_mat expand(i32 size, const i32* side_sizes_1, const i32* side_size_2, const vector<f64>& filled_channel_value) const;

		mt_mat sub_stride(i32 size, const i32* strides) const;

		mt_mat conv(const mt_mat& kernel, mt_Conv_Boundary_Type boundary_type = mt_Conv_Boundary_Type_Valid, const int* conv_strides = NULL) const;
		
		/** Calculate loss for current mat and the matching_mat.

		For mt_Loss_Type_Quardratic loss, the current and matching_mat can be any dimensions, while for mt_Loss_Type_0_1 and mt_Loss_Type_Logarithmic losses, 
		the current mat and the matching_mat must be 2-d, where each line represents a label of a sample.

		@param matching_mat The matching mat.
		@param type @see mt_Loss_Type.
		@return Loss result 1 * 1 mat.
		*/
		mt_mat loss(const mt_mat& mathcing_mat, mt_Loss_Type type) const;

		//double calculate_dist(const mt_mat& other, mt_Dist_Type dist_type) const;
		//void calculate_dist(vector<double>& channel_dists, const mt_mat& other, mt_Dist_Type dist_type) const;
		//void calculate_dist(double* channel_dists, const mt_mat& other, mt_Dist_Type dist_type) const;
		//mt_mat calculate_dist(const mt_mat& other, mt_Dist_Type dist_type) const;

		int get_modified_number() const {
			return *m_modified_number;
		}

		mt_mat activate(mt_Activate_Type type, const vector<f64>& activate_params) const;
		mt_mat activate(mt_Activate_Type type, i32 activate_param_size, const f64* activate_params) const;

		mt_mat& self_activate(mt_Activate_Type type, const vector<f64>& activate_params);
		mt_mat& self_activate(mt_Activate_Type type, i32 activate_param_size, const f64* activate_params);

		void eigen(mt_mat& eigen_value, mt_mat& eigen_vectors) const;

		enum Reduce_Type {
			Reduce_Type_Sum,
			Reduce_Type_Mean,
			Reduce_Type_Max,
			Reduce_Type_Min,
			Reduce_Type_Variance,
			Reduce_Type_Unbias_Variance,
			Reduce_Type_Standard_Variance,
			Reduce_Type_Standard_Unbias_Variance
		};

		mt_mat reduce(Reduce_Type type, i32 reduce_dim) const;

	protected:

		friend class mt_mat_helper;
		friend class mt_mat_cache;

		mt_mat& create_imp(int dims, const int* sizes, int depth_channel);

		void try_deallocate();

		void fill_auto_step();

		void on_vaule_changed();

		friend class private_mat;
		friend class ml_mat_operation;

		int* m_reference;
		int* m_modified_number;

		u8* m_shared_memory;
		u8* m_data;			//!<the first element of this mat for cpu mat or gpu mat	
		
		int m_depth_channel;
		int m_dims;

		int m_fixed_size_steps[mt_Mat_Normal_Support_Dim * 2];
		int* m_dynamic_size_steps;
		int m_dynamic_size_step_size;

		mutable mt_auto_derivative* m_auto_derivative;
	};

	void write(basicsys::sys_strcombine& str, const mt_mat& data);
	void write(basicsys::sys_json_writer& writer, const mt_mat& data);
	void read(mt_mat& data, const basicsys::sys_json_reader& reader);

	mt_mat operator-(const mt_mat& mat);

	mt_mat operator+(f64 value, const mt_mat& mat);
	mt_mat operator-(f64 value, const mt_mat& mat);
	mt_mat operator*(f64 value, const mt_mat& mat);
	mt_mat operator/(f64 value, const mt_mat& mat);

	mt_mat operator+(const mt_scalar& value, const mt_mat& mat);
	mt_mat operator-(const mt_scalar& value, const mt_mat& mat);
	mt_mat operator*(const mt_scalar& value, const mt_mat& mat);
	mt_mat operator/(const mt_scalar& value, const mt_mat& mat);

    mt_mat operator+(const vector<f64>& value, const mt_mat& mat);
	mt_mat operator-(const vector<f64>& value, const mt_mat& mat);
	mt_mat operator*(const vector<f64>& value, const mt_mat& mat);
	mt_mat operator/(const vector<f64>& value, const mt_mat& mat);
	
}