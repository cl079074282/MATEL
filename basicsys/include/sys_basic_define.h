#pragma once

#define basicsys_delete(x)	\
	if ((x) != NULL) {delete (x); (x) = NULL;}

#define basicsys_delete_array(x)	\
	if ((x) != NULL) {delete[] (x); (x) = NULL;}

#define basicsys_delete_vector_ptr(x)	\
	for (int i = 0; i < (int)x.size(); ++i) {\
	basicsys_delete(x[i]);\
	}

#define basicsys_delete_vector_array_ptr(x)	\
	for (int i = 0; i < (int)x.size(); ++i) {\
	basicsys_delete_array(x[i]);\
	}


#define basicsys_disable_copy(class_name)	\
protected:	\
	void operator=(const class_name& other) {}

#define basicsys_disable_construction(class_name)	\
protected:	\
	class_name() {}	\

/**
#pragma omp parallel for num_threads(omp_get_max_threads()) is equivalent to #pragma omp parallel for.
*/
#define basicsys_init_omp_mkl()	\
	int sys_suggest_threads = omp_get_num_procs() - 1;	\
	if (sys_suggest_threads == 0) sys_suggest_threads = 1;	\
	omp_set_num_threads(sys_suggest_threads);	\
	omp_set_dynamic(1);	\
	mkl_set_num_threads(sys_suggest_threads);	\
	mkl_set_dynamic(1);	


namespace basicsys {

	typedef short              i16;
	typedef int                i32;
	typedef long long          i64;
	typedef unsigned char      u8;
	typedef unsigned short     u16;
	typedef unsigned int       u32;
	typedef unsigned long long u64;

	typedef char				i8;
	typedef wchar_t				c16;

	typedef float				f32;
	typedef double				f64;

	/** Instead of bool, because vector<bool> is not an normal vector!
	*/
	struct b8 {

		b8(bool value) {
			m_data = value ? 1 : 0;
		}

		b8(i32 data = 0)
		: m_data(data) {

		}

		b8 operator !() const {
			return m_data ? b8(0) : b8(1);
		}

		operator u8() const {
			return m_data;
		}

		u8 m_data;
	};

	static const b8 sys_true = 1;
	static const b8 sys_false = 0;
}