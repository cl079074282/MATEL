#pragma once



namespace basicmath {
	template<class T>
	class mt_point_t {
	public:

		mt_point_t(T x = -1, T y = -1) {
			m_x = -1;
			m_y = -1;
		}

		T m_x;
		T m_y;
	};

	typedef mt_point_t<int> mt_point;
	typedef mt_point_t<float> mt_point2f;
	typedef mt_point_t<double> mt_point2d;
}