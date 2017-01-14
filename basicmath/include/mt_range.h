#pragma once



namespace basicmath {
	class mt_range {
	public:

		mt_range() {
			m_start = 0;
			m_end = 0;
		}

		mt_range(int start, int end) {
			m_start = start;
			m_end = end;
		}

		bool is_valid() const {
			return m_start >= 0 && (m_end == -1 || m_end > m_start);
		}
		
		int m_start;
		int m_end;

		int size() const {
			return m_end - m_start;
		}
	};
}