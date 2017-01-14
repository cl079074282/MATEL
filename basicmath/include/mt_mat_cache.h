#pragma once



namespace basicmath {

	class mt_mat_cache {
	public:

		mt_mat_cache();
		~mt_mat_cache();

		mt_mat get_as(const mt_mat& src);

		mt_mat get(const vector<i32>& sizes, i32 depth_channel);

		mt_mat get(i32 dim, const i32* sizes, i32 depth_channel);

	protected:

		void statistic_memory();

		vector<mt_mat> m_caches;
		void* m_mutex;
	};


	extern mt_mat_cache s_mat_cache;
}