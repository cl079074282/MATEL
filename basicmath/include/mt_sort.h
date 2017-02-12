#pragma once



namespace basicmath {

	class mt_sort {
	public:

		template<class T>
		static void sort(vector<i32>& indexs, i32 size, T* start, b8 increase) {
			indexs.resize(size, 0);

			for (i32 i = 0; i < size; ++i) {
				indexs[i] = i;
			}

			for (i32 i = 0; i < (i32)indexs.size() - 1; ++i) {
				for (i32 j = 0; j < (i32)indexs.size() - j - 1; ++j) {
					if (increase) {
						if (indexs[j] > indexs[j + 1]) {
							swap(start[j], start[j + 1]);
							swap(indexs[j], indexs[j + 1]);
						}
					}
				}
			}
		}
	};

}