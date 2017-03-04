#pragma once



namespace basicml {

	class ml_statistic_info {
	public:

		void write(sys_json_writer& writer) const;
		void read(sys_json_reader& reader);

		f64 m_total_loss;
		map<wstring, f64> m_losses;
		map<wstring, f64> m_precisions;
		map<wstring, f64> m_sequence_precisions;
	};

}