#include "stdafx.h"

#include "ml_statistic_info.h"

void ml_statistic_info::write(sys_json_writer& writer) const {
	writer<<L"{";

	writer<<L"total_loss"<<m_total_loss;
	writer<<L"losses"<<m_losses;
	writer<<L"precisions"<<m_precisions;
	writer<<L"sequence_precisions"<<m_sequence_precisions;

	writer<<L"}";
}

void ml_statistic_info::read(sys_json_reader& reader) {
	reader[L"total_loss"]>>m_total_loss;
	reader[L"losses"]>>m_losses;
	reader[L"precisions"]>>m_precisions;
	reader[L"sequence_precisions"]>>m_sequence_precisions;
}