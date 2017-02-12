#pragma once




namespace basicml {

	class ml_model_config {
		basicsys_disable_copy(ml_model_config)

	public:

		virtual ~ml_model_config() {}

		virtual void write(const wstring& path) const = 0;
		virtual void write(sys_json_writer& writer) const = 0;
		static ml_model_config* read(const wstring& file);
		static ml_model_config* read(const sys_json_reader& reader);

		virtual ml_model_config* clone() const = 0;
	};


}