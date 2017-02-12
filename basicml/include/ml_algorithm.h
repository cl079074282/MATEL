#pragma once



namespace basicml {

	class ml_supervised_algorithm;

	class ml_algorithm {
		basicsys_disable_copy(ml_algorithm);

	public:

		basicsys_class_name_method(ml_algorithm);

		ml_algorithm() {
			m_depth = mt_F64;
		}

		virtual ~ml_algorithm() {
			
		}

		virtual void write(const wstring& path, b8 text_type = sys_true, b8 write_learned_param = sys_true) const = 0;
		virtual void write(sys_json_writer& writer, b8 write_learned_param = sys_true) const = 0;
		static ml_algorithm* read(const wstring& path, b8 text_type = sys_true);
		static ml_algorithm* read(sys_json_reader& reader);

		virtual b8 empty() {return sys_true;}

		virtual ml_algorithm* clone() const = 0;

		virtual void set_depth(i32 depth) {
			m_depth = depth;
		}

		virtual i32 depth() const {
			return m_depth;
		}

		virtual ml_supervised_algorithm* to_supervised_model() {return NULL;}
		virtual const ml_supervised_algorithm* to_supervised_model() const {return NULL;}

	protected:

		i32 m_depth;

		
	};

}