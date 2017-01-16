/** @file sample.h

@author hailiang xu

Brief description.
Detailed description. 
*/
#pragma once

/** Brief description.

Detailed description.
@note
*/

#define MAX(a, b) ((a) > (b) ? (a) : (b)) 

/** Brief description.

Detailed description.
@note
*/

static int ml_value;

/** Brief description.

Detailed description.
@note
*/
int size();

/** Brief description.

More details about this class. 
@note
*/
class ml_binary_file_node {
public:

/** Brief description.

More details about this class. 
@note
*/
	enum Type {
		NONE, //!< Detailed description.
		INT,
		CHAR,
		FLOAT,
		DOUBLE,
		STR,
		POINT,
		SIZE,
		RECT,
		MAT,
		USER,
	};

	/** Brief description.
	Detailed description.
	@note
	*/
	ml_binary_file_node() 
		: m_file(NULL) 
		, m_type(NONE)
		, m_offset(0) {

	}

	class user_data {
	public:
		byte* m_data;
		int m_size;
	};

	/** Brief description.
	
	Detailed description.
	@note
	@code
	if (valid_type(type)) {
   
	}
	@endcode
	@param type.
	@return true indicates the valid type otherwise invalid type. 
	*/
	static bool valid_type(int type);

	void operator<<(int value);
	void operator<<(wchar_t text);
	void operator<<(float value);
	void operator<<(double value);

	void operator<<(const wstring& text);
	void operator<<(const Mat& mat);

	void operator<<(const user_data& data);

	void operator>>(int& value);
	void operator>>(wchar_t& text);
	void operator>>(float& value);
	void operator>>(double& value);

	void operator>>(wstring& text);
	void operator>>(Mat& mat);

	bool empty() const;
	
	/** Brief description.
	
	Detailed description for the member.
	*/
	FILE* m_file;	
	
	int m_type;	
              
	int m_size;	//!< Brief description.
	__int64 m_offset;	//!< Brief description.
};



源文件中：
方法内注释：
//
