#pragma once
#include <vector>
#include "framework.h"

class CTextManager {
public:
	CTextManager();
	CString GetText(int row);
	CString GetLongestLine();
	size_t GetMaxHeight();

	std::vector<CString>::iterator begin();
	std::vector<CString>::iterator end();
	size_t size();

	void AppendString(const CString& string);
	void Clear();

	void HandleWrite(TCHAR character, int x, int y);
	void HandleHitEnter(int x, int y);
	void HandleHitBackspace(int x, int y);
	void HandleHitTab(int x, int y);
	void HandleHitInsert();
	void HandleHitDelete(int x, int y);
protected:
	bool m_IsOverrideMode;
	std::vector<CString> m_TextBoard;
};
