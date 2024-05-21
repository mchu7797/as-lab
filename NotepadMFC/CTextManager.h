#pragma once

#include <vector>

class CTextManager
{
public:
	CTextManager();

	CString getText(int Row);
	CString getLongestLine();
	size_t getMaxHeight();

	std::vector<CString>::iterator begin() { return TextBoard.begin(); }
	std::vector<CString>::iterator end() { return TextBoard.end(); }
	size_t size() { return TextBoard.size(); }

	void appendString(CString string) { TextBoard.push_back(string); }
	void clear();

	void handleWrite(wchar_t Character, int X, int Y);
	void handleHitEnter(int X, int Y);
	void handleHitBackspace(int X, int Y);
	void handleHitTab(int X, int Y);
	void handleHitInsert();
	void handleHitDelete(int X, int Y);

private:
	bool IsOverrideMode;
	std::vector<CString> TextBoard;
};

