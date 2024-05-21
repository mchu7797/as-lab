#include "pch.h"
#include "CTextManager.h"

#include <vector>

CTextManager::CTextManager() {
	TextBoard = std::vector<CString>();

	TextBoard.push_back(L"");

	IsOverrideMode = false;
}

size_t CTextManager::getMaxHeight() { return TextBoard.size(); }

CString CTextManager::getLongestLine() {
	size_t MaxWidth = 0;
	std::vector<CString>::iterator LongestLine = TextBoard.begin();

	for (std::vector<CString>::iterator iter = TextBoard.begin();
		iter != TextBoard.end(); ++iter) {
		if (MaxWidth < iter->GetLength()) {
			MaxWidth = iter->GetLength();
			LongestLine = iter;
		}
	}

	return *LongestLine;
}

CString CTextManager::getText(int Row) {
	if (Row < 0 || Row >= TextBoard.size()) {
		return L"";
	}

	return TextBoard[Row];
}

void CTextManager::handleWrite(wchar_t Character, int X, int Y) {
	if (Y < 0 || Y > TextBoard.size()) {
		return;
	}

	if (X < 0) {
		return;
	}

	if (X > TextBoard[Y].GetLength()) {
		TextBoard[X].AppendChar(Character);
	}

	if (IsOverrideMode) {
		TextBoard[Y].Delete(X, 1);
	}

	TextBoard[Y].Insert(X, Character);
}

void CTextManager::handleHitEnter(int X, int Y) {
	if (Y < 0) {
		return;
	}

	if (Y > TextBoard.size() - 1) {
		TextBoard.push_back(CString());
	}

	if (X > TextBoard[Y].GetLength()) {
		TextBoard.insert(TextBoard.begin(), Y, CString());
	}

	CString temp = TextBoard[Y].Mid(X);
	TextBoard[Y].Delete(X, TextBoard[Y].GetLength() - X);

	if (Y > TextBoard.size() - 1) {
		TextBoard.push_back(temp);
	}
	else {
		TextBoard.insert(TextBoard.begin() + Y + 1, temp);
	}
}

void CTextManager::handleHitBackspace(int X, int Y) {
	if (Y < 0 || Y > TextBoard.size()) {
		return;
	}

	if (X < 0) {
		return;
	}

	if (X == 0) {
		if (Y == 0) {
			return;
		}

		if (TextBoard[Y].GetLength() > 0) {
			TextBoard[Y - 1].Append(TextBoard[Y]);
			TextBoard.erase(TextBoard.begin() + Y);
		}
		else {
			TextBoard.erase(TextBoard.begin() + Y);
		}

		return;
	}

	if (X > TextBoard[Y].GetLength()) {
		if (!TextBoard[Y].IsEmpty()) {
			TextBoard[Y] = TextBoard[Y].Left(TextBoard[Y].GetLength() - 1);
		}
		return;
	}

	TextBoard[Y].Delete(X - 1, 1);
}

void CTextManager::handleHitTab(int X, int Y) {
	if (Y < 0 || Y > TextBoard.size()) {
		return;
	}

	if (X < 0) {
		return;
	}

	if (X > TextBoard[Y].GetLength()) {
		TextBoard[Y].Append(L"        ");
		return;
	}

	TextBoard[Y].Insert(X, L"        ");
}

void CTextManager::handleHitInsert() { IsOverrideMode = !IsOverrideMode; }

void CTextManager::handleHitDelete(int X, int Y) {
	if (Y < 0 || Y > TextBoard.size()) {
		return;
	}

	if (X >= TextBoard[Y].GetLength()) {
		if (Y + 1 > TextBoard.size()) {
			return;
		}

		TextBoard[Y].Append(TextBoard[Y + 1]);
		TextBoard.erase(TextBoard.begin() + Y + 1);

		return;
	}

	TextBoard[Y].Delete(X, 1);
}

void CTextManager::clear() {
	TextBoard.clear();
}