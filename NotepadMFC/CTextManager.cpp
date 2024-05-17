#include "pch.h"
#include "CTextManager.h"

CTextManager::CTextManager() {
	m_IsOverrideMode = false;
	m_TextBoard.clear();
	m_TextBoard.push_back(L"");
}

CString CTextManager::GetText(int row) {
	if (row < 0 || row >= m_TextBoard.size()) {
		return L"";
	}
	return m_TextBoard[row];
}

CString CTextManager::GetLongestLine() {
	size_t maxWidth = 0;
	auto longestLine = m_TextBoard.begin();
	for (auto iter = m_TextBoard.begin(); iter != m_TextBoard.end(); ++iter) {
		if (maxWidth < iter->GetLength()) {
			maxWidth = iter->GetLength();
			longestLine = iter;
		}
	}

	return *longestLine;
}

size_t CTextManager::GetMaxHeight() {
	return m_TextBoard.size();
}

std::vector<CString>::iterator CTextManager::begin() {
	return m_TextBoard.begin();
}

std::vector<CString>::iterator CTextManager::end() {
	return m_TextBoard.end();
}

size_t CTextManager::size() {
	return m_TextBoard.size();
}

void CTextManager::AppendString(const CString& string) {
	m_TextBoard.push_back(string);
}

void CTextManager::Clear() {
	m_TextBoard.clear();
}

void CTextManager::HandleWrite(TCHAR character, int x, int y) {
	if (y < 0 || y > m_TextBoard.size()) {
		return;
	}
	if (x < 0) {
		return;
	}

	if (x > m_TextBoard[y].GetLength()) {
		m_TextBoard[y].AppendChar(character);
	}

	if (m_IsOverrideMode) {
		m_TextBoard[y].Delete(x);
	}

	m_TextBoard[y].Insert(x, CString(character));
}

void CTextManager::HandleHitEnter(int x, int y) {
	if (y < 0) {
		return;
	}
	if (y > m_TextBoard.size() - 1) {
		m_TextBoard.push_back(CString());
	}

	if (x > m_TextBoard[y].GetLength()) {
		m_TextBoard.insert(m_TextBoard.begin() + y, CString());
	}

	CString temp = m_TextBoard[y].Mid(x);
	m_TextBoard[y].Delete(x, m_TextBoard[y].GetLength() - x);

	if (y > m_TextBoard.size() - 1) {
		m_TextBoard.push_back(temp);
	}
	else {
		m_TextBoard.insert(m_TextBoard.begin() + y + 1, temp);
	}
}

void CTextManager::HandleHitBackspace(int x, int y) {
	if (y < 0 || y > m_TextBoard.size()) {
		return;
	}
	if (x < 0) {
		return;
	}

	if (x == 0) {
		if (y == 0) {
			return;
		}

		if (m_TextBoard[y].GetLength() > 0) {
			m_TextBoard[y - 1].Append(m_TextBoard[y]);
			m_TextBoard.erase(m_TextBoard.begin() + y);
		}
		else {
			m_TextBoard.erase(m_TextBoard.begin() + y);
		}

		return;
	}

	if (x > m_TextBoard[y].GetLength()) {
		m_TextBoard[y].Delete(m_TextBoard[y].GetLength() - 1);
		return;
	}

	m_TextBoard[y].Delete(x - 1);
}

void CTextManager::HandleHitTab(int x, int y) {
	if (y < 0 || y > m_TextBoard.size()) {
		return;
	}
	if (x < 0) {
		return;
	}

	if (x > m_TextBoard[y].GetLength()) {
		m_TextBoard[y].Append(L"        ");
		return;
	}

	m_TextBoard[y].Insert(x, L"        ");
}

void CTextManager::HandleHitInsert() {
	m_IsOverrideMode = !m_IsOverrideMode;
}

void CTextManager::HandleHitDelete(int x, int y) {
	if (y < 0 || y > m_TextBoard.size()) {
		return;
	}
	if (x >= m_TextBoard[y].GetLength()) {
		if (y + 1 > m_TextBoard.size()) {
			return;
		}

		m_TextBoard[y].Append(m_TextBoard[y + 1]);
		m_TextBoard.erase(m_TextBoard.begin() + y + 1);

		return;
	}

	m_TextBoard[y].Delete(x);
}