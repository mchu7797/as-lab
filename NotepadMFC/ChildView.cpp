
// ChildView.cpp : implementation of the CChildView class
//

#include "pch.h"
#include "framework.h"
#include "NotepadMFC.h"
#include "ChildView.h"

#ifdef _DEBUG
#define new DEBUG_NEW
#endif


// CChildView

CChildView::CChildView()
{
}

CChildView::~CChildView()
{
}


BEGIN_MESSAGE_MAP(CChildView, CWnd)
	ON_WM_PAINT()
	ON_WM_CHAR()
	ON_WM_KEYDOWN()
	ON_WM_LBUTTONDOWN()
	ON_WM_ERASEBKGND()
	ON_WM_VSCROLL()
	ON_WM_HSCROLL()
	ON_WM_SIZE()
	ON_WM_MOUSEWHEEL()
END_MESSAGE_MAP()

// CChildView message handlers

BOOL CChildView::PreCreateWindow(CREATESTRUCT& cs)
{
	if (!CWnd::PreCreateWindow(cs))
		return FALSE;

	m_nCaretPosX = 0;
	m_nCaretPosY = 0;
	m_nTextPosX = 0;
	m_nTextPosY = 0;
	
	m_CTextManager = new CTextManager();

	m_Font.CreateFont(16, 0, 0, 0, FW_NORMAL, FALSE, FALSE, FALSE, DEFAULT_CHARSET,
    OUT_DEFAULT_PRECIS, CLIP_DEFAULT_PRECIS, DEFAULT_QUALITY,
    DEFAULT_PITCH | FF_DONTCARE, _T("Courier New"));

	cs.dwExStyle |= WS_EX_CLIENTEDGE;
	cs.style &= ~WS_BORDER;
	cs.lpszClass = AfxRegisterWndClass(CS_HREDRAW | CS_VREDRAW | CS_DBLCLKS,
		::LoadCursor(nullptr, IDC_ARROW), reinterpret_cast<HBRUSH>(COLOR_WINDOW + 1), nullptr);

	return TRUE;
}

void CChildView::OnPaint()
{
	CPaintDC dc(this);

	CRect rcClient;
	GetClientRect(&rcClient);
	dc.FillSolidRect(&rcClient, GetSysColor(COLOR_WINDOW));

	CFont* pOldFont = dc.SelectObject(&m_Font);

	dc.SetBkMode(TRANSPARENT);

	for (int i = 0; i < m_CTextManager->GetMaxHeight(); ++i) {
		CString line = m_CTextManager->GetText(i);
		if (line != L"") {
			dc.TextOut(m_nTextPosX, m_nTextPosY + i * 16, line);
		}
	}

	dc.SelectObject(pOldFont);
}

void CChildView::OnChar(UINT nChar, UINT nRepCnt, UINT nFlags) {
	if (nChar >= 32) {
		m_CTextManager->HandleWrite(static_cast<TCHAR>(nChar), m_nCaretPosX, m_nCaretPosY);
		++m_nCaretPosX;
		UpdateScrollRange();
		Invalidate(TRUE);
	}

	CWnd::OnChar(nChar, nRepCnt, nFlags);
}

void CChildView::OnKeyDown(UINT nChar, UINT nRepCnt, UINT nFlags) {
	switch (nChar) {
	case VK_UP:
		if (m_nCaretPosY > 0) {
			--m_nCaretPosY;
			m_nCaretPosX = min(m_nCaretPosX, m_CTextManager->GetText(m_nCaretPosY).GetLength());
		}
		break;
	case VK_DOWN:
		if (m_nCaretPosY < m_CTextManager->GetMaxHeight() - 1) {
			++m_nCaretPosY;
			m_nCaretPosX = min(m_nCaretPosX, m_CTextManager->GetText(m_nCaretPosY).GetLength());
		}
		break;
	case VK_LEFT:
		if (m_nCaretPosX > 0) {
			--m_nCaretPosX;
		}
		break;
	case VK_RIGHT:
		if (m_nCaretPosX < m_CTextManager->GetText(m_nCaretPosY).GetLength()) {
			++m_nCaretPosX;
		}
		break;
	case VK_BACK:
		m_CTextManager->HandleHitBackspace(m_nCaretPosX, m_nCaretPosY);
		if (m_nCaretPosX > 0) {
			--m_nCaretPosX;
		}
		else if (m_nCaretPosY > 0) {
			--m_nCaretPosY;
			m_nCaretPosX = m_CTextManager->GetText(m_nCaretPosY).GetLength();
		}
		break;
	case VK_DELETE:
		m_CTextManager->HandleHitDelete(m_nCaretPosX, m_nCaretPosY);
		break;
	case VK_TAB:
		m_CTextManager->HandleHitTab(m_nCaretPosX, m_nCaretPosY);
		m_nCaretPosX += 8;
		break;
	case VK_RETURN:
		m_CTextManager->HandleHitEnter(m_nCaretPosX, m_nCaretPosY);
		m_nCaretPosX = 0;
		++m_nCaretPosY;
		break;
	case VK_INSERT:
		m_CTextManager->HandleHitInsert();
		break;
	case VK_HOME:
		m_nCaretPosX = 0;
		break;
	case VK_END:
		m_nCaretPosX = m_CTextManager->GetText(m_nCaretPosY).GetLength();
		break;
	}
	UpdateScrollRange();
	Invalidate(TRUE);
	CWnd::OnKeyDown(nChar, nRepCnt, nFlags);
}

void CChildView::OnLButtonDown(UINT nFlags, CPoint point) {
	UpdateCaret(point.x, point.y);
	Invalidate(TRUE);
	CWnd::OnLButtonDown(nFlags, point);
}

BOOL CChildView::OnEraseBkgnd(CDC* pDC) {
	return TRUE;
}
void CChildView::OnVScroll(UINT nSBCode, UINT nPos, CScrollBar* pScrollBar) {
	RECT rect;
	int nDelta = 0;
	int nMaxPos = m_CTextManager->GetMaxHeight() * 16 - m_nTextPosY;

	GetClientRect(&rect);

	switch (nSBCode) {
	case SB_LINEUP:
		nDelta = -16;
		break;
	case SB_LINEDOWN:
		nDelta = 16;
		break;
	case SB_PAGEUP:
		nDelta = -rect.bottom;
		break;
	case SB_PAGEDOWN:
		nDelta = rect.bottom;
		break;
	case SB_THUMBTRACK:
		nDelta = static_cast<int>(nPos) - m_nTextPosY;
		break;
	}

	nDelta = max(-m_nTextPosY, min(nDelta, nMaxPos));
	m_nTextPosY += nDelta;
	SetScrollPos(SB_VERT, m_nTextPosY);
	Invalidate(TRUE);

	CWnd::OnVScroll(nSBCode, nPos, pScrollBar);
}

void CChildView::OnHScroll(UINT nSBCode, UINT nPos, CScrollBar* pScrollBar) {
	RECT rect;
	int nDelta = 0;
	int nMaxPos = m_CTextManager->GetLongestLine().GetLength() * 8 - m_nTextPosX;

	GetClientRect(&rect);

	switch (nSBCode) {
	case SB_LINELEFT:
		nDelta = -8;
		break;
	case SB_LINERIGHT:
		nDelta = 8;
		break;
	case SB_PAGELEFT:
		nDelta = -rect.right;
		break;
	case SB_PAGERIGHT:
		nDelta = rect.right;
		break;
	case SB_THUMBTRACK:
		nDelta = static_cast<int>(nPos) - m_nTextPosX;
		break;
	}

	nDelta = max(-m_nTextPosX, min(nDelta, nMaxPos));
	m_nTextPosX += nDelta;
	SetScrollPos(SB_HORZ, m_nTextPosX);
	Invalidate(TRUE);

	CWnd::OnHScroll(nSBCode, nPos, pScrollBar);
}

void CChildView::OnSize(UINT nType, int cx, int cy) {
	UpdateScrollRange();
	CWnd::OnSize(nType, cx, cy);
}

BOOL CChildView::OnMouseWheel(UINT nFlags, short zDelta, CPoint pt) {
	int nDelta = -zDelta / WHEEL_DELTA * 16;
	int nMaxPos = m_CTextManager->GetMaxHeight() * 16 - m_nTextPosY;
	nDelta = max(-m_nTextPosY, min(nDelta, nMaxPos));
	m_nTextPosY += nDelta;
	SetScrollPos(SB_VERT, m_nTextPosY);
	Invalidate(TRUE);

	return CWnd::OnMouseWheel(nFlags, zDelta, pt);
}

void CChildView::UpdateCaret(int mouseX, int mouseY) {
	CClientDC dc(this);
	CFont* pOldFont = dc.SelectObject(&m_Font);
	TEXTMETRIC tm;
	dc.GetTextMetrics(&tm);
	dc.SelectObject(pOldFont);
	int nCharWidth = tm.tmAveCharWidth;
	int nCharHeight = tm.tmHeight;

	m_nCaretPosX = (mouseX - m_nTextPosX) / nCharWidth;
	m_nCaretPosY = (mouseY - m_nTextPosY) / nCharHeight;

	m_nCaretPosX = max(0, min(m_nCaretPosX, m_CTextManager->GetText(m_nCaretPosY).GetLength()));
	m_nCaretPosY = max(0, min(m_nCaretPosY, static_cast<int>(m_CTextManager->GetMaxHeight() - 1)));

	CPoint ptCaret(m_nCaretPosX * nCharWidth + m_nTextPosX, m_nCaretPosY * nCharHeight + m_nTextPosY);
	SetCaretPos(ptCaret);
}
void CChildView::UpdateScrollRange() {
	CClientDC dc(this);
	CFont* pOldFont = dc.SelectObject(&m_Font);
	TEXTMETRIC tm;
	dc.GetTextMetrics(&tm);
	dc.SelectObject(pOldFont);
	CRect rcClient;
	GetClientRect(&rcClient);

	SCROLLINFO si;
	si.cbSize = sizeof(si);
	si.fMask = SIF_RANGE | SIF_PAGE;
	si.nMin = 0;
	si.nMax = m_CTextManager->GetMaxHeight() * tm.tmHeight;
	si.nPage = rcClient.Height();
	SetScrollInfo(SB_VERT, &si);

	si.nMax = m_CTextManager->GetLongestLine().GetLength() * tm.tmAveCharWidth;
	si.nPage = rcClient.Width();
	SetScrollInfo(SB_HORZ, &si);
}
void CChildView::GetFontSize(CDC* dc, TCHAR character, int& height, int& width) {
	SIZE size;
	::GetTextExtentPoint32(dc->GetSafeHdc(), &character, 1, &size);
	height = size.cy;
	width = size.cx;
}
