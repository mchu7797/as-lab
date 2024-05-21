
// ChildView.cpp : implementation of the CChildView class
//

#include "pch.h"
#include "framework.h"
#include "NotepadMFC.h"
#include "ChildView.h"
#include "CTextManager.h"

#ifdef _DEBUG
#define new DEBUG_NEW
#endif


// CChildView

CChildView::CChildView()
{
	TextPosX = 0;
	TextPosY = 0;
	CurrentXPos = 0;
	CurrentYPos = 0;
	CaretPosXByChar = 0;
	CaretPosYByChar = 0;
	TempCaretPosXChar = 0;
	TempCaretPosYChar = 0;
}

CChildView::~CChildView()
{
}


BEGIN_MESSAGE_MAP(CChildView, CWnd)
	ON_WM_PAINT()
	ON_COMMAND(ID_FILE_OPEN, &CChildView::OnFileOpen)
	ON_COMMAND(ID_FILE_SAVE, &CChildView::OnFileSave)
	ON_WM_HSCROLL()
	ON_WM_VSCROLL()
	ON_WM_SIZE()
	ON_WM_LBUTTONDOWN()
	ON_WM_CHAR()
	ON_WM_KEYDOWN()
	ON_WM_SETFOCUS()
	ON_WM_KILLFOCUS()
END_MESSAGE_MAP()



// CChildView message handlers

BOOL CChildView::PreCreateWindow(CREATESTRUCT& cs)
{
	if (!CWnd::PreCreateWindow(cs))
		return FALSE;

	cs.dwExStyle |= WS_EX_CLIENTEDGE;
	cs.style &= ~WS_BORDER;
	cs.lpszClass = AfxRegisterWndClass(CS_HREDRAW | CS_VREDRAW | CS_DBLCLKS,
		::LoadCursor(nullptr, IDC_ARROW), reinterpret_cast<HBRUSH>(COLOR_WINDOW + 1), nullptr);

	return TRUE;
}

void CChildView::OnPaint()
{
	CPaintDC dc(this); // device context for painting

	// TODO: Add your message handler code here

	// Do not call CWnd::OnPaint() for painting messages
}

void CChildView::OnFileOpen()
{
	CFileDialog openDlg(TRUE, _T("txt"), NULL, OFN_FILEMUSTEXIST | OFN_HIDEREADONLY, _T("텍스트 파일(*.txt)|*.txt|모든 파일(*.*)|*.*||"));

	if (openDlg.DoModal() != IDOK) {
		// Failed
		AfxMessageBox(_T("파일을 열지 못했습니다."));
		return;
	}

	CString filePath = openDlg.GetPathName();
	CStdioFile file;

	if (!file.Open(filePath, CFile::modeRead | CFile::typeText)) {
		// Failed
		AfxMessageBox(_T("파일을 열지 못했습니다."));
		return;
	}

	TextBoard.clear();
	CString line;

	while (file.ReadString(line)) {
		TextBoard.appendString(line.GetBuffer());
	}

	file.Close();
}

void CChildView::OnFileSave()
{
	CFileDialog saveDlg(FALSE, _T("txt"), NULL, OFN_OVERWRITEPROMPT, _T("텍스트 파일|*.txt|모든 파일|*.*||"));

	if (saveDlg.DoModal() != IDOK) {
		// Failed
		AfxMessageBox(_T("파일을 저장하지 못했습니다."));
		return;
	}

	CString filePath = saveDlg.GetPathName();
	CStdioFile file;

	if (!file.Open(filePath, CFile::modeCreate | CFile::modeWrite | CFile::typeText)) {
		// Failed
		AfxMessageBox(_T("파일을 저장하지 못했습니다."));
		return;
	}

	CString content;

	for (auto it = TextBoard.begin(); it < TextBoard.end(); ++it) {
		content += *it + _T("\r\n");
	}

	file.WriteString(content);
	file.Close();
}


void CChildView::OnHScroll(UINT nSBCode, UINT nPos, CScrollBar* pScrollBar)
{
	SCROLLINFO ScrollInfo;

	ScrollInfo.cbSize = sizeof(ScrollInfo);
	ScrollInfo.fMask = SIF_ALL;

	CWnd::GetScrollInfo(SB_HORZ, &ScrollInfo);
	CurrentXPos = ScrollInfo.nPos;

	switch (nSBCode)
	{
	case SB_TOP:
		ScrollInfo.nPos = ScrollInfo.nMin;
		break;
	case SB_BOTTOM:
		ScrollInfo.nPos = ScrollInfo.nMax;
		break;
	case SB_LINEUP:
		--ScrollInfo.nPos;
		break;
	case SB_LINEDOWN:
		++ScrollInfo.nPos;
		break;
	case SB_PAGEDOWN:
		ScrollInfo.nPos += ScrollInfo.nPage;
		break;
	case SB_PAGEUP:
		ScrollInfo.nPos -= ScrollInfo.nPage;
		break;
	case SB_THUMBTRACK:
		ScrollInfo.nPos = nPos;
		break;
	default:
		break;
	}

	ScrollInfo.fMask = SIF_POS;
	CWnd::SetScrollInfo(SB_HORZ, &ScrollInfo, true);
	CWnd::GetScrollInfo(SB_HORZ, &ScrollInfo);

	if (ScrollInfo.nPos != TextPosX)
	{
		CWnd::ScrollWindow(15 * (CurrentXPos - ScrollInfo.nPos), 0, nullptr, nullptr);
		TextPosX = -ScrollInfo.nPos;
		Invalidate(false);
	}

	CWnd::OnHScroll(nSBCode, nPos, pScrollBar);
}


void CChildView::OnVScroll(UINT nSBCode, UINT nPos, CScrollBar* pScrollBar)
{
	SCROLLINFO ScrollInfo;

	ScrollInfo.cbSize = sizeof(ScrollInfo);
	ScrollInfo.fMask = SIF_ALL;

	CWnd::GetScrollInfo(SB_VERT, &ScrollInfo);
	CurrentYPos = ScrollInfo.nPos;

	switch (nSBCode)
	{
	case SB_TOP:
		ScrollInfo.nPos = ScrollInfo.nMin;
		break;
	case SB_BOTTOM:
		ScrollInfo.nPos = ScrollInfo.nMax;
		break;
	case SB_LINEUP:
		--ScrollInfo.nPos;
		break;
	case SB_LINEDOWN:
		++ScrollInfo.nPos;
		break;
	case SB_PAGEDOWN:
		ScrollInfo.nPos += ScrollInfo.nPage;
		break;
	case SB_PAGEUP:
		ScrollInfo.nPos -= ScrollInfo.nPage;
		break;
	case SB_THUMBTRACK:
		ScrollInfo.nPos = nPos;
		break;
	default:
		break;
	}

	ScrollInfo.fMask = SIF_POS;
	CWnd::SetScrollInfo(SB_VERT, &ScrollInfo, true);
	CWnd::GetScrollInfo(SB_VERT, &ScrollInfo);

	if (ScrollInfo.nPos != TextPosY)
	{
		CWnd::ScrollWindow(15 * (CurrentYPos - ScrollInfo.nPos), 0, nullptr, nullptr);
		TextPosY = -ScrollInfo.nPos;
		Invalidate(false);
	}

	CWnd::OnVScroll(nSBCode, nPos, pScrollBar);
}


void CChildView::OnSize(UINT nType, int cx, int cy)
{
	UpdateScrollRange();

	CWnd::OnSize(nType, cx, cy);
}


void CChildView::OnLButtonDown(UINT nFlags, CPoint point)
{
	HWND pWnd = CWnd::GetSafeHwnd();

	if (TextBoard.getText(0).GetLength() == 0 && TextBoard.size() == 1) {
		return;
	}

	UpdateCaret(point.x, point.y);
	CWnd::Invalidate(true);

	CWnd::OnLButtonDown(nFlags, point);
}


void CChildView::OnChar(UINT nChar, UINT nRepCnt, UINT nFlags)
{
	if (nChar < 32 || nChar == 127)
	{
		CWnd::OnChar(nChar, nRepCnt, nFlags);
		return;
	}

	TextBoard.handleWrite(nChar, CaretPosXByChar, CaretPosYByChar);
	++CaretPosXByChar;

	CPoint CaretPos = CWnd::GetCaretPos();

	int WindowWidth, CharWidth;
	GetWindowSize(nullptr, &WindowWidth);
	GetFontSize(nChar, nullptr, &CharWidth);

	if (CaretPos.x >= WindowWidth - 20)
	{
		TextPosX -= CharWidth;
		UpdateScrollRange();

		SCROLLINFO ScrollInfo;
		memset(&ScrollInfo, 0, sizeof(ScrollInfo));
		ScrollInfo.cbSize = sizeof(ScrollInfo);
		ScrollInfo.fMask = SIF_ALL;

		CWnd::GetScrollInfo(SB_HORZ, &ScrollInfo);
		ScrollInfo.nPos = ScrollInfo.nMax;
		CWnd::SetScrollInfo(SB_HORZ, &ScrollInfo, true);
	}

	Invalidate(false);
}


void CChildView::OnKeyDown(UINT nChar, UINT nRepCnt, UINT nFlags)
{
	// TODO: Add your message handler code here and/or call default

	CWnd::OnKeyDown(nChar, nRepCnt, nFlags);
}


void CChildView::OnSetFocus(CWnd* pOldWnd)
{
	CWnd::OnSetFocus(pOldWnd);
	CWnd::CreateSolidCaret(2, 15);
	CWnd::ShowCaret();

	CWnd::OnSetFocus(pOldWnd);
}


void CChildView::OnKillFocus(CWnd* pNewWnd)
{
	CWnd::OnKillFocus(pNewWnd);
	CWnd::HideCaret();
	DestroyCaret();

	CWnd::OnKillFocus(pNewWnd);
}

void CChildView::UpdateCaret(int MousePosX, int MousePosY)
{
	HWND WindowHandle;
	CClientDC DeviceContext(this);
	TEXTMETRIC TextMetric;

	int NewIndexX = 0;
	int NewIndexY = 0;
	int TempPosX = 0;

	int TextHeight = 0;
	int TextWidth = 0;

	DeviceContext.GetTextMetricsW(&TextMetric);

	for (int i = 0; i < TextBoard.size(); ++i)
	{
		if (MousePosY < i * TextMetric.tmHeight + TextPosY)
		{
			--NewIndexY;
			break;
		}

		++NewIndexY;
	}

	NewIndexY = (NewIndexY >= TextBoard.size() ? TextBoard.size() - 1 : NewIndexY);
	CString CurrentLine = TextBoard.getText(NewIndexY);

	for (int i = 0; i < CurrentLine.GetLength(); ++i)
	{
		GetFontSize(CurrentLine[i], &TextHeight, &TextWidth);

		if (MousePosX < TempPosX + TextWidth + TextPosX) {
			--NewIndexX;
			break;
		}

		TempPosX += TextWidth;
		++NewIndexX;
	}

	CaretPosXByChar = NewIndexX;
	CaretPosYByChar = NewIndexY;
}

void CChildView::GetFontSize(WCHAR character, int* height, int* width)
{
	CClientDC DeviceContext(this);
	TEXTMETRICW TextMetric;

	DeviceContext.GetTextMetricsW(&TextMetric);

	if (height != nullptr)
	{
		*height = TextMetric.tmHeight;
	}

	if (width != nullptr)
	{
		*width = DeviceContext.GetTextExtent(CString(character), 1).cx;
	}
}

void CChildView::UpdateScrollRange()
{
	CClientDC DeviceContext(this);
	SCROLLINFO ScrollInfo;
	TEXTMETRIC TextMetric;
	int TextBoardMaxWidth = 0;
	int WindowHeight, WindowWidth;

	DeviceContext.GetTextMetricsW(&TextMetric);
	GetWindowSize(&WindowHeight, &WindowWidth);

	ScrollInfo.cbSize = sizeof(ScrollInfo);
	ScrollInfo.fMask = SIF_RANGE | SIF_PAGE;
	ScrollInfo.nMin = 0;
	ScrollInfo.nMax = TextBoard.size() * TextMetric.tmHeight - WindowHeight;

	if (ScrollInfo.nMax < 0) {
		ScrollInfo.nMax = 0;
	}

	if (ScrollInfo.nMax != 0) {
		ScrollInfo.nPage = ScrollInfo.nMax / TextBoard.size();
	}
	else {
		ScrollInfo.nPage = 0;
	}

	CWnd::SetScrollInfo(SB_VERT, &ScrollInfo, true);

	if (ScrollInfo.nMax == 0 && ScrollInfo.nPage == 0) {
		TextPosY = 0;
	}

	CString LongestString = TextBoard.getLongestLine();
	int FontHeight, FontWidth;

	for (auto i = 0; i < LongestString.GetLength(); ++i)
	{
		GetFontSize(LongestString[i], &FontHeight, &FontWidth);
		TextBoardMaxWidth += FontWidth;
	}
}

void CChildView::GetWindowSize(int* WindowHeight, int* WindowWidth) {
	RECT WindowRect;

	CWnd::GetClientRect(&WindowRect);

	if (WindowHeight != nullptr) {
		*WindowHeight = WindowRect.bottom - WindowRect.top;
	}

	if (WindowWidth != nullptr) {
		*WindowWidth = WindowRect.bottom - WindowRect.top;
	}
}
