
// ChildView.h : interface of the CChildView class
//


#pragma once

#include "CTextManager.h"

// CChildView window

class CChildView : public CWnd
{
	// Construction
public:
	CChildView();
	// Attributes
public:
	// Operations
public:
	// Overrides
protected:
	virtual BOOL PreCreateWindow(CREATESTRUCT& cs);
	// Implementation
public:
	virtual ~CChildView();
	// Generated message map functions
protected:
	afx_msg void OnPaint();
	DECLARE_MESSAGE_MAP()
public:
	afx_msg void OnFileOpen();
	afx_msg void OnFileSave();
	afx_msg void OnHScroll(UINT nSBCode, UINT nPos, CScrollBar* pScrollBar);
	afx_msg void OnVScroll(UINT nSBCode, UINT nPos, CScrollBar* pScrollBar);
	afx_msg void OnSize(UINT nType, int cx, int cy);
	afx_msg void OnLButtonDown(UINT nFlags, CPoint point);
	afx_msg void OnChar(UINT nChar, UINT nRepCnt, UINT nFlags);
	afx_msg void OnKeyDown(UINT nChar, UINT nRepCnt, UINT nFlags);
	afx_msg void OnSetFocus(CWnd* pOldWnd);
	afx_msg void OnKillFocus(CWnd* pNewWnd);
private:
	CTextManager TextBoard;
	long TextPosX;
	long TextPosY;
	int CurrentXPos;
	int CurrentYPos;
	int CaretPosXByChar;
	int CaretPosYByChar;
	int TempCaretPosXChar;
	int TempCaretPosYChar;

	void UpdateCaret(CWnd *pWnd, int MousePosX, int MousePosY);
	void UpdateScrollRange(CWnd *pWnd);
	void GetWindowSize(CWnd *pWnd, int* WindowHeight, int* WindowWidth);
	void GetFontSize(CWnd *pWnd, WCHAR character, int* height, int* width);
};

