
// ChildView.h : interface of the CChildView class
//


#pragma once

#include "ImageVector.h"

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
protected:
	DECLARE_MESSAGE_MAP()
public:
	afx_msg void OnPaint();
	afx_msg void OnLButtonDown(UINT nFlags, CPoint point);
	afx_msg void OnLButtonUp(UINT nFlags, CPoint point);
	afx_msg void OnMouseMove(UINT nFlags, CPoint point);
	afx_msg void OnEditClearAll();
	afx_msg void OnEditEllipse();
	afx_msg void OnEditEraser();
	afx_msg void OnEditRectangle();
	afx_msg void OnEditPen();
	afx_msg void OnEditSizeUp();
	afx_msg void OnEditSizeDown();
	afx_msg void OnEditFillshape();
	afx_msg void OnColorBlueMinus();
	afx_msg void OnColorBluePlus();
	afx_msg void OnColorGreenMinus();
	afx_msg void OnColorGreenPlus();
	afx_msg void OnColorRedMinus();
	afx_msg void OnColorRedPlus();
	afx_msg void OnFileSave();
	afx_msg void OnFileOpen();
private:
	std::vector<CImageVector> m_vectorHistory;
	CImageVector m_tempVector;

	CPoint m_mousePos;
	CPoint m_tempMousePos;
	
	int m_penWidth = 10;
	
	int m_redIntensity = 0;
	int m_greenIntensity = 0;
	int m_blueIntensity = 0;
	
	int m_drawMode = 0;
	bool m_isDrawing = 0;
	bool m_doFillShape = 0;
	bool m_isImageBackedUp = false;

	BYTE* m_bitmapBuffer;
	CRect m_bitmapBufferRect;

	void Draw();
	void DrawRectangle();
	void DrawEllipse();

	void BackupRectImage(CRect rect);
	void RestoreRectImage();
	void ClearRectImage();
};

