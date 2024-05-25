
// PaintMFC.h : main header file for the PaintMFC application
//
#pragma once

#ifndef __AFXWIN_H__
	#error "include 'pch.h' before including this file for PCH"
#endif

#include "resource.h"       // main symbols


// CPaint:
// See PaintMFC.cpp for the implementation of this class
//

class CPaint : public CWinApp
{
public:
	CPaint() noexcept;


// Overrides
public:
	virtual BOOL InitInstance();
	virtual int ExitInstance();

// Implementation

public:
	afx_msg void OnAppAbout();
	DECLARE_MESSAGE_MAP()
};

extern CPaint theApp;
