
// NotepadMFC.h : main header file for the NotepadMFC application
//
#pragma once

#ifndef __AFXWIN_H__
	#error "include 'pch.h' before including this file for PCH"
#endif

#include "resource.h"       // main symbols


// CNotepad:
// See NotepadMFC.cpp for the implementation of this class
//

class CNotepad : public CWinApp
{
public:
	CNotepad() noexcept;


// Overrides
public:
	virtual BOOL InitInstance();
	virtual int ExitInstance();

// Implementation

public:
	afx_msg void OnAppAbout();
	DECLARE_MESSAGE_MAP()
};

extern CNotepad theApp;
