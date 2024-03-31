#ifndef PAINT_H
#define PAINT_H

#include "resource.h"
#include "framework.h"

#define MAX_LOADSTRING 100

HINSTANCE hInst;
WCHAR szTitle[MAX_LOADSTRING];
WCHAR szWindowClass[MAX_LOADSTRING];

ATOM MyRegisterClass(HINSTANCE hInstance);
BOOL InitInstance(HINSTANCE, int);
LRESULT CALLBACK WndProc(HWND, UINT, WPARAM, LPARAM);
INT_PTR CALLBACK About(HWND, UINT, WPARAM, LPARAM);

void draw(HWND HWnd, HDC Hdc, LPARAM LParam);

int MousePos[2];
bool isDrawing = false;

#endif