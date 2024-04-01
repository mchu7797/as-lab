#ifndef PAINT_H
#define PAINT_H

#include "framework.h"
#include "resource.h"

#define MAX_LOADSTRING 100

HINSTANCE hInst;
WCHAR szTitle[MAX_LOADSTRING];
WCHAR szWindowClass[MAX_LOADSTRING];

ATOM MyRegisterClass(HINSTANCE hInstance);
BOOL InitInstance(HINSTANCE, int);
LRESULT CALLBACK WndProc(HWND, UINT, WPARAM, LPARAM);
INT_PTR CALLBACK About(HWND, UINT, WPARAM, LPARAM);

void draw(HWND HWnd, HDC Hdc, LPARAM LParam);
void createButton(const wchar_t *Name, long X, long Y, long Width, long Height,
                  HMENU Id, HWND HWnd, HINSTANCE HInstance);
void createRGBTable(long X, long Y, long Width, long Height, HMENU Id,
                    HWND HWnd, HINSTANCE HInstance);
void createWidthTable(long X, long Y, long Width, long Height, HMENU Id,
                      HWND HWnd, HINSTANCE HInstance);

bool trySave(HWND HWnd);
bool tryOpen(HWND HWnd);

void setColor(HWND HWnd, HDC Hdc);
void setWidth(HWND HWnd, HDC Hdc);
void handleScroll(HWND HWnd, WPARAM WParam, LPARAM LParam);
void handleDrawMode(HWND HWnd, WPARAM WParam, LPARAM LParam);

int MousePos[2];
int PenWidth = 10;
int Red, Green, Blue;
bool IsDrawing = false;
int DrawMode = 0;

#endif