#include "Paint.h"
#include "framework.h"

int APIENTRY wWinMain(_In_ HINSTANCE hInstance,
                      _In_opt_ HINSTANCE hPrevInstance, _In_ LPWSTR lpCmdLine,
                      _In_ int nCmdShow) {
  UNREFERENCED_PARAMETER(hPrevInstance);
  UNREFERENCED_PARAMETER(lpCmdLine);

  LoadStringW(hInstance, IDS_APP_TITLE, szTitle, MAX_LOADSTRING);
  LoadStringW(hInstance, IDC_PAINT, szWindowClass, MAX_LOADSTRING);
  MyRegisterClass(hInstance);

  if (!InitInstance(hInstance, nCmdShow)) {
    return FALSE;
  }

  HACCEL hAccelTable = LoadAccelerators(hInstance, MAKEINTRESOURCE(IDC_PAINT));

  MSG msg;

  while (GetMessage(&msg, nullptr, 0, 0)) {
    if (!TranslateAccelerator(msg.hwnd, hAccelTable, &msg)) {
      TranslateMessage(&msg);
      DispatchMessage(&msg);
    }
  }

  return (int)msg.wParam;
}

ATOM MyRegisterClass(HINSTANCE hInstance) {
  WNDCLASSEXW wcex;

  wcex.cbSize = sizeof(WNDCLASSEX);

  wcex.style = CS_HREDRAW | CS_VREDRAW;
  wcex.lpfnWndProc = WndProc;
  wcex.cbClsExtra = 0;
  wcex.cbWndExtra = 0;
  wcex.hInstance = hInstance;
  wcex.hIcon = LoadIcon(hInstance, MAKEINTRESOURCE(IDI_PAINT));
  wcex.hCursor = LoadCursor(nullptr, IDC_ARROW);
  wcex.hbrBackground = (HBRUSH)(COLOR_WINDOW + 1);
  wcex.lpszMenuName = MAKEINTRESOURCEW(IDC_PAINT);
  wcex.lpszClassName = szWindowClass;
  wcex.hIconSm = LoadIcon(wcex.hInstance, MAKEINTRESOURCE(IDI_SMALL));

  return RegisterClassExW(&wcex);
}

BOOL InitInstance(HINSTANCE hInstance, int nCmdShow) {
  hInst = hInstance;

  HWND hWnd = CreateWindowW(szWindowClass, szTitle, WS_OVERLAPPED | WS_SYSMENU,
                            CW_USEDEFAULT, 0, 800, 600, nullptr, nullptr,
                            hInstance, nullptr);

  if (!hWnd) {
    return FALSE;
  }

  ShowWindow(hWnd, nCmdShow);
  UpdateWindow(hWnd);

  return TRUE;
}

LRESULT CALLBACK WndProc(HWND hWnd, UINT message, WPARAM wParam,
                         LPARAM lParam) {
  HDC Hdc;

  switch (message) {
  case WM_CREATE:
    createButton(L"Pen", 20, 20, 100, 50, (HMENU)100, hWnd, hInst);
    createButton(L"Erase", 20, 75, 100, 50, (HMENU)101, hWnd, hInst);
    createButton(L"Rectangle", 130, 20, 100, 50, (HMENU)102, hWnd, hInst);
    createButton(L"Ellipse", 130, 75, 100, 50, (HMENU)103, hWnd, hInst);
    CreateWindowW(L"Button", L"ClearAll", WS_CHILD | WS_VISIBLE, 460, 80,
                  100, 50, hWnd, (HMENU)400, hInst, nullptr);
    createRGBTable(250, 20, 150, 15, (HMENU)200, hWnd, hInst);
    createWidthTable(460, 20, 150, 15, (HMENU)300, hWnd, hInst);
    break;
  case WM_HSCROLL:
    handleScroll(hWnd, wParam, lParam);
    InvalidateRect(hWnd, nullptr, false);
    break;
  case WM_LBUTTONDOWN:
    if (DrawMode > 2) {
      Hdc = GetDC(hWnd);
      IsDrawing = true;
      draw(hWnd, Hdc, lParam);
      IsDrawing = false;
      ReleaseDC(hWnd, Hdc);
      InvalidateRect(hWnd, nullptr, false);
    } else {
      IsDrawing = true;
      MousePos[0] = GET_X_LPARAM(lParam);
      MousePos[1] = GET_Y_LPARAM(lParam);
    }
    break;
  case WM_MOUSEMOVE:
    if (IsDrawing) {
      Hdc = GetDC(hWnd);
      draw(hWnd, Hdc, lParam);
      ReleaseDC(hWnd, Hdc);
      InvalidateRect(hWnd, nullptr, false);
    }
    break;
  case WM_LBUTTONUP:
    if (DrawMode < 3) {
      IsDrawing = false;
    }
    break;
  case WM_COMMAND: {
    handleDrawMode(hWnd, wParam, lParam);

    int wmId = LOWORD(wParam);
    switch (wmId) {
    case IDM_ABOUT:
      DialogBox(hInst, MAKEINTRESOURCE(IDD_ABOUTBOX), hWnd, About);
      break;
    case IDM_EXIT:
      DestroyWindow(hWnd);
      break;
    default:
      return DefWindowProc(hWnd, message, wParam, lParam);
    }
  } break;
  case WM_PAINT: {
    PAINTSTRUCT ps;
    HDC hdc = BeginPaint(hWnd, &ps);
    setColor(hWnd, hdc);
    setWidth(hWnd, hdc);
    EndPaint(hWnd, &ps);
  } break;
  case WM_DESTROY:
    PostQuitMessage(0);
    break;
  default:
    return DefWindowProc(hWnd, message, wParam, lParam);
  }
  return 0;
}

INT_PTR CALLBACK About(HWND hDlg, UINT message, WPARAM wParam, LPARAM lParam) {
  UNREFERENCED_PARAMETER(lParam);
  switch (message) {
  case WM_INITDIALOG:
    return (INT_PTR)TRUE;

  case WM_COMMAND:
    if (LOWORD(wParam) == IDOK || LOWORD(wParam) == IDCANCEL) {
      EndDialog(hDlg, LOWORD(wParam));
      return (INT_PTR)TRUE;
    }
    break;
  }
  return (INT_PTR)FALSE;
}

void draw(HWND HWnd, HDC Hdc, LPARAM LParam) {
  if (!IsDrawing || DrawMode == 0) {
    return;
  }

  if (GET_Y_LPARAM(LParam) < 150 || MousePos[1] < 150) {
    return;
  }
  
  if (DrawMode == 1) {
    HPEN NewPen = CreatePen(PS_SOLID, PenWidth, RGB(Red, Green, Blue));
    HGDIOBJ OldPen = SelectObject(Hdc, NewPen);
    MoveToEx(Hdc, MousePos[0], MousePos[1], nullptr);
    MousePos[0] = GET_X_LPARAM(LParam);
    MousePos[1] = GET_Y_LPARAM(LParam);
    LineTo(Hdc, MousePos[0], MousePos[1]);
    SelectObject(Hdc, OldPen);
    DeleteObject(NewPen);
  } else if (DrawMode == 2) {
    HPEN NewPen = CreatePen(PS_SOLID, 40, RGB(255, 255, 255));
    HGDIOBJ OldPen = SelectObject(Hdc, NewPen);
    MoveToEx(Hdc, MousePos[0], MousePos[1], nullptr);
    MousePos[0] = GET_X_LPARAM(LParam);
    MousePos[1] = GET_Y_LPARAM(LParam);
    LineTo(Hdc, MousePos[0], MousePos[1]);
    SelectObject(Hdc, OldPen);
    DeleteObject(NewPen);
  } else if (DrawMode == 3) {
    HBRUSH NewBrush = CreateSolidBrush(RGB(Red, Green, Blue));
    HBRUSH OldBrush = (HBRUSH)SelectObject(Hdc, NewBrush);
    MousePos[0] = GET_X_LPARAM(LParam);
    MousePos[1] = GET_Y_LPARAM(LParam);
    Rectangle(Hdc, MousePos[0] - PenWidth, MousePos[1] - PenWidth,
              MousePos[0] + PenWidth, MousePos[1] + PenWidth);
    SelectObject(Hdc, OldBrush);
    DeleteObject(NewBrush);
  } else {
    HBRUSH NewBrush = CreateSolidBrush(RGB(Red, Green, Blue));
    HBRUSH OldBrush = (HBRUSH)SelectObject(Hdc, NewBrush);
    MousePos[0] = GET_X_LPARAM(LParam);
    MousePos[1] = GET_Y_LPARAM(LParam);
    Ellipse(Hdc, MousePos[0] - PenWidth, MousePos[1] - PenWidth,
            MousePos[0] + PenWidth, MousePos[1] + PenWidth);
    SelectObject(Hdc, OldBrush);
    DeleteObject(NewBrush);
  }

  wchar_t Buffer[100];
  wsprintfW(Buffer, L"X : %05d, Y : %05d", MousePos[0], MousePos[1]);
  TextOutW(Hdc, 460, 60, Buffer, lstrlenW(Buffer));
}

void createButton(const wchar_t *Name, long X, long Y, long Width, long Height,
                  HMENU Id, HWND HWnd, HINSTANCE HInstance) {
  CreateWindowW(L"Button", Name, WS_CHILD | WS_VISIBLE | BS_CHECKBOX, X, Y,
                Width, Height, HWnd, Id, HInstance, nullptr);
}

void createRGBTable(long X, long Y, long Width, long Height, HMENU Id,
                    HWND HWnd, HINSTANCE HInstance) {
  CreateWindowW(L"Scrollbar", L"Red", WS_CHILD | WS_VISIBLE | SBS_HORZ, X, Y,
                Width, Height, HWnd, (HMENU)Id, HInstance, nullptr);
  SetScrollRange(FindWindowExW(HWnd, nullptr, L"Scrollbar", L"Red"), SB_CTL, 0,
                 255, true);
  CreateWindowW(L"Scrollbar", L"Green", WS_CHILD | WS_VISIBLE | SBS_HORZ, X,
                Y + Height + 10, Width, Height, HWnd, (HMENU)(Id + 1),
                HInstance, nullptr);
  SetScrollRange(FindWindowExW(HWnd, nullptr, L"Scrollbar", L"Green"), SB_CTL,
                 0, 255, true);
  CreateWindowW(L"Scrollbar", L"Blue", WS_CHILD | WS_VISIBLE | SBS_HORZ, X,
                Y + (Height + 10) * 2, Width, Height, HWnd, (HMENU)(Id + 2),
                HInstance, nullptr);
  SetScrollRange(FindWindowExW(HWnd, nullptr, L"Scrollbar", L"Blue"), SB_CTL, 0,
                 255, true);
}

void createWidthTable(long X, long Y, long Width, long Height, HMENU Id,
                      HWND HWnd, HINSTANCE HInstance) {
  CreateWindowW(L"Scrollbar", L"Width", WS_CHILD | WS_VISIBLE | SBS_HORZ, X, Y,
                Width, Height, HWnd, (HMENU)Id, HInstance, nullptr);
  SetScrollRange(FindWindowExW(HWnd, nullptr, L"Scrollbar", L"Width"), SB_CTL,
                 0, 255, true);
}

void setColor(HWND HWnd, HDC Hdc) {
  static int FunctionCalled;

  Red =
      GetScrollPos(FindWindowExW(HWnd, nullptr, L"Scrollbar", L"Red"), SB_CTL);
  Green = GetScrollPos(FindWindowExW(HWnd, nullptr, L"Scrollbar", L"Green"),
                       SB_CTL);
  Blue =
      GetScrollPos(FindWindowExW(HWnd, nullptr, L"Scrollbar", L"Blue"), SB_CTL);

  HBRUSH NewBrush = CreateSolidBrush(RGB(Red, Green, Blue));
  HBRUSH OldBrush = (HBRUSH)SelectObject(Hdc, NewBrush);

  Rectangle(Hdc, 410, 20, 450, 85);

  SelectObject(Hdc, OldBrush);
  DeleteObject(NewBrush);

  wchar_t Text[30];

  wsprintf(Text, L"R%03d, G%03d, B%03d", Red, Green, Blue);
  TextOutW(Hdc, 250, 90, Text, lstrlenW(Text));
}

void setWidth(HWND HWnd, HDC Hdc) {
  PenWidth = GetScrollPos(FindWindowExW(HWnd, nullptr, L"Scrollbar", L"Width"),
                          SB_CTL);
  PenWidth = (int)(PenWidth / 10 + 1);

  HBRUSH NewBrush = CreateSolidBrush(RGB(255, 255, 255));
  HBRUSH OldBrush = (HBRUSH)SelectObject(Hdc, NewBrush);
  Ellipse(Hdc, 650 - 26, 45 - 26, 650 + 26, 45 + 26);
  
  NewBrush = CreateSolidBrush(RGB(Red, Green, Blue));
  OldBrush = (HBRUSH)SelectObject(Hdc, NewBrush);
  Ellipse(Hdc, 650 - PenWidth, 45 - PenWidth, 650 + PenWidth, 45 + PenWidth);

  SelectObject(Hdc, OldBrush);
  DeleteObject(NewBrush);
  
  wchar_t Text[15];
  wsprintfW(Text, L"W%02d", PenWidth);
  TextOutW(Hdc, 460, 45, Text, lstrlenW(Text));
}

void handleDrawMode(HWND HWnd, WPARAM WParam, LPARAM LParam) {
  if (HIWORD(WParam) != BN_CLICKED) {
    return;
  }

  switch (LOWORD(WParam)) {
  case 100:
    if (SendMessage((HWND)LParam, BM_GETCHECK, 0, 0) == BST_UNCHECKED) {
      SendMessageW((HWND)LParam, BM_SETCHECK, BST_CHECKED, 0);
      SendMessageW(FindWindowExW(HWnd, NULL, L"Button", L"Erase"), BM_SETCHECK,
                   BST_UNCHECKED, 0);
      SendMessageW(FindWindowExW(HWnd, NULL, L"Button", L"Rectangle"),
                   BM_SETCHECK, BST_UNCHECKED, 0);
      SendMessageW(FindWindowExW(HWnd, NULL, L"Button", L"Ellipse"),
                   BM_SETCHECK, BST_UNCHECKED, 0);
      DrawMode = 1;
    } else {
      SendMessageW((HWND)LParam, BM_SETCHECK, BST_UNCHECKED, 0);
      DrawMode = 0;
    }
    break;
  case 101:
    if (SendMessage((HWND)LParam, BM_GETCHECK, 0, 0) == BST_UNCHECKED) {
      SendMessageW((HWND)LParam, BM_SETCHECK, BST_CHECKED, 0);
      SendMessageW(FindWindowExW(HWnd, NULL, L"Button", L"Pen"), BM_SETCHECK,
                   BST_UNCHECKED, 0);
      SendMessageW(FindWindowExW(HWnd, NULL, L"Button", L"Rectangle"),
                   BM_SETCHECK, BST_UNCHECKED, 0);
      SendMessageW(FindWindowExW(HWnd, NULL, L"Button", L"Ellipse"),
                   BM_SETCHECK, BST_UNCHECKED, 0);
      DrawMode = 2;
    } else {
      SendMessageW((HWND)LParam, BM_SETCHECK, BST_UNCHECKED, 0);
      DrawMode = 0;
    }
    break;
  case 102:
    if (SendMessage((HWND)LParam, BM_GETCHECK, 0, 0) == BST_UNCHECKED) {
      SendMessageW((HWND)LParam, BM_SETCHECK, BST_CHECKED, 0);
      SendMessageW(FindWindowExW(HWnd, NULL, L"Button", L"Erase"), BM_SETCHECK,
                   BST_UNCHECKED, 0);
      SendMessageW(FindWindowExW(HWnd, NULL, L"Button", L"Pen"), BM_SETCHECK,
                   BST_UNCHECKED, 0);
      SendMessageW(FindWindowExW(HWnd, NULL, L"Button", L"Ellipse"),
                   BM_SETCHECK, BST_UNCHECKED, 0);
      DrawMode = 3;
    } else {
      SendMessageW((HWND)LParam, BM_SETCHECK, BST_UNCHECKED, 0);
      DrawMode = 0;
    }
    break;
  case 103:
    if (SendMessage((HWND)LParam, BM_GETCHECK, 0, 0) == BST_UNCHECKED) {
      SendMessageW((HWND)LParam, BM_SETCHECK, BST_CHECKED, 0);
      SendMessageW(FindWindowExW(HWnd, NULL, L"Button", L"Erase"), BM_SETCHECK,
                   BST_UNCHECKED, 0);
      SendMessageW(FindWindowExW(HWnd, NULL, L"Button", L"Pen"), BM_SETCHECK,
                   BST_UNCHECKED, 0);
      SendMessageW(FindWindowExW(HWnd, NULL, L"Button", L"Rectangle"),
                   BM_SETCHECK, BST_UNCHECKED, 0);
      DrawMode = 4;
    } else {
      SendMessageW((HWND)LParam, BM_SETCHECK, BST_UNCHECKED, 0);
      DrawMode = 0;
    }
    break;
  case 400:
    InvalidateRect(HWnd, nullptr, true);
    break;
  }
}

void handleScroll(HWND HWnd, WPARAM WParam, LPARAM LParam) {
  switch (LOWORD(WParam)) {
  case SB_LINELEFT:
    SetScrollPos((HWND)LParam, SB_CTL,
                 max(0, GetScrollPos((HWND)LParam, SB_CTL) - 1), true);
    break;
  case SB_PAGELEFT:
    SetScrollPos((HWND)LParam, SB_CTL,
                 max(0, GetScrollPos((HWND)LParam, SB_CTL) - 5), true);
    break;
  case SB_LINERIGHT:
    SetScrollPos((HWND)LParam, SB_CTL,
                 min(255, GetScrollPos((HWND)LParam, SB_CTL) + 1), true);
    break;
  case SB_PAGERIGHT:
    SetScrollPos((HWND)LParam, SB_CTL,
                 min(255, GetScrollPos((HWND)LParam, SB_CTL) + 5), true);
    break;
  case SB_THUMBTRACK:
    SetScrollPos((HWND)LParam, SB_CTL, HIWORD(WParam), true);
    break;
  }
}