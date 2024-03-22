// NotepadReinvention.cpp : Defines the entry point for the application.
//

#include "NotepadReinvention.h"

#include "framework.h"

#include "TextManager.h"

#define MAX_LOADSTRING 100
#define FONT_SIZE 12

HINSTANCE hInst;
WCHAR szTitle[MAX_LOADSTRING];
WCHAR szWindowClass[MAX_LOADSTRING];
HFONT defaultFont;

HWND mainWindow;

TextManager TextBoard;

long TextPosX = 0;
long TextPosY = 0;

int CurrentXPos;
int CurrentYPos;

int CaretPosXByChar = 0;
int CaretPosYByChar = 0;

int TempCaretPosXChar;
int TempCaretPosYChar;

ATOM MyRegisterClass(HINSTANCE hInstance);
BOOL InitInstance(HINSTANCE, int);
LRESULT CALLBACK WndProc(HWND, UINT, WPARAM, LPARAM);
INT_PTR CALLBACK About(HWND, UINT, WPARAM, LPARAM);

void GetFontSize(HWND hWnd, HDC deviceContext, WCHAR character, int *height,
                 int *width);
void UpdateCaret(HWND hWnd, int MousePosX, int MousePosY);
void UpdateScrollRange(HWND hWnd);
void GetWindowSize(HWND hWnd, int *WindowHeight, int *WindowWidth);
BOOL TryOpen();
BOOL TrySave();

int APIENTRY wWinMain(_In_ HINSTANCE hInstance,
                      _In_opt_ HINSTANCE hPrevInstance, _In_ LPWSTR lpCmdLine,
                      _In_ int nCmdShow) {
  UNREFERENCED_PARAMETER(hPrevInstance);
  UNREFERENCED_PARAMETER(lpCmdLine);

  LoadStringW(hInstance, IDS_APP_TITLE, szTitle, MAX_LOADSTRING);
  LoadStringW(hInstance, IDC_NOTEPADREINVENTION, szWindowClass, MAX_LOADSTRING);
  MyRegisterClass(hInstance);

  if (!InitInstance(hInstance, nCmdShow)) {
    return FALSE;
  }

  HACCEL hAccelTable =
      LoadAccelerators(hInstance, MAKEINTRESOURCE(IDC_NOTEPADREINVENTION));

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
  wcex.hIcon = LoadIcon(hInstance, MAKEINTRESOURCE(IDI_NOTEPADREINVENTION));
  wcex.hCursor = LoadCursor(nullptr, IDC_ARROW);
  wcex.hbrBackground = (HBRUSH)(COLOR_WINDOW + 1);
  wcex.lpszMenuName = MAKEINTRESOURCEW(IDC_NOTEPADREINVENTION);
  wcex.lpszClassName = szWindowClass;
  wcex.hIconSm = LoadIcon(wcex.hInstance, MAKEINTRESOURCE(IDI_SMALL));

  return RegisterClassExW(&wcex);
}

BOOL InitInstance(HINSTANCE hInstance, int nCmdShow) {
  hInst = hInstance;

  mainWindow = CreateWindowW(
      szWindowClass, szTitle, WS_OVERLAPPEDWINDOW | WS_HSCROLL | WS_VSCROLL,
      CW_USEDEFAULT, 0, CW_USEDEFAULT, 0, nullptr, nullptr, hInstance, nullptr);

  if (!mainWindow) {
    return FALSE;
  }

  ShowWindow(mainWindow, nCmdShow);
  UpdateWindow(mainWindow);

  return TRUE;
}

LRESULT CALLBACK WndProc(HWND hWnd, UINT message, WPARAM wParam,
                         LPARAM lParam) {
  HDC hdc;
  SCROLLINFO ScrollInfo;
  TEXTMETRIC TextMetric;
  POINT CaretPos;
  int WindowHeight, WindowWidth;
  int CharHeight, CharWidth;

  switch (message) {
  case WM_CREATE:
    defaultFont = CreateFont(FONT_SIZE, 0, 0, 0, 0, 0, 0, 0, HANGEUL_CHARSET, 0,
                             0, 0, 0, _T("����ü"));
    SendMessage(hWnd, WM_SETFONT, (WPARAM)defaultFont, TRUE);
    TextBoard = TextManager();
    break;
  case WM_SIZE:
    UpdateScrollRange(hWnd);
    break;
  case WM_HSCROLL:
    ScrollInfo.cbSize = sizeof(ScrollInfo);
    ScrollInfo.fMask = SIF_ALL;

    GetScrollInfo(hWnd, SB_HORZ, &ScrollInfo);
    CurrentXPos = ScrollInfo.nPos;

    switch (LOWORD(wParam)) {
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
      ScrollInfo.nPos = HIWORD(wParam);
      break;
    default:
      break;
    }

    ScrollInfo.fMask = SIF_POS;
    SetScrollInfo(hWnd, SB_HORZ, &ScrollInfo, true);
    GetScrollInfo(hWnd, SB_HORZ, &ScrollInfo);

    if (ScrollInfo.nPos != TextPosX) {
      ScrollWindow(hWnd, 15 * (CurrentXPos - ScrollInfo.nPos), 0, NULL, NULL);
      TextPosX = -ScrollInfo.nPos;
      InvalidateRect(hWnd, nullptr, false);
    }

    break;
  case WM_VSCROLL:
    ScrollInfo.cbSize = sizeof(ScrollInfo);
    ScrollInfo.fMask = SIF_ALL;

    GetScrollInfo(hWnd, SB_VERT, &ScrollInfo);
    CurrentYPos = ScrollInfo.nPos;

    switch (LOWORD(wParam)) {
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
      ScrollInfo.nPos = HIWORD(wParam);
      break;
    default:
      break;
    }

    ScrollInfo.fMask = SIF_POS;
    SetScrollInfo(hWnd, SB_VERT, &ScrollInfo, true);
    GetScrollInfo(hWnd, SB_VERT, &ScrollInfo);

    if (ScrollInfo.nPos != TextPosY) {
      ScrollWindow(hWnd, 15 * (CurrentYPos - ScrollInfo.nPos), 0, NULL, NULL);
      TextPosY = -ScrollInfo.nPos;
      InvalidateRect(hWnd, nullptr, false);
    }

    break;
  case WM_COMMAND: {
    int wmId = LOWORD(wParam);
    // Parse the menu selections:
    switch (wmId) {
    case IDM_ABOUT:
      DialogBox(hInst, MAKEINTRESOURCE(IDD_ABOUTBOX), hWnd, About);
      break;
    case IDM_EXIT:
      DestroyWindow(hWnd);
      break;
    case IDM_FILE_OPEN:
      if (!TryOpen()) {
        MessageBox(mainWindow, L"������ �� �� �����ϴ�!", L"����", MB_OK);
      }

      InvalidateRect(hWnd, nullptr, false);
      break;
    case IDM_FILE_SAVE:
      if (!TrySave()) {
        MessageBox(mainWindow, L"������ ������ �� �����ϴ�!", L"����", MB_OK);
      }
      break;
    default:
      return DefWindowProc(hWnd, message, wParam, lParam);
    }
  } break;
  case WM_LBUTTONDOWN:
    if (TextBoard.getText(0).has_value()) {
      if (TextBoard.getText(0).value().length() == 0) {
        break;
      }
    } else {
      break;
    }

    UpdateCaret(hWnd, GET_X_LPARAM(lParam), GET_Y_LPARAM(lParam));
    InvalidateRect(hWnd, nullptr, false);
    UpdateWindow(hWnd);
    break;
  case WM_CHAR:
    if (wParam < 32 || wParam == 127) {
      break;
    }

    TextBoard.handleWrite((wchar_t)wParam, CaretPosXByChar, CaretPosYByChar);
    ++CaretPosXByChar;

    GetCaretPos(&CaretPos);
    GetWindowSize(hWnd, nullptr, &WindowWidth);

    hdc = GetDC(hWnd);
    GetFontSize(hWnd, hdc, wParam, &CharHeight, &CharWidth);
    ReleaseDC(hWnd, hdc);

    if (CaretPos.x >= WindowWidth - 20) {
      TextPosX -= CharWidth;
      UpdateScrollRange(hWnd);

      memset(&ScrollInfo, 0, sizeof(ScrollInfo));
      ScrollInfo.cbSize = sizeof(ScrollInfo);
      ScrollInfo.fMask = SIF_ALL;

      GetScrollInfo(hWnd, SB_HORZ, &ScrollInfo);
      ScrollInfo.nPos = ScrollInfo.nMax;
      SetScrollInfo(hWnd, SB_HORZ, &ScrollInfo, true);
    }

    InvalidateRect(hWnd, nullptr, false);
    break;
  case WM_KEYDOWN:
    switch (wParam) {
    case VK_UP:
      if (CaretPosYByChar > 0) {
        --CaretPosYByChar;
      }

      if (TextBoard.getText(CaretPosYByChar).has_value()) {
        if (CaretPosXByChar >
            TextBoard.getText(CaretPosYByChar).value().length() - 1) {
          CaretPosXByChar = TextBoard.getText(CaretPosYByChar).value().length();
        }
      }

      GetCaretPos(&CaretPos);
      GetWindowSize(hWnd, &WindowHeight, nullptr);

      hdc = GetDC(hWnd);
      GetTextMetrics(hdc, &TextMetric);
      ReleaseDC(hWnd, hdc);

      if (CaretPos.y < 20 && TextPosY < 0) {
        TextPosY += TextMetric.tmHeight;
        UpdateScrollRange(hWnd);

        memset(&ScrollInfo, 0, sizeof(ScrollInfo));
        ScrollInfo.cbSize = sizeof(ScrollInfo);
        ScrollInfo.fMask = SIF_ALL;

        GetScrollInfo(hWnd, SB_VERT, &ScrollInfo);
        ScrollInfo.nPos -= TextMetric.tmHeight + TextMetric.tmExternalLeading +
                           TextMetric.tmInternalLeading;
        SetScrollInfo(hWnd, SB_VERT, &ScrollInfo, true);
      }

      InvalidateRect(hWnd, nullptr, false);
      break;
    case VK_DOWN:
      if (CaretPosYByChar < TextBoard.size() - 1) {
        ++CaretPosYByChar;
      }

      if (TextBoard.getText(CaretPosYByChar).has_value()) {
        if (CaretPosXByChar >
            TextBoard.getText(CaretPosYByChar).value().length() - 1) {
          CaretPosXByChar = TextBoard.getText(CaretPosYByChar).value().length();
        }
      }

      GetCaretPos(&CaretPos);
      GetWindowSize(hWnd, &WindowHeight, nullptr);

      hdc = GetDC(hWnd);
      GetTextMetrics(hdc, &TextMetric);
      ReleaseDC(hWnd, hdc);

      if (CaretPos.y > WindowHeight - 20 &&
          CaretPosYByChar < TextBoard.size()) {
        TextPosY -= TextMetric.tmHeight;
        UpdateScrollRange(hWnd);

        memset(&ScrollInfo, 0, sizeof(ScrollInfo));
        ScrollInfo.cbSize = sizeof(ScrollInfo);
        ScrollInfo.fMask = SIF_ALL;

        GetScrollInfo(hWnd, SB_VERT, &ScrollInfo);
        ScrollInfo.nPos += TextMetric.tmHeight + TextMetric.tmExternalLeading +
                           TextMetric.tmInternalLeading;
        SetScrollInfo(hWnd, SB_VERT, &ScrollInfo, true);
      }

      InvalidateRect(hWnd, nullptr, false);
      break;
    case VK_LEFT:
      if (CaretPosXByChar > 0) {
        --CaretPosXByChar;
      }

      GetCaretPos(&CaretPos);
      GetWindowSize(hWnd, nullptr, &WindowWidth);

      hdc = GetDC(hWnd);
      GetFontSize(hWnd, hdc,
                  TextBoard.getText(CaretPosYByChar).value()[CaretPosXByChar],
                  &CharHeight, &CharWidth);
      ReleaseDC(hWnd, hdc);

      if (CaretPos.x <= 20 && TextPosX < 0) {
        TextPosX += CharWidth;
        UpdateScrollRange(hWnd);

        memset(&ScrollInfo, 0, sizeof(ScrollInfo));
        ScrollInfo.cbSize = sizeof(ScrollInfo);
        ScrollInfo.fMask = SIF_ALL;

        GetScrollInfo(hWnd, SB_HORZ, &ScrollInfo);
        ScrollInfo.nPos = ScrollInfo.nMin;
        SetScrollInfo(hWnd, SB_HORZ, &ScrollInfo, true);
      }

      InvalidateRect(hWnd, nullptr, false);
      break;
    case VK_RIGHT:
      if (TextBoard.getText(CaretPosYByChar).has_value()) {
        if (CaretPosXByChar <
            TextBoard.getText(CaretPosYByChar).value().length()) {
          ++CaretPosXByChar;
        }
      }

      GetCaretPos(&CaretPos);
      GetWindowSize(hWnd, nullptr, &WindowWidth);

      hdc = GetDC(hWnd);
      GetFontSize(hWnd, hdc,
                  TextBoard.getText(CaretPosYByChar).value()[CaretPosXByChar],
                  &CharHeight, &CharWidth);
      ReleaseDC(hWnd, hdc);

      if (CaretPos.x <= WindowWidth - 20 &&
          CaretPosXByChar <
              TextBoard.getText(CaretPosYByChar).value().length()) {
        TextPosX -= CharWidth;
        UpdateScrollRange(hWnd);

        memset(&ScrollInfo, 0, sizeof(ScrollInfo));
        ScrollInfo.cbSize = sizeof(ScrollInfo);
        ScrollInfo.fMask = SIF_ALL;

        GetScrollInfo(hWnd, SB_HORZ, &ScrollInfo);
        ScrollInfo.nPos = ScrollInfo.nMax;
        SetScrollInfo(hWnd, SB_HORZ, &ScrollInfo, true);
      }

      InvalidateRect(hWnd, nullptr, false);
      break;
    case VK_BACK:
      TempCaretPosXChar = CaretPosXByChar;
      TempCaretPosYChar = CaretPosYByChar;

      if (CaretPosXByChar > 0) {
        --CaretPosXByChar;
      } else if (CaretPosYByChar > 0) {
        --CaretPosYByChar;

        if (TextBoard.getText(CaretPosYByChar).has_value()) {
          CaretPosXByChar = TextBoard.getText(CaretPosYByChar).value().length();
        } else {
          CaretPosXByChar = 0;
        }
      } else {
        // Do nothing
      }

      TextBoard.handleHitBackspace(TempCaretPosXChar, TempCaretPosYChar);

      GetCaretPos(&CaretPos);
      GetWindowSize(hWnd, nullptr, &WindowWidth);

      hdc = GetDC(hWnd);
      GetFontSize(hWnd, hdc,
                  TextBoard.getText(CaretPosYByChar).value()[CaretPosXByChar],
                  &CharHeight, &CharWidth);
      ReleaseDC(hWnd, hdc);

      if (CaretPos.x >= 20 && TextPosX < 0) {
        TextPosX += CharWidth;
        UpdateScrollRange(hWnd);

        memset(&ScrollInfo, 0, sizeof(ScrollInfo));
        ScrollInfo.cbSize = sizeof(ScrollInfo);
        ScrollInfo.fMask = SIF_ALL;

        GetScrollInfo(hWnd, SB_HORZ, &ScrollInfo);
        ScrollInfo.nPos = ScrollInfo.nMax;
        SetScrollInfo(hWnd, SB_HORZ, &ScrollInfo, true);
      }

      InvalidateRect(hWnd, nullptr, false);

      break;
    case VK_TAB:
      TextBoard.handleHitTab(CaretPosXByChar, CaretPosYByChar);

      CaretPosXByChar += 8;

      InvalidateRect(hWnd, nullptr, false);

      break;
    case VK_RETURN:
      TextBoard.handleHitEnter(CaretPosXByChar, CaretPosYByChar);

      CaretPosXByChar = 0;
      ++CaretPosYByChar;

      GetCaretPos(&CaretPos);
      GetWindowSize(hWnd, &WindowHeight, nullptr);

      hdc = GetDC(hWnd);
      GetTextMetrics(hdc, &TextMetric);
      ReleaseDC(hWnd, hdc);

      if (CaretPos.y >= WindowHeight - 20) {
        TextPosY -= TextMetric.tmHeight + TextMetric.tmExternalLeading +
                    TextMetric.tmInternalLeading;
        UpdateScrollRange(hWnd);

        memset(&ScrollInfo, 0, sizeof(ScrollInfo));
        ScrollInfo.cbSize = sizeof(ScrollInfo);
        ScrollInfo.fMask = SIF_ALL;

        GetScrollInfo(hWnd, SB_VERT, &ScrollInfo);
        ScrollInfo.nPos = ScrollInfo.nMax;
        SetScrollInfo(hWnd, SB_VERT, &ScrollInfo, true);
      }

      if (TextPosX < 0) {
        TextPosX = 0;
        UpdateScrollRange(hWnd);

        memset(&ScrollInfo, 0, sizeof(ScrollInfo));
        ScrollInfo.cbSize = sizeof(ScrollInfo);
        ScrollInfo.fMask = SIF_ALL;

        GetScrollInfo(hWnd, SB_HORZ, &ScrollInfo);
        ScrollInfo.nPos = ScrollInfo.nMin;
        SetScrollInfo(hWnd, SB_HORZ, &ScrollInfo, true);
      }

      InvalidateRect(hWnd, nullptr, false);
      break;
    case VK_HOME:
      CaretPosXByChar = 0;
      InvalidateRect(hWnd, nullptr, false);
      break;
    case VK_END:
      if (TextBoard.getText(CaretPosYByChar).has_value()) {
        CaretPosXByChar = TextBoard.getText(CaretPosYByChar).value().length();
      }
      InvalidateRect(hWnd, nullptr, false);
      break;
    case VK_DELETE:
      TextBoard.handleHitDelete(CaretPosXByChar, CaretPosYByChar);

      InvalidateRect(hWnd, nullptr, false);
      break;
    case VK_INSERT:
      TextBoard.handleHitInsert();
      break;
    default:
      break;
    }
    UpdateScrollRange(hWnd);
    break;
  case WM_PAINT: {
    /* ���� ���÷��� ���� */
    PAINTSTRUCT ps;
    hdc = BeginPaint(hWnd, &ps);
    HDC hdcBuffer = CreateCompatibleDC(hdc);
    HBITMAP hbmBuffer =
        CreateCompatibleBitmap(hdc, ps.rcPaint.right, ps.rcPaint.bottom);
    HBITMAP hbmOldBuffer = (HBITMAP)SelectObject(hdcBuffer, hbmBuffer);
    TEXTMETRICW TextMetric;

    PatBlt(hdcBuffer, 0, 0, ps.rcPaint.right, ps.rcPaint.bottom, WHITENESS);

    GetTextMetrics(hdcBuffer, &TextMetric);

    /* ���� �׸��� */
    for (int i = 0; i < TextBoard.size(); ++i) {
      std::optional<std::wstring> str = TextBoard.getText(i);

      if (!str.has_value()) {
        continue;
      }

      TextOutW(hdcBuffer, TextPosX, TextPosY + (i * TextMetric.tmHeight),
               str.value().c_str(), str.value().length());
    }

    /* Ŀ�� ��ġ�ϱ� */
    if (!TextBoard.getText(CaretPosYByChar).has_value()) {
      SetCaretPos(0, TextMetric.tmHeight * CaretPosYByChar);
      ShowCaret(hWnd);
    } else {
      std::wstring Text = TextBoard.getText(CaretPosYByChar).value();
      int CaretPosXByPixel = 0;
      int CharWidth, CharHeight;

      for (int i = 0; i < CaretPosXByChar; ++i) {
        if (CaretPosXByChar > Text.length()) {
          break;
        }

        GetFontSize(hWnd, hdc, Text[i], &CharHeight, &CharWidth);
        CaretPosXByPixel += CharWidth;
      }

      SetCaretPos(CaretPosXByPixel + TextPosX,
                  TextMetric.tmHeight * CaretPosYByChar + TextPosY);
      ShowCaret(hWnd);
    }

    /* ���� ���÷��̷� ���� */
    BitBlt(hdc, 0, 0, ps.rcPaint.right, ps.rcPaint.bottom, hdcBuffer, 0, 0,
           SRCCOPY);
    SelectObject(hdcBuffer, hbmOldBuffer);
    DeleteObject(hbmBuffer);
    DeleteDC(hdcBuffer);

    EndPaint(hWnd, &ps);
  } break;
  case WM_SETFOCUS:
    CreateCaret(hWnd, (HBITMAP)NULL, 2, 15);
    ShowCaret(hWnd);
    break;
  case WM_KILLFOCUS:
    DestroyCaret();
    break;
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

void UpdateCaret(HWND hWnd, int MousePosX, int MousePosY) {
  HDC DeviceContext;
  TEXTMETRIC TextMetric;
  int newIndexX = 0;
  int newIndexY = 0;
  int tempPosX = 0;

  int textHeight = 0;
  int textWidth = 0;

  DeviceContext = GetDC(hWnd);

  GetTextMetrics(DeviceContext, &TextMetric);

  for (int i = 0; i < TextBoard.size(); ++i) {
    if (MousePosY < i * TextMetric.tmHeight + TextPosY) {
      --newIndexY;
      break;
    }

    ++newIndexY;
  }

  newIndexY =
      (newIndexY >= TextBoard.size() ? TextBoard.size() - 1 : newIndexY);
  std::wstring CurrentLine = TextBoard.getText(newIndexY).value();

  for (int i = 0; i < CurrentLine.length(); ++i) {
    GetFontSize(hWnd, DeviceContext, CurrentLine[i], &textHeight, &textWidth);

    if (MousePosX < tempPosX + textWidth + TextPosX) {
      --newIndexX;
      break;
    }

    tempPosX += textWidth;
    ++newIndexX;
  }

  CaretPosXByChar = newIndexX;
  CaretPosYByChar = newIndexY;

  ReleaseDC(hWnd, DeviceContext);
}

void GetWindowSize(HWND hWnd, int *height, int *width) {
  RECT WindowRect;
  GetClientRect(hWnd, &WindowRect);

  if (height != nullptr) {
    *height = WindowRect.bottom - WindowRect.top;
  }

  if (width != nullptr) {
    *width = WindowRect.right - WindowRect.left;
  }
}

void UpdateScrollRange(HWND hWnd) {
  HDC DeviceContext;
  SCROLLINFO ScrollInfo;
  RECT WindowRect;
  TEXTMETRIC TextMetric;
  int TextBoardMaxWidth = 0;
  int WindowHeight, WindowWidth;

  DeviceContext = GetDC(hWnd);
  GetTextMetrics(DeviceContext, &TextMetric);

  GetClientRect(hWnd, &WindowRect);

  WindowHeight = WindowRect.bottom - WindowRect.top;
  WindowWidth = WindowRect.right - WindowRect.left;

  // ���� ��ũ�ѹ�
  ScrollInfo.cbSize = sizeof(ScrollInfo);
  ScrollInfo.fMask = SIF_RANGE | SIF_PAGE;
  ScrollInfo.nMin = 0;
  ScrollInfo.nMax = TextBoard.size() * TextMetric.tmHeight - WindowHeight;

  if (ScrollInfo.nMax < 0) {
    ScrollInfo.nMax = 0;
  }

  if (ScrollInfo.nMax != 0) {
    ScrollInfo.nPage = ScrollInfo.nMax / TextBoard.size();
  } else {
    ScrollInfo.nPage = 0;
  }

  SetScrollInfo(hWnd, SB_VERT, &ScrollInfo, true);

  if (ScrollInfo.nMax == 0 && ScrollInfo.nPage == 0) {
    TextPosY = 0;
  }

  // ���� �ִ� �ʺ� ���ϱ�
  std::wstring LongestString = TextBoard.getLongestLine();
  int FontHeight, FontWidth;

  for (auto it = LongestString.begin(); it < LongestString.end(); ++it) {
    GetFontSize(hWnd, DeviceContext, *it, &FontHeight, &FontWidth);
    TextBoardMaxWidth += FontWidth;
  }

  // ���� ��ũ�ѹ�
  ScrollInfo.cbSize = sizeof(ScrollInfo);
  ScrollInfo.fMask = SIF_RANGE | SIF_PAGE;
  ScrollInfo.nMin = 0;
  ScrollInfo.nMax = TextBoardMaxWidth - WindowWidth;

  if (ScrollInfo.nMax < 0) {
    ScrollInfo.nMax = 0;
  }

  if (ScrollInfo.nMax != 0) {
    ScrollInfo.nPage = ScrollInfo.nMax / LongestString.length();
  } else {
    ScrollInfo.nPage = 0;
  }

  SetScrollInfo(hWnd, SB_HORZ, &ScrollInfo, true);

  if (ScrollInfo.nMax == 0 && ScrollInfo.nPage == 0) {
    TextPosX = 0;
  }

  ReleaseDC(hWnd, DeviceContext);
}

void GetFontSize(HWND hWnd, HDC deviceContext, WCHAR character, int *height,
                 int *width) {
  TEXTMETRICW TextMetric;
  ABC Abc;

  GetCharABCWidths(deviceContext, (UINT)character, (UINT)character, &Abc);
  GetTextMetrics(deviceContext, &TextMetric);

  *height = TextMetric.tmHeight;
  *width = Abc.abcA + Abc.abcB + Abc.abcC;
}

BOOL TryOpen() {
  OPENFILENAME openFileName;
  TCHAR lpstrFile[256] = L"";

  memset(&openFileName, 0, sizeof(OPENFILENAME));
  openFileName.lStructSize = sizeof(OPENFILENAME);
  openFileName.hwndOwner = mainWindow;
  openFileName.lpstrFile = lpstrFile;
  openFileName.nMaxFile = 256;
  openFileName.lpstrInitialDir = L".";
  openFileName.lpstrDefExt = L"txt";
  openFileName.lpstrFilter = L"�ؽ�Ʈ ����\0*.txt\0��� ����\0*.*";

  if (GetOpenFileName(&openFileName) == 0) {
    return false;
  }

  std::wifstream ReadStream(lpstrFile);

  ReadStream.imbue(std::locale("kor"));

  if (!ReadStream.is_open()) {
    ReadStream.close();
    return false;
  }

  TextBoard.clear();

  std::wstring Line;

  while (getline(ReadStream, Line)) {
    TextBoard.appendString(Line);
  }

  return true;
}

BOOL TrySave() {
  OPENFILENAME saveFileName;
  TCHAR lpstrFile[256] = L"";

  memset(&saveFileName, 0, sizeof(OPENFILENAME));
  saveFileName.lStructSize = sizeof(OPENFILENAME);
  saveFileName.hwndOwner = mainWindow;
  saveFileName.lpstrFile = lpstrFile;
  saveFileName.nMaxFile = 256;
  saveFileName.lpstrInitialDir = L".";
  saveFileName.lpstrDefExt = L"txt";
  saveFileName.lpstrFilter = L"�ؽ�Ʈ ����\0*.txt\0��� ����\0*.*";

  if (GetSaveFileName(&saveFileName) == 0) {
    return false;
  }

  std::wofstream WriteStream(lpstrFile);

  WriteStream.imbue(std::locale("kor"));

  if (!WriteStream.is_open()) {
    WriteStream.close();
    return false;
  }

  for (auto it = TextBoard.begin(); it < TextBoard.end(); ++it) {
    WriteStream << *it << std::endl;
  }

  WriteStream.close();

  return true;
}