#include "Notepad.h"

#include "framework.h"

#define MAX_LOADSTRING 100

HINSTANCE hInst;
WCHAR szTitle[MAX_LOADSTRING];
WCHAR szWindowClass[MAX_LOADSTRING];
HWND mainWindow;
HWND editorControl;
WNDPROC currentEditorProc;
BOOL insertMode = FALSE;
BOOL hangulFlag = FALSE;
long long currentTextSize = 0;

ATOM MyRegisterClass(HINSTANCE hInstance);
BOOL InitInstance(HINSTANCE, int);
LRESULT CALLBACK WndProc(HWND, UINT, WPARAM, LPARAM);
LRESULT CALLBACK EditorProc(HWND, UINT, WPARAM, LPARAM);
INT_PTR CALLBACK About(HWND, UINT, WPARAM, LPARAM);
BOOL TryOpen();
BOOL TrySave();

int APIENTRY wWinMain(_In_ HINSTANCE hInstance,
                      _In_opt_ HINSTANCE hPrevInstance, _In_ LPWSTR lpCmdLine,
                      _In_ int nCmdShow) {
  UNREFERENCED_PARAMETER(hPrevInstance);
  UNREFERENCED_PARAMETER(lpCmdLine);

  LoadStringW(hInstance, IDS_APP_TITLE, szTitle, MAX_LOADSTRING);
  LoadStringW(hInstance, IDC_NOTEPAD, szWindowClass, MAX_LOADSTRING);
  MyRegisterClass(hInstance);

  if (!InitInstance(hInstance, nCmdShow)) {
    return FALSE;
  }

  HACCEL hAccelTable =
      LoadAccelerators(hInstance, MAKEINTRESOURCE(IDC_NOTEPAD));

  MSG msg;

  // Main message loop:
  while (GetMessage(&msg, nullptr, 0, 0)) {
    if (!TranslateAccelerator(mainWindow, hAccelTable, &msg)) {
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
  wcex.hIcon = LoadIcon(hInstance, MAKEINTRESOURCE(IDI_NOTEPAD));
  wcex.hCursor = LoadCursor(nullptr, IDC_ARROW);
  wcex.hbrBackground = (HBRUSH)(COLOR_WINDOW + 1);
  wcex.lpszMenuName = MAKEINTRESOURCEW(IDC_NOTEPAD);
  wcex.lpszClassName = szWindowClass;
  wcex.hIconSm = LoadIcon(wcex.hInstance, MAKEINTRESOURCE(IDI_SMALL));

  return RegisterClassExW(&wcex);
}

BOOL InitInstance(HINSTANCE hInstance, int nCmdShow) {
  hInst = hInstance;

  mainWindow =
      CreateWindowW(szWindowClass, szTitle, WS_OVERLAPPEDWINDOW, CW_USEDEFAULT,
                    0, CW_USEDEFAULT, 0, nullptr, nullptr, hInstance, nullptr);

  if (!mainWindow) {
    return FALSE;
  }

  ShowWindow(mainWindow, nCmdShow);
  UpdateWindow(mainWindow);

  return TRUE;
}

LRESULT CALLBACK EditorProc(HWND hWnd, UINT message, WPARAM wParam,
                            LPARAM lParam) {
  DWORD start, end, textEnd;

  if (insertMode) {
    if (message == WM_CHAR && wParam == VK_BACK) {
      return CallWindowProc(currentEditorProc, hWnd, message, wParam, lParam);
    }

    if (message == WM_CHAR && wParam == VK_DELETE) {
      return CallWindowProc(currentEditorProc, hWnd, message, wParam, lParam);
    }

    if (message == WM_IME_ENDCOMPOSITION ||
        (message == WM_CHAR && !hangulFlag)) {
      SendMessage(editorControl, EM_GETSEL, (WPARAM)&start, (LPARAM)&end);
      textEnd = SendMessage(editorControl, WM_GETTEXTLENGTH, 0, 0);

      if (start == end && start < textEnd) {
        SendMessage(editorControl, EM_SETSEL, start, start + 1);
        SendMessage(editorControl, EM_REPLACESEL, TRUE, (LPARAM) "");
      }
    }
  }

  if (message == WM_IME_NOTIFY) {
    hangulFlag = !hangulFlag;
  }

  return CallWindowProc(currentEditorProc, hWnd, message, wParam, lParam);
}

LRESULT CALLBACK WndProc(HWND hWnd, UINT message, WPARAM wParam,
                         LPARAM lParam) {
  switch (message) {
    case WM_CREATE:
      editorControl = CreateWindowExW(
          0, L"EDIT", NULL,
          WS_CHILD | WS_VISIBLE | WS_VSCROLL | WS_HSCROLL | ES_LEFT |
              ES_MULTILINE | ES_AUTOVSCROLL | ES_AUTOHSCROLL,
          0, 0, 0, 0, hWnd, (HMENU)100, hInst, NULL);
      currentEditorProc = (WNDPROC)SetWindowLongPtr(editorControl, GWLP_WNDPROC,
                                                    (LONG_PTR)EditorProc);
      break;
    case WM_SIZE:
      MoveWindow(editorControl, 0, 0, LOWORD(lParam), HIWORD(lParam), TRUE);
      break;
    case WM_COMMAND:
      switch (LOWORD(wParam)) {
        case IDM_ABOUT:
          DialogBox(hInst, MAKEINTRESOURCE(IDD_ABOUTBOX), hWnd, About);
          break;
        case IDM_COPY:
          SendMessage(editorControl, WM_COPY, 0, 0);
          break;
        case IDM_CUT:
          SendMessage(editorControl, WM_CUT, 0, 0);
          break;
        case IDM_PASTE:
          SendMessage(editorControl, WM_PASTE, 0, 0);
          break;
        case IDM_SELECT_ALL:
          SendMessage(editorControl, EM_SETSEL, 0, -1);
          break;
        case IDM_EXIT:
          DestroyWindow(hWnd);
          break;
        case IDM_TOGGLE_INSERT:
          insertMode = !insertMode;
          break;
        case IDM_FILE_OPEN:
          if (!TryOpen()) {
            MessageBox(mainWindow, L"파일을 열 수 없습니다!", L"오류", MB_OK);
          }
          break;
        case IDM_FILE_SAVE:
          if (!TrySave()) {
            MessageBox(mainWindow, L"파일을 저장할 수 없습니다!", L"오류",
                       MB_OK);
          }
          break;
        default:
          return DefWindowProc(hWnd, message, wParam, lParam);
      }
      break;
    case WM_SETFOCUS:
      SetFocus(editorControl);
      break;
    case WM_DESTROY:
      PostQuitMessage(0);
      break;
    default:
      return DefWindowProc(hWnd, message, wParam, lParam);
  }
  return 0;
}

// Message handler for about box.
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

// Edit 컨트롤은 내부적으로 UTF16-LE를 사용하기 때문에 처리에 주의할 것
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
  openFileName.lpstrFilter = L"텍스트 파일\0*.txt\0모든 파일\0*.*";

  if (GetOpenFileName(&openFileName) == 0) {
    return false;
  }

  LARGE_INTEGER fileSize;
  HANDLE fileHandle = CreateFile(lpstrFile, GENERIC_READ, 0, NULL,
                                 OPEN_EXISTING, FILE_ATTRIBUTE_NORMAL, NULL);

  if (fileHandle == INVALID_HANDLE_VALUE) {
    return false;
  }

  if (!GetFileSizeEx(fileHandle, &fileSize)) {
    CloseHandle(fileHandle);
    return false;
  }

  WCHAR *buffer = (WCHAR *)malloc((fileSize.QuadPart + 1) * sizeof(WCHAR));
  DWORD bytesRead;

  if (!ReadFile(fileHandle, &buffer[0], fileSize.QuadPart, &bytesRead, NULL)) {
    CloseHandle(fileHandle);
    return false;
  }

  // 널 문자 붙여주기 (널 문자가 없으면 오버플로우가 날 수 있음.)
  *(buffer + (bytesRead / sizeof(WCHAR))) = '\0';

  SendMessage(editorControl, WM_SETTEXT, 0, (LPARAM)buffer);

  free(buffer);
  CloseHandle(fileHandle);

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
  saveFileName.lpstrFilter = L"텍스트 파일\0*.txt\0모든 파일\0*.*";

  if (GetSaveFileName(&saveFileName) == 0) {
    return false;
  }

  long long fileSize = SendMessage(editorControl, WM_GETTEXTLENGTH, 0, 0);

  WCHAR *buffer = (WCHAR *)malloc((fileSize + 1) * sizeof(WCHAR));

  SendMessage(editorControl, WM_GETTEXT, (WPARAM)fileSize + 1, (LPARAM)buffer);

  HANDLE fileHandle = CreateFile(lpstrFile, GENERIC_WRITE, 0, NULL,
                                 CREATE_ALWAYS, FILE_ATTRIBUTE_NORMAL, NULL);

  if (fileHandle == INVALID_HANDLE_VALUE) {
    free(buffer);
    return false;
  }

  DWORD writtenBytes;

  if (!WriteFile(fileHandle, buffer, fileSize * sizeof(WCHAR), &writtenBytes,
                 NULL)) {
    CloseHandle(fileHandle);
    return false;
  }

  free(buffer);
  CloseHandle(fileHandle);

  return true;
}
