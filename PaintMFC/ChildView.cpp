// ChildView.cpp : implementation of the CChildView class
//

#include "pch.h"
#include "framework.h"
#include "PaintMFC.h"
#include "ChildView.h"

#ifdef _DEBUG
#define new DEBUG_NEW
#endif

// CChildView
CChildView::CChildView()
{
}

CChildView::~CChildView()
{
}

BEGIN_MESSAGE_MAP(CChildView, CWnd)
    ON_WM_PAINT()
    ON_WM_LBUTTONDOWN()
    ON_WM_LBUTTONUP()
    ON_WM_MOUSEMOVE()
    ON_COMMAND(ID_EDIT_CLEAR, &CChildView::OnEditClearAll)
    ON_COMMAND(ID_EDIT_ELLIPSE, &CChildView::OnEditEllipse)
    ON_COMMAND(ID_EDIT_ERASER, &CChildView::OnEditEraser)
    ON_COMMAND(ID_EDIT_RECTANGLE, &CChildView::OnEditRectangle)
    ON_COMMAND(ID_EDIT_PEN, &CChildView::OnEditPen)
    ON_COMMAND(ID_EDIT_SIZE_UP, &CChildView::OnEditSizeUp)
    ON_COMMAND(ID_EDIT_SIZE_DOWN, &CChildView::OnEditSizeDown)
    ON_COMMAND(ID_EDIT_FILLSHAPE, &CChildView::OnEditFillshape)
    ON_COMMAND(ID_COLOR_BLUE_MINUS, &CChildView::OnColorBlueMinus)
    ON_COMMAND(ID_COLOR_BLUE_PLUS, &CChildView::OnColorBluePlus)
    ON_COMMAND(ID_COLOR_GREEN_MINUS, &CChildView::OnColorGreenMinus)
    ON_COMMAND(ID_COLOR_GREEN_PLUS, &CChildView::OnColorGreenPlus)
    ON_COMMAND(ID_COLOR_RED_MINUS, &CChildView::OnColorRedMinus)
    ON_COMMAND(ID_COLOR_RED_PLUS, &CChildView::OnColorRedPlus)
    ON_COMMAND(ID_FILE_SAVE, &CChildView::OnFileSave)
    ON_COMMAND(ID_FILE_OPEN, &CChildView::OnFileOpen)
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

    if (m_isDrawing)
    {
        switch (m_drawMode)
        {
        case 2:
            DrawRectangle();
            break;
        case 3:
            DrawEllipse();
            break;
        default:
            Draw();
            break;
        }
    }
}

void CChildView::OnLButtonDown(UINT nFlags, CPoint point)
{
    switch (m_drawMode)
    {
    case 0:
        m_tempVector = CImageVector(1);
        m_tempVector.SetPenConfig(RGB(m_redIntensity, m_greenIntensity, m_blueIntensity), m_penWidth / 5 + 1);
        break;
    case 1:
        m_tempVector = CImageVector(1);
        m_tempVector.SetPenConfig(RGB(255, 255, 255), 40);
        break;
    case 2:
        m_tempVector = CImageVector(2);
        m_tempVector.SetShapeConfig(RGB(m_redIntensity, m_greenIntensity, m_blueIntensity), m_doFillShape);
        break;
    case 3:
        m_tempVector = CImageVector(3);
        m_tempVector.SetShapeConfig(RGB(m_redIntensity, m_greenIntensity, m_blueIntensity), m_doFillShape);
        break;
    }

    m_tempVector.AppendPoint(point);

    if (m_drawMode > 1)
    {
        m_tempMousePos.x = point.x;
        m_tempMousePos.y = point.y;
    }

    m_mousePos = point;
    m_isDrawing = true;

    CWnd::OnLButtonDown(nFlags, point);
}

void CChildView::OnLButtonUp(UINT nFlags, CPoint point)
{
    m_isDrawing = false;
    ClearRectImage();

    m_tempVector.AppendPoint(point);
    m_vectorHistory.push_back(m_tempVector);

    CWnd::OnLButtonUp(nFlags, point);
}

void CChildView::OnMouseMove(UINT nFlags, CPoint point)
{
    if (m_isDrawing)
    {
        m_mousePos = point;

        m_tempVector.AppendPoint(point);

        Invalidate(false);
    }

    CWnd::OnMouseMove(nFlags, point);
}

void CChildView::OnEditClearAll()
{
    Invalidate();
}

void CChildView::OnEditEllipse()
{
    m_drawMode = 3;
}

void CChildView::OnEditEraser()
{
    m_drawMode = 1;
}

void CChildView::OnEditRectangle()
{
    m_drawMode = 2;
}

void CChildView::OnEditPen()
{
    m_drawMode = 0;
}

void CChildView::OnEditSizeUp()
{
    if (m_penWidth < 100)
    {
        m_penWidth += 10;
    }
}

void CChildView::OnEditSizeDown()
{
    if (m_penWidth > 0)
    {
        m_penWidth -= 10;
    }
}

void CChildView::OnEditFillshape()
{
    m_doFillShape = !m_doFillShape;
}

void CChildView::OnColorRedPlus()
{
    if (m_redIntensity < 250)
    {
        m_redIntensity += 10;
    }
}

void CChildView::OnColorRedMinus()
{
    if (m_redIntensity > 0)
    {
        m_redIntensity -= 10;
    }
}

void CChildView::OnColorGreenPlus()
{
    if (m_greenIntensity < 250)
    {
        m_greenIntensity += 10;
    }
}

void CChildView::OnColorGreenMinus()
{
    if (m_greenIntensity > 0)
    {
        m_greenIntensity -= 10;
    }
}

void CChildView::OnColorBluePlus()
{
    if (m_blueIntensity < 250)
    {
        m_blueIntensity += 10;
    }
}

void CChildView::OnColorBlueMinus()
{
    if (m_blueIntensity > 0)
    {
        m_blueIntensity -= 10;
    }
}

void CChildView::OnFileSave()
{
    CFileDialog dlg(FALSE, _T("bmp"), NULL, OFN_OVERWRITEPROMPT | OFN_PATHMUSTEXIST,
        _T("Bitmap File (*.bmp)|*.bmp|JPEG File (*.jpg;*.jpeg)|*.jpg;*.jpeg|PNG File (*.png)|*.png||"), this);
    if (dlg.DoModal() != IDOK)
    {
        return;
    }

    CString filePath = dlg.GetPathName();
    CClientDC deviceContext(this);

    CDC memDC;
    memDC.CreateCompatibleDC(&deviceContext);

    CRect windowRect;
    CWnd::GetClientRect(&windowRect);

    CBitmap windowBitmap;
    windowBitmap.CreateCompatibleBitmap(&deviceContext, windowRect.Width(), windowRect.Height());

    CBitmap* pOldBitmap = memDC.SelectObject(&windowBitmap);
    memDC.BitBlt(0, 0, windowRect.Width(), windowRect.Height(), &deviceContext, 0, 0, SRCCOPY);

    CImage currentImage;
    currentImage.Attach((HBITMAP)windowBitmap.Detach());
    currentImage.Save(filePath);

    memDC.SelectObject(pOldBitmap);
    memDC.DeleteDC();
}

void CChildView::OnFileOpen()
{
    CFileDialog dlg(TRUE, _T("bmp"), NULL, OFN_FILEMUSTEXIST | OFN_PATHMUSTEXIST,
        _T("Bitmap File (*.bmp)|*.bmp|JPEG File (*.jpg;*.jpeg)|*.jpg;*.jpeg|PNG File (*.png)|*.png||"), this);
    if (dlg.DoModal() != IDOK)
    {
        return;
    }

    CString filePath = dlg.GetPathName();
    CImage newImage;
    newImage.Load(filePath);

    CRect windowRect;
    CWnd::GetClientRect(&windowRect);

    CClientDC deviceContext(this);
    CDC memDC;
    memDC.CreateCompatibleDC(&deviceContext);

    CBitmap bitmap;
    bitmap.CreateCompatibleBitmap(&deviceContext, windowRect.Width(), windowRect.Height());

    CBitmap* pOldBitmap = memDC.SelectObject(&bitmap);
    newImage.Draw(memDC.m_hDC, 0, 0, windowRect.Width(), windowRect.Height());
    deviceContext.StretchBlt(0, 0, windowRect.Width(), windowRect.Height(), &memDC, 0, 0, windowRect.Width(), windowRect.Height(), SRCCOPY);

    memDC.SelectObject(pOldBitmap);
    memDC.DeleteDC();
}

void CChildView::Draw()
{
    if (!m_isDrawing)
    {
        return;
    }

    CClientDC deviceContext(this);

    if (m_drawMode == 0)
    {
        HPEN newPen = CreatePen(PS_SOLID, m_penWidth / 5 + 1, RGB(m_redIntensity, m_greenIntensity, m_blueIntensity));
        HGDIOBJ oldPen = deviceContext.SelectObject(newPen);

        deviceContext.MoveTo(m_mousePos);
        deviceContext.LineTo(m_mousePos);

        deviceContext.SelectObject(oldPen);
        DeleteObject(newPen);
        DeleteObject(oldPen);
    }
    else if (m_drawMode == 1)
    {
        HPEN newPen = CreatePen(PS_SOLID, 40, RGB(255, 255, 255));
        HGDIOBJ oldPen = deviceContext.SelectObject(newPen);

        deviceContext.MoveTo(m_mousePos);
        deviceContext.LineTo(m_mousePos);

        deviceContext.SelectObject(newPen);
        DeleteObject(newPen);
        DeleteObject(oldPen);
    }
    else
    {
        return;
    }
}

void CChildView::DrawEllipse()
{
    RestoreRectImage();

    CRect imageRect = CRect(m_tempMousePos, m_mousePos);
    imageRect.NormalizeRect();
    BackupRectImage(imageRect);

    CClientDC deviceContext(this);
    HPEN newPen, oldPen;
    HBRUSH newBrush, oldBrush;

    if (m_doFillShape)
    {
        newPen = CreatePen(PS_SOLID, 2, RGB(0, 0, 0));
        newBrush = CreateSolidBrush(RGB(m_redIntensity, m_greenIntensity, m_blueIntensity));
    }
    else
    {
        newPen = CreatePen(PS_SOLID, 2, RGB(m_redIntensity, m_greenIntensity, m_blueIntensity));
        newBrush = (HBRUSH)GetStockObject(NULL_BRUSH);
    }

    oldPen = (HPEN)deviceContext.SelectObject(newPen);
    oldBrush = (HBRUSH)deviceContext.SelectObject(newBrush);

    CRect shapeRect = CRect(m_tempMousePos, m_mousePos);
    shapeRect.NormalizeRect();
    deviceContext.Ellipse(shapeRect);

    deviceContext.SelectObject(oldPen);
    deviceContext.SelectObject(oldBrush);

    DeleteObject(newPen);
    DeleteObject(oldPen);
    DeleteObject(newBrush);
    DeleteObject(oldBrush);
}

void CChildView::DrawRectangle()
{
    RestoreRectImage();

    CRect imageRect = CRect(m_tempMousePos, m_mousePos);
    imageRect.NormalizeRect();
    BackupRectImage(imageRect);

    CClientDC deviceContext(this);
    HPEN newPen, oldPen;
    HBRUSH newBrush, oldBrush;

    if (m_doFillShape)
    {
        newPen = CreatePen(PS_SOLID, 2, RGB(0, 0, 0));
        newBrush = CreateSolidBrush(RGB(m_redIntensity, m_greenIntensity, m_blueIntensity));
    }
    else
    {
        newPen = CreatePen(PS_SOLID, 2, RGB(m_redIntensity, m_greenIntensity, m_blueIntensity));
        newBrush = (HBRUSH)GetStockObject(NULL_BRUSH);
    }

    oldPen = (HPEN)deviceContext.SelectObject(newPen);
    oldBrush = (HBRUSH)deviceContext.SelectObject(newBrush);

    CRect shapeRect = CRect(m_tempMousePos, m_mousePos);
    shapeRect.NormalizeRect();
    deviceContext.Rectangle(shapeRect);

    deviceContext.SelectObject(oldPen);
    deviceContext.SelectObject(oldBrush);

    DeleteObject(newPen);
    DeleteObject(oldPen);
    DeleteObject(newBrush);
    DeleteObject(oldBrush);
}

void CChildView::BackupRectImage(CRect rect)
{
    rect.InflateRect(2, 2);

    CDC* pDC = GetDC();
    m_bitmapBufferRect = rect;
    m_bitmapBuffer = new BYTE[rect.Width() * rect.Height() * 4];

    CDC memDC;
    memDC.CreateCompatibleDC(pDC);

    CBitmap bitmap;
    bitmap.CreateCompatibleBitmap(pDC, rect.Width(), rect.Height());

    CBitmap* pOldBitmap = memDC.SelectObject(&bitmap);
    memDC.BitBlt(0, 0, rect.Width(), rect.Height(), pDC, rect.left, rect.top, SRCCOPY);

    BITMAPINFO bmi;
    ZeroMemory(&bmi, sizeof(BITMAPINFO));
    bmi.bmiHeader.biSize = sizeof(BITMAPINFOHEADER);
    bmi.bmiHeader.biWidth = rect.Width();
    bmi.bmiHeader.biHeight = -rect.Height();
    bmi.bmiHeader.biPlanes = 1;
    bmi.bmiHeader.biBitCount = 32;
    bmi.bmiHeader.biCompression = BI_RGB;
    bmi.bmiHeader.biSizeImage = 0;

    GetDIBits(memDC.GetSafeHdc(), bitmap, 0, rect.Height(), m_bitmapBuffer, &bmi, DIB_RGB_COLORS);

    memDC.SelectObject(pOldBitmap);
    DeleteObject(bitmap);
    DeleteDC(memDC);
    ReleaseDC(pDC);

    m_isImageBackedUp = true;
}

void CChildView::RestoreRectImage()
{
    if (!m_isImageBackedUp)
    {
        return;
    }

    CDC* pDC = GetDC();
    CDC memDC;
    memDC.CreateCompatibleDC(pDC);

    CBitmap bitmap;
    bitmap.CreateCompatibleBitmap(pDC, m_bitmapBufferRect.Width(), m_bitmapBufferRect.Height());

    CBitmap* pOldBitmap = memDC.SelectObject(&bitmap);

    BITMAPINFO bmi;
    ZeroMemory(&bmi, sizeof(BITMAPINFO));
    bmi.bmiHeader.biSize = sizeof(BITMAPINFOHEADER);
    bmi.bmiHeader.biWidth = m_bitmapBufferRect.Width();
    bmi.bmiHeader.biHeight = -m_bitmapBufferRect.Height();
    bmi.bmiHeader.biPlanes = 1;
    bmi.bmiHeader.biBitCount = 32;
    bmi.bmiHeader.biCompression = BI_RGB;
    bmi.bmiHeader.biSizeImage = 0;

    SetDIBits(memDC.GetSafeHdc(), bitmap, 0, m_bitmapBufferRect.Height(), m_bitmapBuffer, &bmi, DIB_RGB_COLORS);
    pDC->BitBlt(m_bitmapBufferRect.left, m_bitmapBufferRect.top, m_bitmapBufferRect.Width(), m_bitmapBufferRect.Height(),
        &memDC, 0, 0, SRCCOPY);

    memDC.SelectObject(pOldBitmap);
    delete[] m_bitmapBuffer;
    ReleaseDC(pDC);

    m_isImageBackedUp = false;
}

void CChildView::ClearRectImage()
{
    if (!m_isImageBackedUp)
    {
        return;
    }

    delete[] m_bitmapBuffer;
    ZeroMemory(&m_bitmapBufferRect, sizeof(RECT));

    m_isImageBackedUp = false;
}