#include "pch.h"
#include "ImageVector.h"

CImageVector::CImageVector(int shapeKind)
{
	/* shapeKind의 뜻
	 * 1 : 펜 
	 * 2 : 직사각형
	 * 3 : 타원
	 */

	if (shapeKind < 1 || shapeKind > 3)
	{
		AfxThrowInvalidArgException();
	}

	m_shapeKind = shapeKind;
	
	m_color = 0;
	m_penWidth = 0;
	m_fillShape = false;
}

CImageVector::CImageVector()
{
	m_shapeKind = 0;

	m_color = 0;
	m_penWidth = 0;
	m_fillShape = false;
}

void CImageVector::SetPenConfig(COLORREF color, int penWidth)
{
	m_color = color;
	m_penWidth = penWidth;
}

void CImageVector::SetShapeConfig(COLORREF color, bool fillShape)
{
	m_color = color;
	m_fillShape = fillShape;
}

void CImageVector::AppendPoint(CPoint point)
{
	m_points.push_back(point);
}

void CImageVector::Draw(CClientDC& deviceContext)
{
	if (m_shapeKind == 1)
	{
		DrawLine(deviceContext);
	}
	else if (m_shapeKind == 2 || m_shapeKind == 3)
	{
		DrawShape(deviceContext);
	}
	else
	{
		return;
	}
}

void CImageVector::DrawLine(CClientDC& deviceContext)
{
	if (m_shapeKind == 0)
	{
		return;
	}

	HPEN newPen = CreatePen(PS_SOLID, m_penWidth / 5 + 1, m_color);
	HGDIOBJ oldPen = deviceContext.SelectObject(newPen);

	for (auto i = m_points.begin(); i < m_points.end() - 1; ++i)
	{
		deviceContext.MoveTo(*i);
		deviceContext.LineTo(*(i + 1));
	}

	deviceContext.SelectObject(oldPen);
	DeleteObject(newPen);
	DeleteObject(oldPen);
}

void CImageVector::DrawShape(CClientDC& deviceContext)
{
	if (m_shapeKind == 0)
	{
		return;
	}

	HPEN newPen, oldPen;
	HBRUSH newBrush, oldBrush;

	if (m_fillShape)
	{
		newPen = CreatePen(PS_SOLID, 2, RGB(0, 0, 0));
		newBrush = CreateSolidBrush(m_color);
	}
	else
	{
		newPen = CreatePen(PS_SOLID, 2, m_color);
		newBrush = (HBRUSH)GetStockObject(NULL_BRUSH);
	}

	oldPen = (HPEN)deviceContext.SelectObject(newPen);
	oldBrush = (HBRUSH)deviceContext.SelectObject(newBrush);

	CRect shapeRect = CRect(m_points[0], m_points[1]);
	shapeRect.NormalizeRect();
	
	if (m_shapeKind == 2)
	{
		deviceContext.Rectangle(shapeRect);
	}
	else
	{
		deviceContext.Ellipse(shapeRect);
	}

	deviceContext.SelectObject(oldPen);
	deviceContext.SelectObject(oldBrush);

	DeleteObject(newPen);
	DeleteObject(oldPen);
	DeleteObject(newBrush);
	DeleteObject(oldBrush);
}

CString CImageVector::ExportText()
{
	CString data;

	data.Format(_T("%d:%d:%d:%d:"), m_shapeKind, m_color, m_penWidth, m_fillShape);

	for (size_t i = 0; i < m_points.size(); ++i)
	{
		CString pointData;
		pointData.Format(_T("%d,%d"), m_points[i].x, m_points[i].y);
		data += pointData;

		if (i < m_points.size() - 1)
			data += _T("|");
	}

	return data;
}

void CImageVector::ImportText(CString data)
{
	m_points.clear();

	int nTokenPos = 0;
	CString strToken = data.Tokenize(_T(":"), nTokenPos);

	m_shapeKind = _tstoi(strToken);
	strToken = data.Tokenize(_T(":"), nTokenPos);
	m_color = _tstoi(strToken);
	strToken = data.Tokenize(_T(":"), nTokenPos);
	m_penWidth = _tstoi(strToken);
	strToken = data.Tokenize(_T(":"), nTokenPos);
	m_fillShape = _tstoi(strToken) != 0;

	strToken = data.Tokenize(_T("|"), nTokenPos);
	while (!strToken.IsEmpty())
	{
		int commaPos = strToken.Find(_T(','));
		if (commaPos != -1)
		{
			int x = _tstoi(strToken.Left(commaPos));
			int y = _tstoi(strToken.Mid(commaPos + 1));
			m_points.push_back(CPoint(x, y));
		}
		strToken = data.Tokenize(_T("|"), nTokenPos);
	}
}