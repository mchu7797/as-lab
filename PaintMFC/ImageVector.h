#pragma once

class CImageVector
{
public:
	CImageVector(int shapeKind);
	CImageVector();

	int GetShapeKind() { return m_shapeKind; }

	void AppendPoint(CPoint point);
	void Draw(CClientDC& deviceContext);
	void SetPenConfig(COLORREF color, int penWidth);
	void SetShapeConfig(COLORREF color, bool fillShape);
	CString ExportText();
	void ImportText(CString data);
private:
	int m_shapeKind;

	COLORREF m_color;
	int m_penWidth;
	bool m_fillShape;
	
	std::vector<CPoint> m_points;

	void DrawShape(CClientDC& deviceContext);
	void DrawLine(CClientDC& deviceContext);
};

