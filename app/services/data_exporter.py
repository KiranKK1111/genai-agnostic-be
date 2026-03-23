"""Data export — CSV, XLSX, JSON, PDF from query results."""
import os
import csv
import json
import uuid
import logging
from datetime import datetime
from app.config import get_settings

logger = logging.getLogger(__name__)


async def export_data(rows: list[dict], columns: list[str], format: str = "csv",
                      file_name: str = None) -> str:
    """Export data to a file. Returns the file path."""
    settings = get_settings()
    export_dir = os.path.join(settings.UPLOAD_DIR, "exports")
    os.makedirs(export_dir, exist_ok=True)

    if not file_name:
        ts = datetime.now().strftime("%Y%m%d_%H%M%S")
        file_name = f"export_{ts}_{uuid.uuid4().hex[:6]}"

    if format == "csv":
        path = os.path.join(export_dir, f"{file_name}.csv")
        with open(path, "w", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=columns)
            writer.writeheader()
            for row in rows:
                writer.writerow({k: row.get(k, "") for k in columns})
        return path

    elif format == "json":
        path = os.path.join(export_dir, f"{file_name}.json")
        with open(path, "w") as f:
            json.dump({"columns": columns, "rows": rows, "count": len(rows)}, f, indent=2, default=str)
        return path

    elif format == "xlsx":
        try:
            import openpyxl
            path = os.path.join(export_dir, f"{file_name}.xlsx")
            wb = openpyxl.Workbook()
            ws = wb.active
            ws.title = "Export"
            ws.append(columns)
            for row in rows:
                ws.append([row.get(c, "") for c in columns])
            wb.save(path)
            return path
        except ImportError:
            logger.error("openpyxl not installed for XLSX export")
            raise ValueError("XLSX export requires openpyxl")

    elif format == "pdf":
        path = os.path.join(export_dir, f"{file_name}.pdf")
        try:
            from reportlab.lib import colors
            from reportlab.lib.pagesizes import A4, landscape
            from reportlab.platypus import SimpleDocTemplate, Table, TableStyle
            doc = SimpleDocTemplate(path, pagesize=landscape(A4))
            # Build table data: header + rows
            table_data = [columns]
            for row in rows[:500]:  # Cap PDF to 500 rows for performance
                table_data.append([str(row.get(c, ""))[:50] for c in columns])
            table = Table(table_data)
            table.setStyle(TableStyle([
                ("BACKGROUND", (0, 0), (-1, 0), colors.HexColor("#2563EB")),
                ("TEXTCOLOR", (0, 0), (-1, 0), colors.white),
                ("FONTSIZE", (0, 0), (-1, 0), 9),
                ("FONTSIZE", (0, 1), (-1, -1), 8),
                ("GRID", (0, 0), (-1, -1), 0.5, colors.grey),
                ("ROWBACKGROUNDS", (0, 1), (-1, -1), [colors.white, colors.HexColor("#F3F4F6")]),
            ]))
            doc.build([table])
            return path
        except ImportError:
            # Fallback: generate a simple text-based PDF-like file
            logger.warning("reportlab not installed. Generating plain-text PDF fallback.")
            with open(path, "w", encoding="utf-8") as f:
                f.write("\t".join(columns) + "\n")
                f.write("-" * 80 + "\n")
                for row in rows:
                    f.write("\t".join(str(row.get(c, "")) for c in columns) + "\n")
            return path

    else:
        raise ValueError(f"Unsupported export format: {format}")
