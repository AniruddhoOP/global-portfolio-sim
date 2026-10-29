"""Excel export of the simulation results."""
from io import BytesIO

import pandas as pd


def to_excel(sheets):
    """sheets = {sheet name: DataFrame}. Returns .xlsx bytes."""
    buf = BytesIO()
    with pd.ExcelWriter(buf, engine="openpyxl") as writer:
        for name, df in sheets.items():
            df.to_excel(writer, sheet_name=name[:31])
            ws = writer.sheets[name[:31]]
            for col in ws.columns:  # fit column widths to content
                width = max(len(str(c.value)) if c.value is not None else 0 for c in col)
                ws.column_dimensions[col[0].column_letter].width = min(width + 2, 40)
    return buf.getvalue()
