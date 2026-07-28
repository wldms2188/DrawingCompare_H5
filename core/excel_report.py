from openpyxl.drawing.image import Image
from openpyxl.utils import get_column_letter

import os
 
from openpyxl import Workbook
from openpyxl.styles import Font
from openpyxl.styles import Alignment
 
 
class ExcelReport:
 
    def create(self, changes, output_path):
 
        wb = Workbook()
 
        ws = wb.active
        ws.title = "Drawing Compare"
 
        headers = [
            "No",
            "Page",
            "Type",
            "Before",
            "After",
            "Before Image",
            "After Image"
        ]
 
        for col, title in enumerate(headers, start=1):
 
            cell = ws.cell(row=1, column=col)
 
            cell.value = title
            cell.font = Font(bold=True)
            cell.alignment = Alignment(horizontal="center")
 
       for row, change in enumerate(changes, start=2):
 
    ws.cell(row=row, column=1).value = change.id
    ws.cell(row=row, column=2).value = change.page
    ws.cell(row=row, column=3).value = change.change_type
    ws.cell(row=row, column=4).value = change.before_text
    ws.cell(row=row, column=5).value = change.after_text
 
    ws.row_dimensions[row].height = 120
 
    ws.column_dimensions[get_column_letter(6)].width = 25
    ws.column_dimensions[get_column_letter(7)].width = 25
 
    if hasattr(change, "before_image_path") and change.before_image_path:
        img = Image(change.before_image_path)
        img.width = 140
        img.height = 100
        ws.add_image(img, f"F{row}")
 
    if hasattr(change, "after_image_path") and change.after_image_path:
        img = Image(change.after_image_path)
        img.width = 140
        img.height = 100
        ws.add_image(img, f"G{row}")

 
        os.makedirs(
            os.path.dirname(output_path),
            exist_ok=True
        )
 
        wb.save(output_path)
 
        print(f"Excel 저장 완료 : {output_path}")