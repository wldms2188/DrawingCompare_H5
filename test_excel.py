from core.change_info import ChangeInfo
from core.excel_report import ExcelReport
 
changes = []
 
c1 = ChangeInfo(
    id=1,
    x=0,
    y=0,
    w=0,
    h=0
)
 
c1.before_text = "Ø10"
c1.after_text = "Ø12"
c1.change_type = "Dimension"
 
changes.append(c1)
 
ExcelReport().create(
    changes,
    "output/report.xlsx"
)