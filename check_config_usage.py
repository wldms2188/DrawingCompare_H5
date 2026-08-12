import ast
from pathlib import Path
 
 
files = [
    Path("core/image_loader.py"),
    Path("core/change_detector.py"),
    Path("report_generator.py"),
]
 
 
results = set()
 
 
for file in files:
    if not file.exists():
        continue
 
    text = file.read_text(encoding="utf-8")
 
    try:
        tree = ast.parse(text)
    except SyntaxError as e:
        print(f"SYNTAX ERROR: {file}")
        print(e)
        continue
 
    for node in ast.walk(tree):
        if isinstance(node, ast.Attribute):
            parts = []
 
            current = node
 
            while isinstance(current, ast.Attribute):
                parts.append(current.attr)
                current = current.value
 
            if isinstance(current, ast.Name) and current.id in {
                "config",
                "CONFIG",
            }:
                parts.append(current.id)
                results.add(".".join(reversed(parts)))
 
 
print()
print("===== CONFIG USAGE =====")
 
for item in sorted(results):
    print(item)
 
print()
print("===== TOTAL =====")
print(len(results))