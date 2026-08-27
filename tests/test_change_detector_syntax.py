import ast
from pathlib import Path


def test_change_detector_parses():
    path = Path(__file__).resolve().parents[1] / 'core' / 'change_detector.py'
    ast.parse(path.read_text(encoding='utf-8'))


def test_config_parses():
    path = Path(__file__).resolve().parents[1] / 'config.py'
    ast.parse(path.read_text(encoding='utf-8'))
