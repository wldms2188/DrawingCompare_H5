from pathlib import Path
 
from version import APP_NAME, VERSION
import config
 
 
def create_folders():
 
    folders = [
        config.INPUT_DIR,
        config.OUTPUT_DIR,
        config.CAPTURE_DIR,
        config.TEMP_DIR,
        config.LOG_DIR,
        config.TEMPLATE_DIR
    ]
 
    for folder in folders:
        Path(folder).mkdir(parents=True, exist_ok=True)
 
 
def print_banner():
 
    print("=" * 60)
    print(f"{APP_NAME} {VERSION}")
    print("=" * 60)
    print("Engineering Drawing Compare")
    print("=" * 60)
 
 
def main():
 
    create_folders()
 
    print_banner()
 
    print()
 
    print("Project initialized successfully.")
 
    print()
 
    print("Next Step : PDF Parser")
 
 
if __name__ == "__main__":
    main()