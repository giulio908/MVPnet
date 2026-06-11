import os
import gdown
import zipfile

FILES = {
    "data.zip": "1ZYxt4ejY0XU5Pr6Hf3FlmNNdihA_e9nu",
}
OUTPUT_DIR = "./data"

os.makedirs(OUTPUT_DIR, exist_ok=True)
for filename, file_id in FILES.items():
    output_path = os.path.join(OUTPUT_DIR, filename)
    if os.path.exists(output_path):
        print(f"[SKIP] File già esistente: {filename}")
    else:
        print(f"Downloading {filename}...")
        gdown.download(
            id=file_id,
            output=output_path,
            quiet=False
        )
    if filename.endswith('.zip') and os.path.exists(output_path):
        print(f"Extracting {filename}...")
        try:
            with zipfile.ZipFile(output_path, 'r') as zip_ref:
                zip_ref.extractall(OUTPUT_DIR)
            print(f"Extracted {filename}!")
        except zipfile.BadZipFile:
            print(f"[ERROR] {filename} is not a valid zip file. The download may have failed.")
print("\nDownload completato.")