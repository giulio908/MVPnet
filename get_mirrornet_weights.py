import os
import gdown

FILES = {
    "160.pth": "1DPg3ElaUCnrQdHxG1apkVU95ILj7FbZ0",
}
OUTPUT_DIR = "./ICCV2019_MirrorNet/ckpt/MirrorNet/"
os.makedirs(OUTPUT_DIR, exist_ok=True)

for filename, file_id in FILES.items():
    output_path = os.path.join(OUTPUT_DIR, filename)
    if os.path.exists(output_path):
        print(f"[SKIP] File già esistente: {filename}")
        continue
    url = f"https://drive.google.com/uc?id={file_id}"
    print(f"[DOWNLOAD] {filename}")
    gdown.download(
        url=url,
        output=output_path,
        quiet=False
    )
print("\nDownload completato.")