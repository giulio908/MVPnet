import os
import gdown

FILES = {
    "mvpnet_5_10_02.pth": "16gMC1NxJO9RxS3qNcFGtKDTq_g1f2uWt",
}

OUTPUT_DIR = "./checkpoints"

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
        quiet=False,
        fuzzy=True
    )

print("\nDownload completato.")