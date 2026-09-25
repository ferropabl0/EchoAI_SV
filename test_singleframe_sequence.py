import os
import numpy as np
import pydicom

SRC = "/run/user/1004/gvfs/smb-share:server=130.241.81.42,share=barnhjärtan/Bilder"
PATIENT = "disk_20260211_160545567_120_120_24"

DICOM_DIR = os.path.join(SRC, PATIENT, "DICOM")

TARGET_SERIES = "1.3.6.1.4.1.9590.100.1.4.319383919407526537121515644511504861475"


files = []

for root, _, filenames in os.walk(DICOM_DIR):
    for name in filenames:
        if name.startswith("."):
            continue
        if not name.lower().endswith(".dcm"):
            continue

        path = os.path.join(root, name)

        try:
            ds = pydicom.dcmread(path, stop_before_pixels=True)

            if str(getattr(ds, "SeriesInstanceUID", "")) == TARGET_SERIES:
                files.append({
                    "path": path,
                    "instance": int(ds.InstanceNumber),
                })

        except Exception:
            pass


files.sort(key=lambda x: x["instance"])

print("Number of frames:", len(files))
print("First instances:", [x["instance"] for x in files[:10]])
print("Last instances:", [x["instance"] for x in files[-10:]])


# Read three frames
indices = [0, len(files) // 2, len(files) - 1]

images = []

for idx in indices:
    path = files[idx]["path"]

    ds = pydicom.dcmread(path)
    arr = ds.pixel_array

    print(
        f"\nFrame {files[idx]['instance']}:"
        f"\n  shape = {arr.shape}"
        f"\n  dtype = {arr.dtype}"
        f"\n  min = {arr.min()}"
        f"\n  max = {arr.max()}"
        f"\n  mean = {arr.mean():.3f}"
        f"\n  std = {arr.std():.3f}"
    )

    images.append(arr.astype(np.float32))


# Compare consecutive-ish frames
print("\nFrame differences:")

for i in range(len(images) - 1):
    diff = np.mean(np.abs(images[i] - images[i + 1]))
    print(f"{indices[i]} -> {indices[i+1]}: mean abs diff = {diff:.3f}")
