import os
import csv
import pydicom
from collections import Counter

SRC = "/run/user/1004/gvfs/smb-share:server=130.241.81.42,share=barnhjärtan/Bilder"
OUT = os.path.expanduser(
    "~/echofocus/ventricle_project/metadata/pixel_audit.csv"
)

os.makedirs(os.path.dirname(OUT), exist_ok=True)

patient_folders = [
    x for x in os.listdir(SRC)
    if os.path.isdir(os.path.join(SRC, x)) and not x.startswith(".")
]

print(f"Patient folders: {len(patient_folders)}")
print("Testing pixel decoding of multiframe DICOMs...")
print()

results = []

for patient_index, patient in enumerate(patient_folders):

    dicom_dir = os.path.join(SRC, patient, "DICOM")

    if not os.path.isdir(dicom_dir):
        continue

    for filename in os.listdir(dicom_dir):

        if filename.startswith("."):
            continue

        if not filename.lower().endswith(".dcm"):
            continue

        path = os.path.join(dicom_dir, filename)

        try:
            # First read metadata
            ds = pydicom.dcmread(path, stop_before_pixels=True)

            n_frames = int(getattr(ds, "NumberOfFrames", 1))

            # We only test multiframe files
            if n_frames <= 1:
                continue

            result = {
                "patient_index": patient_index,
                "file": filename,
                "n_frames_metadata": n_frames,
                "decode_ok": False,
                "n_frames_decoded": "",
                "pixel_shape": "",
                "pixel_dtype": "",
                "error": "",
            }

            try:
                # Actually decode PixelData
                ds = pydicom.dcmread(path)
                pixels = ds.pixel_array

                result["decode_ok"] = True
                result["n_frames_decoded"] = (
                    pixels.shape[0]
                    if pixels.ndim >= 3
                    else 1
                )
                result["pixel_shape"] = str(pixels.shape)
                result["pixel_dtype"] = str(pixels.dtype)

            except Exception as e:
                result["error"] = (
                    type(e).__name__ + ": " + str(e)[:300]
                )

            results.append(result)

        except Exception as e:
            results.append({
                "patient_index": patient_index,
                "file": filename,
                "n_frames_metadata": "",
                "decode_ok": False,
                "n_frames_decoded": "",
                "pixel_shape": "",
                "pixel_dtype": "",
                "error": (
                    "metadata: "
                    + type(e).__name__
                    + ": "
                    + str(e)[:300]
                ),
            })

    if (patient_index + 1) % 10 == 0:
        print(
            f"Processed patients: "
            f"{patient_index + 1}/{len(patient_folders)}"
        )


fieldnames = [
    "patient_index",
    "file",
    "n_frames_metadata",
    "decode_ok",
    "n_frames_decoded",
    "pixel_shape",
    "pixel_dtype",
    "error",
]

with open(OUT, "w", newline="", encoding="utf-8") as f:
    writer = csv.DictWriter(f, fieldnames=fieldnames)
    writer.writeheader()
    writer.writerows(results)


total = len(results)
ok = sum(r["decode_ok"] for r in results)
failed = total - ok

print()
print("========================================")
print("PIXEL AUDIT COMPLETE")
print("========================================")
print(f"Multiframe DICOMs tested: {total}")
print(f"Pixel decoding successful: {ok}")
print(f"Pixel decoding failed: {failed}")

if total:
    print(f"Success rate: {100 * ok / total:.2f}%")

patients_with_ok = len({
    r["patient_index"]
    for r in results
    if r["decode_ok"]
})

patients_with_failed = len({
    r["patient_index"]
    for r in results
    if not r["decode_ok"]
})

print(f"Patients with >=1 decodable cine: {patients_with_ok}")
print(f"Patients with >=1 decoding failure: {patients_with_failed}")

if failed:
    print()
    print("Error types:")
    error_types = Counter(
        r["error"].split(":")[0]
        for r in results
        if not r["decode_ok"]
    )

    for error, count in error_types.most_common():
        print(f"  {error}: {count}")

print()
print(f"Saved to: {OUT}")
