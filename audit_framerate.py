import os
import pydicom
from collections import Counter

SRC = "/run/user/1004/gvfs/smb-share:server=130.241.81.42,share=barnhjärtan/Bilder"

patients = [
    x for x in os.listdir(SRC)
    if os.path.isdir(os.path.join(SRC, x))
    and not x.startswith(".")
]

# Valores encontrados en los tags temporales
cine_rates = Counter()
display_rates = Counter()
frame_times = Counter()

tested = 0

for patient in patients:
    dicom_dir = os.path.join(SRC, patient, "DICOM")

    if not os.path.isdir(dicom_dir):
        continue

    for filename in os.listdir(dicom_dir):

        if filename.startswith(".") or not filename.lower().endswith(".dcm"):
            continue

        path = os.path.join(dicom_dir, filename)

        try:
            ds = pydicom.dcmread(path, stop_before_pixels=True)

            if int(getattr(ds, "NumberOfFrames", 1)) <= 1:
                continue

            tested += 1

            cine = getattr(ds, "CineRate", None)
            display = getattr(ds, "RecommendedDisplayFrameRate", None)
            frame_time = getattr(ds, "FrameTime", None)

            if cine is not None:
                cine_rates[round(float(cine), 3)] += 1

            if display is not None:
                display_rates[round(float(display), 3)] += 1

            if frame_time is not None:
                frame_times[round(float(frame_time), 3)] += 1

        except Exception:
            pass


def print_counter(title, counter):
    print()
    print(title)

    if not counter:
        print("  No values found")
        return

    for value, count in counter.most_common():
        print(f"  {value}: {count}")


print("========================================")
print("FRAME RATE AUDIT")
print("========================================")
print(f"Multiframe DICOMs inspected: {tested}")

print_counter("CineRate (fps):", cine_rates)
print_counter("RecommendedDisplayFrameRate (fps):", display_rates)
print_counter("FrameTime (ms/frame):", frame_times)

print()
