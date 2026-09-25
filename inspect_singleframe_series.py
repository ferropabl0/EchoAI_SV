import os
from collections import defaultdict
import pydicom

SRC = "/run/user/1004/gvfs/smb-share:server=130.241.81.42,share=barnhjärtan/Bilder"
PATIENT = "disk_20260211_160545567_120_120_24"

DICOM_DIR = os.path.join(SRC, PATIENT, "DICOM")


files = []

for root, _, filenames in os.walk(DICOM_DIR):
    for name in filenames:
        # Ignore macOS metadata files such as ._file.dcm
        if name.startswith("."):
            continue

        if name.lower().endswith(".dcm"):
            files.append(os.path.join(root, name))

print(f"Patient: {PATIENT}")
print(f"Real .dcm files: {len(files)}")


series = defaultdict(list)
errors = []

for i, path in enumerate(files, 1):
    try:
        ds = pydicom.dcmread(path, stop_before_pixels=True)

        series_uid = str(getattr(ds, "SeriesInstanceUID", "<missing>"))

        series[series_uid].append({
            "path": path,
            "instance": getattr(ds, "InstanceNumber", None),
            "acquisition": getattr(ds, "AcquisitionDateTime", None),
            "trigger": getattr(ds, "TriggerTime", None),
            "frame_time": getattr(ds, "FrameTime", None),
            "rows": getattr(ds, "Rows", None),
            "columns": getattr(ds, "Columns", None),
            "sop_class": getattr(ds, "SOPClassUID", None),
            "image_type": getattr(ds, "ImageType", None),
        })

    except Exception as e:
        errors.append((path, repr(e)))


print(f"Readable: {sum(len(v) for v in series.values())}")
print(f"Errors: {len(errors)}")
print(f"Series: {len(series)}")


for n, (uid, items) in enumerate(
    sorted(series.items(), key=lambda x: len(x[1]), reverse=True),
    1
):

    print("\n" + "=" * 100)
    print(f"SERIES {n}")
    print("=" * 100)

    print("SeriesInstanceUID:", uid)
    print("Number of images:", len(items))

    # Sort by InstanceNumber where possible
    def instance_key(x):
        value = x["instance"]
        try:
            return (0, int(value))
        except Exception:
            return (1, str(value))

    items = sorted(items, key=instance_key)

    print("\nInstanceNumber:")
    print([x["instance"] for x in items[:20]])

    if len(items) > 20:
        print("...")

    print("\nAcquisitionDateTime:")
    print([x["acquisition"] for x in items[:10]])

    print("\nTriggerTime:")
    print([x["trigger"] for x in items[:20]])

    print("\nFrameTime:")
    print([x["frame_time"] for x in items[:20]])

    print("\nDimensions:")
    dimensions = sorted(
        set((x["rows"], x["columns"]) for x in items)
    )
    print(dimensions)

    print("\nSOP Class:")
    print(sorted(set(str(x["sop_class"]) for x in items)))

    print("\nImageType:")
    image_types = sorted(
        set(str(x["image_type"]) for x in items)
    )
    for x in image_types:
        print(" ", x)

    print("\nFirst 3 files:")
    for x in items[:3]:
        print(" ", x["path"])


if errors:
    print("\n" + "=" * 100)
    print("ERRORS")
    print("=" * 100)

    for path, error in errors[:20]:
        print(path)
        print(error)

