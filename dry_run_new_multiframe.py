import csv
import os
import pydicom
from collections import defaultdict

SRC = "/run/user/1004/gvfs/smb-share:server=130.241.81.42,share=barnhjärtan/Bilder"

PATIENT_LIST = os.path.expanduser(
    "~/echofocus/ventricle_project/metadata/patients_new_multiframe.txt"
)

CLIP_LEN = 16
NUM_CLIPS = 16

patients = []

with open(PATIENT_LIST, encoding="utf-8") as f:
    patients = [x.strip() for x in f if x.strip()]

print("=" * 90)
print("DRY RUN — NEW MULTIFRAME PATIENTS")
print("=" * 90)
print("Patients:", len(patients))
print()

total_dicom = 0
total_frames = 0
total_embeddings = 0
total_series = 0

patient_results = []

for idx, patient in enumerate(patients, 1):

    dicom_dir = os.path.join(SRC, patient, "DICOM")

    # Recursive search, ignoring macOS metadata
    files = []

    for root, _, filenames in os.walk(dicom_dir):
        for name in filenames:

            if name.startswith("."):
                continue

            if not name.lower().endswith(".dcm"):
                continue

            files.append(os.path.join(root, name))

    patient_dicom = 0
    patient_frames = 0
    patient_series = 0

    series_seen = set()

    for path in files:

        try:
            ds = pydicom.dcmread(
                path,
                stop_before_pixels=True
            )

            if str(getattr(ds, "Modality", "")) != "US":
                continue

            nframes = getattr(ds, "NumberOfFrames", None)

            try:
                nframes = int(nframes)
            except:
                nframes = 1

            if nframes <= 1:
                continue

            uid = str(
                getattr(ds, "SeriesInstanceUID", "")
            )

            series_seen.add(uid)

            patient_dicom += 1
            patient_frames += nframes

        except Exception:
            continue

    patient_series = len(series_seen)

    # The current EchoFocus/PanEcho pipeline samples
    # 16 clips per multiframe DICOM.
    patient_embeddings = patient_dicom * NUM_CLIPS

    total_dicom += patient_dicom
    total_frames += patient_frames
    total_series += patient_series
    total_embeddings += patient_embeddings

    patient_results.append({
        "patient": patient,
        "dicom": patient_dicom,
        "frames": patient_frames,
        "series": patient_series,
        "embeddings": patient_embeddings,
    })

    print(
        f"[{idx:3d}/{len(patients)}] "
        f"{patient[:48]:48s} "
        f"DICOM={patient_dicom:4d} "
        f"frames={patient_frames:6d} "
        f"series={patient_series:3d} "
        f"emb={patient_embeddings:5d}"
    )


print()
print("=" * 90)
print("TOTALS")
print("=" * 90)

print(f"Patients:             {len(patients)}")
print(f"Multiframe DICOMs:    {total_dicom:,}")
print(f"Total frames:        {total_frames:,}")
print(f"Total series:         {total_series:,}")
print(f"PanEcho embeddings:   {total_embeddings:,}")

print()

# ------------------------------------------------------------
# Estimate based on the previous benchmark
# ------------------------------------------------------------
#
# Previous direct DICOM → PanEcho benchmark:
#
# 27 DICOM
# 1384 frames
# 86 clips
# ~37.94 s total
#
# Mass pipeline uses 16 random clips per DICOM.
# Therefore estimate using approximately 47.62 s
# for one patient with 27 DICOM / 432 embeddings.
#
# This is intentionally only an estimate because SMB
# access and DICOM sizes vary.
#

SECONDS_PER_DICOM = 47.62 / 27.0

estimated_seconds = total_dicom * SECONDS_PER_DICOM

print("ROUGH TIME ESTIMATE")
print("=" * 90)

print(
    f"Assumed time per DICOM: "
    f"{SECONDS_PER_DICOM:.2f} s"
)

print(
    f"Estimated total: "
    f"{estimated_seconds / 3600:.2f} hours"
)

print()

# ------------------------------------------------------------
# Save summary
# ------------------------------------------------------------

OUT = os.path.expanduser(
    "~/echofocus/ventricle_project/metadata/"
    "new_multiframe_dry_run.csv"
)

with open(
    OUT,
    "w",
    newline="",
    encoding="utf-8"
) as f:

    writer = csv.DictWriter(
        f,
        fieldnames=[
            "patient",
            "dicom",
            "frames",
            "series",
            "embeddings",
        ],
    )

    writer.writeheader()
    writer.writerows(patient_results)

print("Saved:", OUT)
