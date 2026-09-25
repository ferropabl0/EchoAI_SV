import os
import csv
import pydicom
from collections import Counter

SRC = (
    "/run/user/1004/gvfs/smb-share:"
    "server=130.241.81.42,"
    "share=barnhjärtan/Bilder"
)

OUT = os.path.expanduser(
    "~/echofocus/ventricle_project/metadata/"
    "non_multiframe_audit.csv"
)

# Patients that did NOT produce a multiframe-US HDF5
patients = sorted([
    d for d in os.listdir(SRC)
    if not d.startswith(".")
    and os.path.isdir(os.path.join(SRC, d))
])

rows = []

print("=" * 70)
print("AUDIT OF PATIENTS WITHOUT MULTIFRAME US")
print("=" * 70)

for i, patient in enumerate(patients, 1):

    dicom_dir = os.path.join(
        SRC, patient, "DICOM"
    )

    if not os.path.isdir(dicom_dir):
        continue

    # First determine whether this patient has any multiframe US
    has_multiframe_us = False

    files = [
        f for f in os.listdir(dicom_dir)
        if f.lower().endswith(".dcm")
    ]

    metadata = []

    for filename in files:

        path = os.path.join(
            dicom_dir, filename
        )

        try:
            ds = pydicom.dcmread(
                path,
                stop_before_pixels=True
            )

            modality = str(
                getattr(ds, "Modality", "")
            )

            nframes = int(
                getattr(ds, "NumberOfFrames", 1)
            )

            rows_count = int(
                getattr(ds, "Rows", 0)
            )

            cols_count = int(
                getattr(ds, "Columns", 0)
            )

            photometric = str(
                getattr(
                    ds,
                    "PhotometricInterpretation",
                    ""
                )
            )

            sop_class = str(
                getattr(
                    ds,
                    "SOPClassUID",
                    ""
                )
            )

            metadata.append({
                "filename": filename,
                "modality": modality,
                "nframes": nframes,
                "rows": rows_count,
                "cols": cols_count,
                "photometric": photometric,
                "sop_class": sop_class,
            })

            if (
                modality == "US"
                and nframes > 1
            ):
                has_multiframe_us = True

        except Exception as e:

            metadata.append({
                "filename": filename,
                "modality": "READ_ERROR",
                "nframes": -1,
                "rows": 0,
                "cols": 0,
                "photometric": "",
                "sop_class": str(e),
            })

    # We only want patients without multiframe US
    if has_multiframe_us:
        continue

    modality_counts = Counter(
        x["modality"]
        for x in metadata
    )

    multiframe_count = sum(
        x["nframes"] > 1
        for x in metadata
    )

    singleframe_count = sum(
        x["nframes"] == 1
        for x in metadata
    )

    us_count = sum(
        x["modality"] == "US"
        for x in metadata
    )

    us_singleframe_count = sum(
        x["modality"] == "US"
        and x["nframes"] == 1
        for x in metadata
    )

    errors = sum(
        x["modality"] == "READ_ERROR"
        for x in metadata
    )

    # Unique dimensions among US images
    us_dimensions = sorted(set(
        f'{x["rows"]}x{x["cols"]}'
        for x in metadata
        if x["modality"] == "US"
        and x["rows"] > 0
        and x["cols"] > 0
    ))

    row = {
        "patient": patient,
        "dicom_files": len(metadata),
        "us_files": us_count,
        "us_singleframe": us_singleframe_count,
        "all_singleframe": (
            singleframe_count == len(metadata)
        ),
        "multiframe_files": multiframe_count,
        "read_errors": errors,
        "modalities": ";".join(
            f"{k}:{v}"
            for k, v in sorted(
                modality_counts.items()
            )
        ),
        "us_dimensions": ";".join(
            us_dimensions
        ),
    }

    rows.append(row)

    print(
        f"{len(rows):3d} | "
        f"{patient} | "
        f"DICOM={len(metadata):3d} | "
        f"US={us_count:3d} | "
        f"US single={us_singleframe_count:3d} | "
        f"multiframe={multiframe_count:3d} | "
        f"errors={errors}"
    )


# ============================================================
# SAVE
# ============================================================

os.makedirs(
    os.path.dirname(OUT),
    exist_ok=True
)

fieldnames = [
    "patient",
    "dicom_files",
    "us_files",
    "us_singleframe",
    "all_singleframe",
    "multiframe_files",
    "read_errors",
    "modalities",
    "us_dimensions",
]

with open(
    OUT,
    "w",
    newline=""
) as f:

    writer = csv.DictWriter(
        f,
        fieldnames=fieldnames
    )

    writer.writeheader()
    writer.writerows(rows)


# ============================================================
# SUMMARY
# ============================================================

print()
print("=" * 70)
print("SUMMARY")
print("=" * 70)

print(
    f"Patients without multiframe US: "
    f"{len(rows)}"
)

print(
    f"Patients with at least one US: "
    f"{sum(r['us_files'] > 0 for r in rows)}"
)

print(
    f"Patients with US single-frame only: "
    f"{sum(r['us_singleframe'] > 0 for r in rows)}"
)

print(
    f"Patients with zero US files: "
    f"{sum(r['us_files'] == 0 for r in rows)}"
)

print(
    f"Patients with read errors: "
    f"{sum(r['read_errors'] > 0 for r in rows)}"
)

print()
print(f"Output: {OUT}")
