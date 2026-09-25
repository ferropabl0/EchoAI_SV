import os
import csv
from collections import Counter
import pydicom

SRC = (
    "/run/user/1004/gvfs/smb-share:"
    "server=130.241.81.42,"
    "share=barnhjärtan/Bilder"
)

OUT = os.path.expanduser(
    "~/echofocus/ventricle_project/metadata/"
    "non_multiframe_audit_v2.csv"
)

patients = sorted([
    d for d in os.listdir(SRC)
    if not d.startswith(".")
    and os.path.isdir(os.path.join(SRC, d))
])


def find_all_files(root):
    """
    Recursively find every file.
    """
    result = []

    for dirpath, _, filenames in os.walk(root):
        for filename in filenames:
            result.append(
                os.path.join(dirpath, filename)
            )

    return result


def try_metadata(path):
    """
    Read DICOM metadata without pixels.
    """

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

        return {
            "ok": True,
            "modality": modality,
            "nframes": nframes,
            "rows": int(getattr(ds, "Rows", 0)),
            "cols": int(getattr(ds, "Columns", 0)),
            "photometric": str(
                getattr(
                    ds,
                    "PhotometricInterpretation",
                    ""
                )
            ),
            "series_uid": str(
                getattr(
                    ds,
                    "SeriesInstanceUID",
                    ""
                )
            ),
            "series_description": str(
                getattr(
                    ds,
                    "SeriesDescription",
                    ""
                )
            ),
            "instance_number": str(
                getattr(
                    ds,
                    "InstanceNumber",
                    ""
                )
            ),
            "error": "",
        }

    except Exception as e:

        return {
            "ok": False,
            "modality": "",
            "nframes": -1,
            "rows": 0,
            "cols": 0,
            "photometric": "",
            "series_uid": "",
            "series_description": "",
            "instance_number": "",
            "error": repr(e),
        }


rows = []

print("=" * 70)
print("NON-MULTIFRAME DICOM DIAGNOSTIC AUDIT")
print("=" * 70)

print(f"Total patient folders: {len(patients)}")
print()

for patient in patients:

    dicom_dir = os.path.join(
        SRC,
        patient,
        "DICOM"
    )

    if not os.path.isdir(dicom_dir):
        continue

    all_files = find_all_files(dicom_dir)

    # All files whose filename ends in .dcm
    dcm_files = [
        p for p in all_files
        if os.path.basename(p).lower().endswith(".dcm")
    ]

    # Also inspect files without .dcm, because some DICOM
    # datasets may not have the extension.
    other_files = [
        p for p in all_files
        if not os.path.basename(p).lower().endswith(".dcm")
    ]

    modality_counts = Counter()

    us_single = 0
    us_multi = 0
    successful = 0
    errors = 0

    error_examples = []

    series_uids = set()

    for path in dcm_files:

        result = try_metadata(path)

        if result["ok"]:

            successful += 1

            modality_counts[
                result["modality"]
            ] += 1

            if result["modality"] == "US":

                if result["nframes"] > 1:
                    us_multi += 1
                else:
                    us_single += 1

            if result["series_uid"]:
                series_uids.add(
                    result["series_uid"]
                )

        else:

            errors += 1

            if len(error_examples) < 3:
                error_examples.append(
                    result["error"]
                )

    # --------------------------------------------------------
    # Only print patients that had no multiframe US.
    # --------------------------------------------------------

    if us_multi == 0:

        print(
            f"{patient}"
        )

        print(
            f"  all files:       {len(all_files)}"
        )

        print(
            f"  .dcm files:      {len(dcm_files)}"
        )

        print(
            f"  other files:     {len(other_files)}"
        )

        print(
            f"  readable:        {successful}"
        )

        print(
            f"  read errors:     {errors}"
        )

        print(
            f"  US single:       {us_single}"
        )

        print(
            f"  US multiframe:   {us_multi}"
        )

        print(
            f"  unique series:   {len(series_uids)}"
        )

        print(
            f"  modalities:      "
            f"{dict(modality_counts)}"
        )

        if error_examples:

            print("  error examples:")

            for error in error_examples:
                print(
                    f"    {error}"
                )

        print()

        rows.append({
            "patient": patient,
            "all_files": len(all_files),
            "dcm_files": len(dcm_files),
            "other_files": len(other_files),
            "readable_dicom": successful,
            "read_errors": errors,
            "us_singleframe": us_single,
            "us_multiframe": us_multi,
            "unique_series": len(series_uids),
            "modalities": ";".join(
                f"{k}:{v}"
                for k, v in sorted(
                    modality_counts.items()
                )
            ),
            "error_examples": " || ".join(
                error_examples
            ),
        })


# ============================================================
# SAVE
# ============================================================

os.makedirs(
    os.path.dirname(OUT),
    exist_ok=True
)

fieldnames = [
    "patient",
    "all_files",
    "dcm_files",
    "other_files",
    "readable_dicom",
    "read_errors",
    "us_singleframe",
    "us_multiframe",
    "unique_series",
    "modalities",
    "error_examples",
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

print("=" * 70)
print("SUMMARY")
print("=" * 70)

print(
    f"Patients without multiframe US: "
    f"{len(rows)}"
)

print(
    f"Patients with single-frame US: "
    f"{sum(r['us_singleframe'] > 0 for r in rows)}"
)

print(
    f"Patients with zero readable DICOM: "
    f"{sum(r['readable_dicom'] == 0 for r in rows)}"
)

print(
    f"Patients with read errors: "
    f"{sum(r['read_errors'] > 0 for r in rows)}"
)

print(
    f"Patients with files in subdirectories: "
    f"{sum(r['all_files'] > r['dcm_files'] for r in rows)}"
)

print()
print(f"Output: {OUT}")
