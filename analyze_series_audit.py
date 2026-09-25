import csv
import os
from collections import defaultdict

INPUT = os.path.expanduser(
    "~/echofocus/ventricle_project/metadata/all_series_audit.csv"
)

OUTPUT = os.path.expanduser(
    "~/echofocus/ventricle_project/metadata/series_classification.csv"
)


def to_int(x):
    try:
        return int(x)
    except:
        return None


def to_float(x):
    try:
        return float(x)
    except:
        return None


# ------------------------------------------------------------
# Load DICOM rows
# ------------------------------------------------------------

series = defaultdict(list)

with open(INPUT, newline="", encoding="utf-8") as f:

    reader = csv.DictReader(f)

    for row in reader:

        if row["category"] != "series_file":
            continue

        key = (
            row["patient"],
            row["series_uid"]
        )

        series[key].append(row)


# ------------------------------------------------------------
# Classify each series
# ------------------------------------------------------------

output = []

for (patient, series_uid), rows in series.items():

    modality = sorted(
        set(r["modality"] for r in rows)
    )

    modality_str = ";".join(modality)

    n_images = len(rows)

    number_of_frames = [
        to_int(r["number_of_frames"])
        for r in rows
    ]

    number_of_frames = [
        x for x in number_of_frames
        if x is not None
    ]

    # True multiframe
    is_multiframe = any(
        x > 1 for x in number_of_frames
    )

    # Instance numbers
    instances = [
        to_int(r["instance_number"])
        for r in rows
    ]

    instances = [
        x for x in instances
        if x is not None
    ]

    instances_sorted = sorted(set(instances))

    instance_unique = (
        len(instances_sorted) == len(instances)
    )

    contiguous = False
    starts_at_one = False

    if len(instances_sorted) >= 2:

        contiguous = (
            instances_sorted ==
            list(
                range(
                    instances_sorted[0],
                    instances_sorted[-1] + 1
                )
            )
        )

        starts_at_one = (
            instances_sorted[0] == 1
        )

    # FrameTime
    frame_times = [
        to_float(r["frame_time_ms"])
        for r in rows
    ]

    frame_times = [
        x for x in frame_times
        if x is not None and x > 0
    ]

    if frame_times:
        frame_time_ms = sum(frame_times) / len(frame_times)
        fps = 1000.0 / frame_time_ms
    else:
        frame_time_ms = None
        fps = None

    # ImageType
    image_types = " ".join(
        r["image_type"]
        for r in rows
    )

    has_gems_multiframe = (
        "GEMSMULTIFRAME" in image_types
    )

    has_gems_singleframe = (
        "GEMSSINGLEFRAME" in image_types
    )

    # Dimensions
    dimensions = sorted(
        set(
            f'{r["rows"]}x{r["columns"]}'
            for r in rows
            if r["rows"] and r["columns"]
        )
    )

    dimensions_str = ";".join(dimensions)

    # --------------------------------------------------------
    # Classification
    # --------------------------------------------------------

    if modality_str != "US":

        classification = "NON_US"

    elif is_multiframe:

        classification = "MULTIFRAME_US"

    elif (
        n_images >= 16
        and instance_unique
        and contiguous
        and starts_at_one
        and frame_times
    ):

        classification = "SINGLEFRAME_CINE_STRONG"

    elif (
        n_images >= 16
        and instance_unique
        and contiguous
        and starts_at_one
    ):

        classification = "SINGLEFRAME_CINE_POSSIBLE"

    elif n_images >= 16:

        classification = "SINGLEFRAME_LONG_UNCERTAIN"

    else:

        classification = "SINGLEFRAME_STATIC_OR_SHORT"

    # Number of usable 16-frame clips
    if classification.startswith("SINGLEFRAME_CINE"):

        n_clips = n_images // 16

    else:

        n_clips = 0

    output.append({
        "patient": patient,
        "series_uid": series_uid,
        "classification": classification,
        "n_images": n_images,
        "instance_min": min(instances_sorted)
            if instances_sorted else "",
        "instance_max": max(instances_sorted)
            if instances_sorted else "",
        "instance_unique": instance_unique,
        "instance_contiguous": contiguous,
        "instance_starts_at_1": starts_at_one,
        "frame_time_ms": frame_time_ms
            if frame_time_ms is not None else "",
        "fps": fps if fps is not None else "",
        "has_GEMSMULTIFRAME": has_gems_multiframe,
        "has_GEMSSINGLEFRAME": has_gems_singleframe,
        "dimensions": dimensions_str,
        "n_clips_16": n_clips,
    })


# ------------------------------------------------------------
# Save
# ------------------------------------------------------------

fieldnames = list(output[0].keys())

with open(
    OUTPUT,
    "w",
    newline="",
    encoding="utf-8"
) as f:

    writer = csv.DictWriter(
        f,
        fieldnames=fieldnames
    )

    writer.writeheader()
    writer.writerows(output)


# ------------------------------------------------------------
# Summary
# ------------------------------------------------------------

from collections import Counter

counts = Counter(
    r["classification"]
    for r in output
)

print("=" * 80)
print("SERIES CLASSIFICATION")
print("=" * 80)

for category, count in sorted(counts.items()):
    print(f"{category:35s} {count:6d}")

print()
print("Total series:", len(output))

print()
print("Patients per classification:")

patient_categories = defaultdict(set)

for r in output:
    patient_categories[
        r["patient"]
    ].add(r["classification"])

patient_counts = Counter()

for patient, cats in patient_categories.items():

    if "MULTIFRAME_US" in cats and any(
        c.startswith("SINGLEFRAME_CINE")
        for c in cats
    ):
        category = "MIXED"

    elif "MULTIFRAME_US" in cats:
        category = "MULTIFRAME_ONLY"

    elif "SINGLEFRAME_CINE_STRONG" in cats:
        category = "SINGLEFRAME_CINE_STRONG"

    elif "SINGLEFRAME_CINE_POSSIBLE" in cats:
        category = "SINGLEFRAME_CINE_POSSIBLE"

    elif "SINGLEFRAME_LONG_UNCERTAIN" in cats:
        category = "LONG_UNCERTAIN"

    elif "SINGLEFRAME_STATIC_OR_SHORT" in cats:
        category = "STATIC_OR_SHORT"

    else:
        category = "OTHER"

    patient_counts[category] += 1

for category, count in sorted(patient_counts.items()):
    print(f"{category:35s} {count:6d}")


print()
print("Strong cine candidates:")
print()

strong = [
    r for r in output
    if r["classification"] == "SINGLEFRAME_CINE_STRONG"
]

strong.sort(
    key=lambda x: (x["patient"], -x["n_images"])
)

for r in strong[:50]:

    print(
        f'{r["patient"]} | '
        f'n={r["n_images"]} | '
        f'FPS={r["fps"]:.2f} | '
        f'FrameTime={r["frame_time_ms"]:.3f} ms | '
        f'clips={r["n_clips_16"]}'
    )

print()
print("Output:")
print(OUTPUT)
