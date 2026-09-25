import csv
import os
import h5py

AUDIT = os.path.expanduser(
    "~/echofocus/ventricle_project/metadata/series_classification.csv"
)

EMBED_DIR = os.path.expanduser(
    "~/echofocus/ventricle_project/embeddings/dicom"
)

# ------------------------------------------------------------
# Patients with multiframe US according to the new audit
# ------------------------------------------------------------

multiframe_patients = set()

with open(AUDIT, newline="", encoding="utf-8") as f:

    reader = csv.DictReader(f)

    for row in reader:

        if row["classification"] == "MULTIFRAME_US":
            multiframe_patients.add(row["patient"])


# ------------------------------------------------------------
# Patients already processed
# ------------------------------------------------------------

processed_patients = set()

if os.path.isdir(EMBED_DIR):

    for filename in os.listdir(EMBED_DIR):

        if not filename.lower().endswith(".hdf5"):
            continue

        path = os.path.join(EMBED_DIR, filename)

        try:
            with h5py.File(path, "r") as h5:

                # Our pipeline stores the patient name
                # as the top-level group.
                groups = list(h5.keys())

                if groups:
                    processed_patients.add(groups[0])

        except Exception as e:

            print(
                "Could not read:",
                filename,
                "|",
                repr(e)
            )


# ------------------------------------------------------------
# Compare
# ------------------------------------------------------------

new_multiframe = sorted(
    multiframe_patients - processed_patients
)

already_processed = sorted(
    multiframe_patients & processed_patients
)

processed_but_not_currently_multiframe = sorted(
    processed_patients - multiframe_patients
)


print("=" * 80)
print("PATIENT COMPARISON")
print("=" * 80)

print()
print("Multiframe patients in new audit:",
      len(multiframe_patients))

print("Already processed:",
      len(already_processed))

print("NOT processed yet:",
      len(new_multiframe))

print("Processed but not classified as multiframe now:",
      len(processed_but_not_currently_multiframe))


print()
print("=" * 80)
print("NEW MULTIFRAME PATIENTS")
print("=" * 80)

for p in new_multiframe:
    print(p)


print()
print("=" * 80)
print("PROCESSED BUT NOT CURRENTLY MULTIFRAME")
print("=" * 80)

for p in processed_but_not_currently_multiframe:
    print(p)


# ------------------------------------------------------------
# Save new multiframe list
# ------------------------------------------------------------

OUT = os.path.expanduser(
    "~/echofocus/ventricle_project/metadata/"
    "patients_new_multiframe.txt"
)

with open(OUT, "w", encoding="utf-8") as f:

    for p in new_multiframe:
        f.write(p + "\n")

print()
print("Saved:", OUT)
