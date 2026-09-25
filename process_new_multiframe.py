import os
import time
import random
import csv
import traceback

import cv2
import h5py
import numpy as np
import pydicom
import torch

from tqdm import tqdm
from torchvision import tv_tensors
from torchvision.transforms import v2


# ============================================================
# CONFIGURATION
# ============================================================

SRC = (
    "/run/user/1004/gvfs/smb-share:"
    "server=130.241.81.42,"
    "share=barnhjärtan/Bilder"
)

BASE_DIR = os.path.expanduser(
    "~/echofocus/ventricle_project"
)

OUT_DIR = os.path.join(
    BASE_DIR,
    "embeddings",
    "dicom"
)

LOG_DIR = os.path.join(
    BASE_DIR,
    "logs"
)

PROGRESS_FILE = os.path.join(
    LOG_DIR,
    "new_multiframe_progress.csv"
)

ERROR_FILE = os.path.join(
    LOG_DIR,
    "new_multiframe_errors.csv"
)

PATIENT_LIST = os.path.expanduser(
    "~/echofocus/ventricle_project/metadata/"
    "patients_new_multiframe.txt"
)

NUM_CLIPS = 16
CLIP_LEN = 16

SEED = 0

# ------------------------------------------------------------
# SAFETY SETTING
# ------------------------------------------------------------
#
# FIRST RUN:
#     MAX_PATIENTS = 1
#
# After confirming the first patient works:
#     MAX_PATIENTS = None
#
# ------------------------------------------------------------

MAX_PATIENTS = None


# ============================================================
# REPRODUCIBILITY
# ============================================================

random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)


# ============================================================
# TRANSFORMS
# Same preprocessing as EchoFocus process_batch.py
# ============================================================

Test_Transforms = torch.nn.Sequential(
    v2.CenterCrop((224, 224)),
    v2.Normalize(
        [0.485, 0.456, 0.406],
        [0.229, 0.224, 0.225],
    ),
)


# ============================================================
# DIRECTORIES
# ============================================================

os.makedirs(OUT_DIR, exist_ok=True)
os.makedirs(LOG_DIR, exist_ok=True)


# ============================================================
# LOGGING
# ============================================================

def ensure_csv(path, header):

    if not os.path.exists(path):

        with open(
            path,
            "w",
            newline="",
            encoding="utf-8"
        ) as f:

            writer = csv.writer(f)
            writer.writerow(header)


ensure_csv(
    PROGRESS_FILE,
    [
        "patient",
        "status",
        "dicom_count",
        "successful_dicoms",
        "failed_dicoms",
        "total_frames",
        "total_clips",
        "elapsed_seconds",
        "output_path",
    ],
)


ensure_csv(
    ERROR_FILE,
    [
        "patient",
        "dicom",
        "error",
    ],
)


# ============================================================
# PATIENT DISCOVERY
# ============================================================

print("=" * 70)
print("ECHOFOCUS NEW MULTIFRAME DICOM → PAN-ECHO → HDF5")
print("=" * 70)

print(f"Source: {SRC}")
print(f"Output: {OUT_DIR}")
print(f"Patient list: {PATIENT_LIST}")


if not os.path.exists(PATIENT_LIST):

    raise FileNotFoundError(
        f"Patient list not found: {PATIENT_LIST}"
    )


with open(
    PATIENT_LIST,
    "r",
    encoding="utf-8"
) as f:

    patients = [
        line.strip()
        for line in f
        if line.strip()
    ]


# Remove accidental duplicate lines
patients = sorted(set(patients))


print()
print(f"Patients in processing list: {len(patients)}")


# Safety check: we expect exactly 104
if len(patients) != 104:

    raise RuntimeError(
        f"Expected exactly 104 patients, "
        f"but found {len(patients)}."
    )


if MAX_PATIENTS is not None:

    if MAX_PATIENTS < 1:

        raise ValueError(
            "MAX_PATIENTS must be >= 1 or None."
        )

    patients = patients[:MAX_PATIENTS]

    print(
        f"TEST MODE: processing first "
        f"{MAX_PATIENTS} patient(s)"
    )

else:

    print("FULL MODE: processing all 104 patients")


print()
print("IMPORTANT:")
print("Only patients from patients_new_multiframe.txt")
print("will be processed.")
print("Existing valid HDF5 files will be skipped.")
print()


# ============================================================
# LOAD PAN-ECHO
# ============================================================

device = (
    "cuda"
    if torch.cuda.is_available()
    else "cpu"
)

print(f"Device: {device}")


if device == "cuda":

    print(
        f"GPU: "
        f"{torch.cuda.get_device_name(0)}"
    )

    print(
        f"GPU memory: "
        f"{torch.cuda.get_device_properties(0).total_memory / 1e9:.1f} GB"
    )


print()
print("Loading PanEcho...")


model = torch.hub.load(
    "CarDS-Yale/PanEcho",
    "PanEcho",
    force_reload=False,
    backbone_only=True,
)


model = model.to(device)
model.eval()


print("PanEcho loaded.")


# ============================================================
# HELPERS
# ============================================================

def get_framerate(ds):
    """
    Preserve the original temporal metadata.

    Priority:
        1. CineRate
        2. RecommendedDisplayFrameRate
        3. FrameTime -> FPS

    Returns:
        float or None
    """

    value = getattr(
        ds,
        "CineRate",
        None
    )

    if value is not None:

        try:
            return float(value)

        except Exception:
            pass


    value = getattr(
        ds,
        "RecommendedDisplayFrameRate",
        None
    )

    if value is not None:

        try:
            return float(value)

        except Exception:
            pass


    value = getattr(
        ds,
        "FrameTime",
        None
    )

    if value is not None:

        try:

            frame_time_ms = float(value)

            if frame_time_ms > 0:

                return 1000.0 / frame_time_ms

        except Exception:
            pass


    return None


def is_valid_hdf5(path):
    """
    Basic integrity check for an already processed patient.

    This function is deliberately conservative.
    A patient is skipped only if the HDF5 appears valid.
    """

    if not os.path.exists(path):
        return False


    try:

        with h5py.File(path, "r") as h5:

            # Exactly one patient group
            if len(h5.keys()) != 1:
                return False


            patient_group = next(
                iter(h5.values())
            )


            # Must contain at least one DICOM group
            if len(patient_group.keys()) == 0:
                return False


            for dicom_name, group in patient_group.items():

                if "emb" not in group:
                    return False


                emb = group["emb"]


                if emb.ndim != 2:
                    return False


                if emb.shape[1] != 768:
                    return False


                if not np.all(
                    np.isfinite(emb[()])
                ):
                    return False


            return True


    except Exception:

        return False


def log_progress(row):

    with open(
        PROGRESS_FILE,
        "a",
        newline="",
        encoding="utf-8"
    ) as f:

        writer = csv.writer(f)
        writer.writerow(row)


def log_error(
    patient,
    dicom,
    error
):

    with open(
        ERROR_FILE,
        "a",
        newline="",
        encoding="utf-8"
    ) as f:

        writer = csv.writer(f)

        writer.writerow(
            [
                patient,
                dicom,
                str(error),
            ]
        )


# ============================================================
# PROCESS ONE PATIENT
# ============================================================

def process_patient(patient):

    patient_start = time.perf_counter()


    dicom_dir = os.path.join(
        SRC,
        patient,
        "DICOM"
    )


    output_path = os.path.join(
        OUT_DIR,
        f"{patient}_embed.hdf5"
    )


    temp_path = (
        output_path
        + ".tmp"
    )


    # --------------------------------------------------------
    # Resume / safety support
    # --------------------------------------------------------

    if is_valid_hdf5(output_path):

        print()
        print(f"SKIP: {patient}")
        print("      Valid HDF5 already exists.")
        print(f"      {output_path}")

        return {
            "status": "skipped",
            "dicom_count": 0,
            "successful_dicoms": 0,
            "failed_dicoms": 0,
            "total_frames": 0,
            "total_clips": 0,
            "elapsed": 0.0,
            "output": output_path,
        }


    # Remove incomplete temporary file
    # from an earlier interrupted run.

    if os.path.exists(temp_path):

        print(
            f"Removing previous temporary file: "
            f"{temp_path}"
        )

        os.remove(temp_path)


    if not os.path.isdir(dicom_dir):

        raise RuntimeError(
            f"DICOM directory does not exist: "
            f"{dicom_dir}"
        )


    # --------------------------------------------------------
    # Find multiframe US DICOMs RECURSIVELY
    # --------------------------------------------------------

    dicom_files = []


    for root, _, filenames in os.walk(
        dicom_dir
    ):

        for filename in filenames:

            # Ignore macOS metadata
            # such as ._xxxx.dcm

            if filename.startswith("."):
                continue


            if not filename.lower().endswith(".dcm"):
                continue


            path = os.path.join(
                root,
                filename
            )


            try:

                ds = pydicom.dcmread(
                    path,
                    stop_before_pixels=True
                )


                modality = str(
                    getattr(
                        ds,
                        "Modality",
                        ""
                    )
                )


                number_of_frames = int(
                    getattr(
                        ds,
                        "NumberOfFrames",
                        1
                    )
                )


                if (
                    modality == "US"
                    and number_of_frames > 1
                ):

                    dicom_files.append(path)


            except Exception as e:

                log_error(
                    patient,
                    path,
                    f"Metadata read failed: {e}"
                )


    dicom_files = sorted(
        dicom_files
    )


    # --------------------------------------------------------
    # Patient information
    # --------------------------------------------------------

    print()
    print("=" * 70)
    print(f"PATIENT: {patient}")
    print("=" * 70)

    print(
        f"DICOM directory: {dicom_dir}"
    )

    print(
        f"Multiframe US DICOMs: "
        f"{len(dicom_files)}"
    )


    if not dicom_files:

        raise RuntimeError(
            "No multiframe US DICOMs found."
        )


    total_frames = 0
    total_clips = 0

    successful_dicoms = 0
    failed_dicoms = 0


    # --------------------------------------------------------
    # Create HDF5
    # --------------------------------------------------------

    with h5py.File(
        temp_path,
        "w"
    ) as h5:

        study_group = h5.create_group(
            patient
        )


        study_group.attrs[
            "source"
        ] = "DICOM"


        study_group.attrs[
            "model"
        ] = "PanEcho"


        study_group.attrs[
            "clip_length"
        ] = CLIP_LEN


        study_group.attrs[
            "clips_per_dicom"
        ] = NUM_CLIPS


        study_group.attrs[
            "seed"
        ] = SEED


        # ----------------------------------------------------
        # Process every DICOM
        # ----------------------------------------------------

        for dicom_number, path in enumerate(
            tqdm(
                dicom_files,
                desc=patient,
                unit="DICOM"
            ),
            start=1
        ):

            try:

                # ------------------------------------------------
                # Relative path inside DICOM/
                # ------------------------------------------------

                relative_path = os.path.relpath(
                    path,
                    dicom_dir
                )


                # HDF5 group names cannot contain "/"
                # because "/" has hierarchical meaning.
                #
                # Example:
                #
                # __82812/foo.dcm
                #
                # becomes:
                #
                # __82812__foo.dcm

                hdf5_name = relative_path.replace(
                    os.sep,
                    "__"
                )


                # ------------------------------------------------
                # READ DICOM
                # ------------------------------------------------

                ds = pydicom.dcmread(
                    path
                )


                frames = ds.pixel_array


                frame_count = len(frames)


                total_frames += frame_count


                # ------------------------------------------------
                # FRAMERATE
                # ------------------------------------------------

                cine_rate = getattr(
                    ds,
                    "CineRate",
                    None
                )


                recommended_rate = getattr(
                    ds,
                    "RecommendedDisplayFrameRate",
                    None
                )


                frame_time = getattr(
                    ds,
                    "FrameTime",
                    None
                )


                framerate = get_framerate(
                    ds
                )


                # ------------------------------------------------
                # PREPROCESS FRAMES
                #
                # Same validated pipeline:
                #
                # DICOM
                #   ↓
                # resize 256x256
                #   ↓
                # /255
                #   ↓
                # HWC -> CHW
                # ------------------------------------------------

                processed_frames = []


                for frame in frames:

                    if frame.ndim == 2:

                        frame = np.stack(
                            [
                                frame,
                                frame,
                                frame
                            ],
                            axis=-1
                        )


                    frame = cv2.resize(
                        frame,
                        (256, 256),
                        interpolation=cv2.INTER_AREA
                    )


                    frame = (
                        frame.astype(
                            np.float32
                        )
                        / 255.0
                    )


                    frame = np.transpose(
                        frame,
                        (2, 0, 1)
                    )


                    processed_frames.append(
                        frame
                    )


                frames_tensor = torch.from_numpy(
                    np.stack(
                        processed_frames
                    )
                )


                # ------------------------------------------------
                # SAMPLE RANDOM CLIPS
                #
                # Same strategy as EchoFocus pull_clip():
                #
                # 16 clips
                # each 16 consecutive frames
                # ------------------------------------------------

                clip_list = []


                for _ in range(NUM_CLIPS):

                    if frame_count < CLIP_LEN:

                        start_idx = 0

                    else:

                        start_idx = np.random.randint(
                            0,
                            frame_count - CLIP_LEN + 1
                        )


                    clip = frames_tensor[
                        start_idx:
                        start_idx + CLIP_LEN
                    ]


                    clip = tv_tensors.Video(
                        clip
                    )


                    clip = Test_Transforms(
                        clip
                    )


                    # T,C,H,W
                    #
                    # ->
                    #
                    # 1,C,T,H,W

                    clip = clip.unsqueeze(
                        0
                    ).transpose(
                        1,
                        2
                    )


                    clip_list.append(
                        clip
                    )


                clips = torch.vstack(
                    clip_list
                )


                # ------------------------------------------------
                # PAN-ECHO
                # ------------------------------------------------

                with torch.inference_mode():

                    embeddings = model(
                        clips.to(device)
                    )


                embeddings = (
                    embeddings
                    .detach()
                    .cpu()
                    .numpy()
                )


                # ------------------------------------------------
                # VALIDATE EMBEDDINGS
                # ------------------------------------------------

                if embeddings.shape != (
                    NUM_CLIPS,
                    768
                ):

                    raise RuntimeError(
                        "Unexpected embedding shape: "
                        f"{embeddings.shape}"
                    )


                if not np.all(
                    np.isfinite(
                        embeddings
                    )
                ):

                    raise RuntimeError(
                        "Embeddings contain NaN or Inf."
                    )


                # ------------------------------------------------
                # WRITE HDF5
                # ------------------------------------------------

                group = study_group.create_group(
                    hdf5_name
                )


                group.create_dataset(
                    "emb",
                    data=embeddings,
                    compression="gzip"
                )


                # ------------------------------------------------
                # Metadata
                # ------------------------------------------------

                group.attrs[
                    "source_relative_path"
                ] = relative_path


                if cine_rate is not None:

                    group.attrs[
                        "CineRate"
                    ] = str(
                        cine_rate
                    )


                if recommended_rate is not None:

                    group.attrs[
                        "RecommendedDisplayFrameRate"
                    ] = str(
                        recommended_rate
                    )


                if frame_time is not None:

                    group.attrs[
                        "FrameTime"
                    ] = str(
                        frame_time
                    )


                if framerate is not None:

                    group.attrs[
                        "framerate_fps"
                    ] = framerate


                group.attrs[
                    "frame_count"
                ] = frame_count


                group.attrs[
                    "clip_length"
                ] = CLIP_LEN


                group.attrs[
                    "num_clips"
                ] = NUM_CLIPS


                successful_dicoms += 1

                total_clips += NUM_CLIPS


                # ------------------------------------------------
                # Free memory
                # ------------------------------------------------

                del frames
                del processed_frames
                del frames_tensor
                del clips
                del embeddings


                if device == "cuda":

                    torch.cuda.empty_cache()


            except Exception as e:

                failed_dicoms += 1


                log_error(
                    patient,
                    path,
                    traceback.format_exc()
                )


                print()
                print(
                    f"WARNING: failed DICOM "
                    f"{path}: {e}"
                )


        # --------------------------------------------------------
        # Study-level metadata
        # --------------------------------------------------------

        study_group.attrs[
            "total_frames"
        ] = total_frames


        study_group.attrs[
            "total_clips"
        ] = total_clips


        study_group.attrs[
            "successful_dicoms"
        ] = successful_dicoms


        study_group.attrs[
            "failed_dicoms"
        ] = failed_dicoms


        # Flush before closing
        h5.flush()


    # ------------------------------------------------------------
    # Atomic completion
    # ------------------------------------------------------------

    if successful_dicoms == 0:

        if os.path.exists(temp_path):
            os.remove(temp_path)


        raise RuntimeError(
            "Patient produced zero successful "
            "DICOM embeddings."
        )


    # ------------------------------------------------------------
    # Final safety check
    # ------------------------------------------------------------

    if os.path.exists(output_path):

        # We should never get here unless another
        # process created the file unexpectedly.

        if is_valid_hdf5(output_path):

            print()
            print(
                "WARNING: valid output already exists:"
            )

            print(output_path)

            print(
                "Removing newly generated temporary result "
                "instead of overwriting existing HDF5."
            )

            os.remove(temp_path)

            return {
                "status": "existing_output",
                "dicom_count": len(dicom_files),
                "successful_dicoms": 0,
                "failed_dicoms": 0,
                "total_frames": 0,
                "total_clips": 0,
                "elapsed": 0.0,
                "output": output_path,
            }


    # Atomic rename
    os.replace(
        temp_path,
        output_path
    )


    elapsed = (
        time.perf_counter()
        - patient_start
    )


    print()
    print(f"Completed: {patient}")

    print(
        f"Successful DICOMs: "
        f"{successful_dicoms}"
    )

    print(
        f"Failed DICOMs: "
        f"{failed_dicoms}"
    )

    print(
        f"Frames: "
        f"{total_frames}"
    )

    print(
        f"Clips: "
        f"{total_clips}"
    )

    print(
        f"Time: "
        f"{elapsed / 60:.2f} min"
    )

    print(
        f"Output: "
        f"{output_path}"
    )


    return {
        "status": (
            "success"
            if failed_dicoms == 0
            else "partial"
        ),
        "dicom_count": len(dicom_files),
        "successful_dicoms": successful_dicoms,
        "failed_dicoms": failed_dicoms,
        "total_frames": total_frames,
        "total_clips": total_clips,
        "elapsed": elapsed,
        "output": output_path,
    }


# ============================================================
# MAIN
# ============================================================

overall_start = time.perf_counter()


for patient in patients:

    try:

        result = process_patient(
            patient
        )


        log_progress(
            [
                patient,
                result["status"],
                result["dicom_count"],
                result["successful_dicoms"],
                result["failed_dicoms"],
                result["total_frames"],
                result["total_clips"],
                f'{result["elapsed"]:.2f}',
                result["output"],
            ]
        )


    except KeyboardInterrupt:

        print()
        print("Interrupted by user.")
        print(
            "Completed patients remain valid."
        )
        print(
            "Any incomplete .tmp file can be removed."
        )

        raise


    except Exception as e:

        elapsed = (
            time.perf_counter()
            - overall_start
        )


        print()
        print("=" * 70)
        print(
            f"PATIENT FAILED: {patient}"
        )
        print("=" * 70)
        print(e)


        log_progress(
            [
                patient,
                "failed",
                "",
                "",
                "",
                "",
                "",
                "",
                "",
            ]
        )


        log_error(
            patient,
            "__PATIENT__",
            traceback.format_exc()
        )


# ============================================================
# SUMMARY
# ============================================================

overall_elapsed = (
    time.perf_counter()
    - overall_start
)


print()
print("=" * 70)
print("PROCESSING FINISHED")
print("=" * 70)


print(
    f"Elapsed: "
    f"{overall_elapsed / 3600:.2f} hours"
)


print(
    f"Progress log: "
    f"{PROGRESS_FILE}"
)


print(
    f"Error log: "
    f"{ERROR_FILE}"
)


print(
    f"Embeddings: "
    f"{OUT_DIR}"
)
