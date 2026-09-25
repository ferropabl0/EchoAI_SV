import os
import random
import time
import csv
import tempfile

import cv2
import h5py
import numpy as np
import pydicom
import torch
from tqdm import tqdm

from torchvision.transforms import v2

from echofocus import EchoFocus
from echofocus import load_model_and_random_state


# =============================================================================
# CONFIGURATION
# =============================================================================

SOURCE_DIR = (
    "/run/user/1004/gvfs/"
    "smb-share:server=130.241.81.42,share=barnhjärtan/Bilder"
)

OUTPUT_DIR = os.path.expanduser(
    "~/echofocus/ventricle_project/embeddings/dicom"
)

MODEL_NAME = "EchoFocus_CHD_test"

NUM_CLIPS = 16
CLIP_LEN = 16
SEED = 0


# These are the 7 remaining patients identified by the validated
# single-frame cine audit.
TARGET_PATIENTS = [
    "disk_20260211_232245567_203_203_116",
    "disk_20260212_005448490_227_227_142",
]

# Pilot already processed separately.
PILOT_PATIENT = (
    "disk_20260211_160545567_120_120_24"
)


PROGRESS_LOG = os.path.join(
    OUTPUT_DIR,
    "singleframe_production_progress.csv"
)

ERROR_LOG = os.path.join(
    OUTPUT_DIR,
    "singleframe_production_errors.csv"
)

PREDICTION_LOG = os.path.join(
    OUTPUT_DIR,
    "singleframe_chd_predictions.csv"
)


# =============================================================================
# REPRODUCIBILITY
# =============================================================================

random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)

if torch.cuda.is_available():
    torch.cuda.manual_seed_all(SEED)


# =============================================================================
# DEVICE
# =============================================================================

DEVICE = torch.device(
    "cuda" if torch.cuda.is_available() else "cpu"
)

print("=" * 80)
print("SINGLE-FRAME CINE PRODUCTION PIPELINE")
print("=" * 80)

print("Device:", DEVICE)

if torch.cuda.is_available():
    print(
        "GPU:",
        torch.cuda.get_device_name(0)
    )


# =============================================================================
# OUTPUT DIRECTORY
# =============================================================================

os.makedirs(
    OUTPUT_DIR,
    exist_ok=True
)


# =============================================================================
# FRAME TRANSFORM
# =============================================================================
#
# Same preprocessing validated in the pilot:
#
# RGB frame
#   -> 256 x 256
#   -> [0,1]
#   -> CenterCrop 224 x 224
#   -> ImageNet normalization
#
# Applied independently to each frame.
# =============================================================================

FRAME_TRANSFORM = v2.Compose([
    v2.ToImage(),
    v2.ToDtype(torch.float32, scale=False),
    v2.CenterCrop((224, 224)),
    v2.Normalize(
        mean=[0.485, 0.456, 0.406],
        std=[0.229, 0.224, 0.225],
    ),
])


# =============================================================================
# LOAD PANECHO
# =============================================================================

print("\nLoading PanEcho...")

panecho = torch.hub.load(
    "CarDS-Yale/PanEcho",
    "PanEcho",
    force_reload=False,
    backbone_only=True,
)

panecho = panecho.to(DEVICE)
panecho.eval()

print("PanEcho loaded.")


# =============================================================================
# LOAD ECHOFOCUS
# =============================================================================

print("\nLoading EchoFocus...")

echo = EchoFocus(
    model_name=MODEL_NAME,
    dataset="outside",
    task="chd",
    epoch_lim=-1,
    batch_size=1,
    test_only=True,
)

model, _, _, _, _ = echo._setup_model()

checkpoint_path = os.path.join(
    echo.model_path,
    "best_checkpoint.pt"
)

if not os.path.exists(checkpoint_path):
    raise FileNotFoundError(
        f"Checkpoint not found: {checkpoint_path}"
    )

model, _, _, _, _ = load_model_and_random_state(
    checkpoint_path,
    model,
)

model = model.to(DEVICE)
model.eval()

print("EchoFocus loaded.")
print("Checkpoint:", checkpoint_path)
print("Task labels:")
print(echo.task_labels)


# =============================================================================
# HELPERS
# =============================================================================

def append_csv(path, row, fieldnames):
    """Append a row to a CSV, creating it if necessary."""

    exists = os.path.exists(path)

    with open(
        path,
        "a",
        newline=""
    ) as f:

        writer = csv.DictWriter(
            f,
            fieldnames=fieldnames
        )

        if not exists:
            writer.writeheader()

        writer.writerow(row)


def read_singleframe_cine_series(series_files):
    """
    Read one single-frame cine series in temporal order.

    Returns:
        frames: list of RGB uint8 arrays
        metadata: basic sequence metadata
    """

    datasets = []

    for path in series_files:

        if os.path.basename(path).startswith("."):
            continue

        try:
            ds = pydicom.dcmread(
                path,
                force=False,
                stop_before_pixels=False,
            )

        except Exception:
            continue

        if getattr(ds, "Modality", None) != "US":
            continue

        if int(
            getattr(ds, "NumberOfFrames", 1)
        ) != 1:
            continue

        if "PixelData" not in ds:
            continue

        instance = getattr(
            ds,
            "InstanceNumber",
            None
        )

        if instance is None:
            instance = 10**9

        datasets.append(
            (
                int(instance),
                path,
                ds
            )
        )

    if not datasets:
        raise RuntimeError(
            "No readable single-frame US DICOMs."
        )

    # Temporal ordering.
    datasets.sort(
        key=lambda x: x[0]
    )

    # Remove duplicate InstanceNumbers.
    unique = []
    seen_instances = set()

    for instance, path, ds in datasets:

        if instance in seen_instances:
            continue

        seen_instances.add(instance)

        unique.append(
            (
                instance,
                path,
                ds
            )
        )

    frames = []

    for instance, path, ds in unique:

        arr = ds.pixel_array

        if arr.ndim != 3 or arr.shape[-1] != 3:
            raise RuntimeError(
                f"Expected RGB image, got {arr.shape}"
            )

        if arr.dtype != np.uint8:

            arr = arr.astype(
                np.float32
            )

            if arr.max() > 1.0:
                arr = arr / 255.0

            arr = np.clip(
                arr * 255.0,
                0,
                255
            ).astype(np.uint8)

        # Same resize used in the validated pilot.
        arr = cv2.resize(
            arr,
            (256, 256),
            interpolation=cv2.INTER_LINEAR
        )

        frames.append(arr)

    metadata = {
        "n_frames": len(frames),
        "first_instance": unique[0][0],
        "last_instance": unique[-1][0],
    }

    return frames, metadata


def make_random_clips(
    frames,
    num_clips=16,
    clip_len=16
):
    """
    Generate random temporal clips.

    Uses seed 0, as in the validated pilot.
    """

    n_frames = len(frames)

    rng = np.random.default_rng(
        SEED
    )

    clips = []

    for _ in range(num_clips):

        if n_frames < clip_len:

            indices = list(
                range(n_frames)
            )

            while len(indices) < clip_len:
                indices.append(
                    n_frames - 1
                )

        else:

            max_start = (
                n_frames - clip_len
            )

            start = int(
                rng.integers(
                    0,
                    max_start + 1
                )
            )

            indices = list(
                range(
                    start,
                    start + clip_len
                )
            )

        clips.append(
            [frames[i] for i in indices]
        )

    return clips


def embed_clip(clip):
    """
    Convert one temporal RGB clip into PanEcho input.

    Returns:
        Tensor of shape (1, 3, 16, 224, 224)
    """

    transformed_frames = []

    for frame in clip:

        # uint8 -> [0,1]
        frame_float = (
            frame.astype(
                np.float32
            ) / 255.0
        )

        # HWC -> CHW
        tensor = torch.from_numpy(
            frame_float.transpose(
                2, 0, 1
            )
        )

        tensor = FRAME_TRANSFORM(
            tensor
        )

        transformed_frames.append(
            tensor
        )

    # T,C,H,W
    clip_tensor = torch.stack(
        transformed_frames,
        dim=0
    )

    # C,T,H,W
    clip_tensor = clip_tensor.permute(
        1, 0, 2, 3
    )

    # B,C,T,H,W
    clip_tensor = clip_tensor.unsqueeze(
        0
    )

    return clip_tensor.to(
        DEVICE,
        non_blocking=True
    )


def process_series(series_files):
    """
    Generate 16 PanEcho embeddings for one cine series.

    Returns:
        embeddings: (16,768)
        metadata
    """

    frames, metadata = (
        read_singleframe_cine_series(
            series_files
        )
    )

    if len(frames) < 1:
        raise RuntimeError(
            "Series contains no usable frames."
        )

    clips = make_random_clips(
        frames,
        NUM_CLIPS,
        CLIP_LEN
    )

    embeddings = []

    with torch.no_grad():

        for clip in clips:

            x = embed_clip(clip)

            emb = panecho(x)

            emb = (
                emb
                .detach()
                .cpu()
                .numpy()
            )

            if emb.shape != (1, 768):
                raise RuntimeError(
                    "Unexpected PanEcho "
                    f"output: {emb.shape}"
                )

            embeddings.append(
                emb[0]
            )

    embeddings = np.stack(
        embeddings,
        axis=0
    ).astype(np.float32)

    if embeddings.shape != (
        NUM_CLIPS,
        768
    ):
        raise RuntimeError(
            "Unexpected embedding shape: "
            f"{embeddings.shape}"
        )

    if not np.isfinite(
        embeddings
    ).all():
        raise RuntimeError(
            "PanEcho produced NaN/Inf."
        )

    return embeddings, metadata


def discover_cine_series(patient_dir):
    """
    Discover eligible single-frame US cine series.

    Groups DICOM files by SeriesInstanceUID.

    The patients are already restricted to the seven patients
    identified by the validated audit.
    """

    dicom_dir = os.path.join(
        patient_dir,
        "DICOM"
    )

    if not os.path.isdir(dicom_dir):
        raise RuntimeError(
            f"DICOM directory not found: "
            f"{dicom_dir}"
        )

    series = {}

    for root, dirs, files in os.walk(
        dicom_dir
    ):

        dirs[:] = [
            d for d in dirs
            if not d.startswith(".")
        ]

        for filename in files:

            if filename.startswith("."):
                continue

            path = os.path.join(
                root,
                filename
            )

            try:

                ds = pydicom.dcmread(
                    path,
                    stop_before_pixels=True,
                    force=False
                )

            except Exception:
                continue

            if getattr(
                ds,
                "Modality",
                None
            ) != "US":
                continue

            if int(
                getattr(
                    ds,
                    "NumberOfFrames",
                    1
                )
            ) != 1:
                continue

            uid = getattr(
                ds,
                "SeriesInstanceUID",
                None
            )

            if uid is None:
                continue

            series.setdefault(
                uid,
                []
            ).append(path)

    eligible = []

    for uid, files in series.items():

        metadata = []

        for path in files:

            try:

                ds = pydicom.dcmread(
                    path,
                    stop_before_pixels=True,
                    force=False
                )

            except Exception:
                continue

            instance = getattr(
                ds,
                "InstanceNumber",
                None
            )

            frame_time = getattr(
                ds,
                "FrameTime",
                None
            )

            cine_rate = getattr(
                ds,
                "CineRate",
                None
            )

            metadata.append(
                (
                    instance,
                    frame_time,
                    cine_rate
                )
            )

        n = len(metadata)

        # Eligibility criterion consistent with the validated
        # series_classification.csv:
        # at least 16 frames.
        #
        # Temporal metadata (FrameTime/CineRate) is NOT required
        # because these two patients were classified as
        # SINGLEFRAME_CINE_POSSIBLE based on sequential images.
        if n < 16:
            continue

        eligible.append(
            (
                uid,
                files
            )
        )

    # Stable ordering.
    def series_sort_key(item):

        uid, files = item

        instances = []

        for path in files:

            try:

                ds = pydicom.dcmread(
                    path,
                    stop_before_pixels=True,
                    force=False
                )

                instance = getattr(
                    ds,
                    "InstanceNumber",
                    10**9
                )

                instances.append(
                    int(instance)
                )

            except Exception:
                pass

        return (
            min(instances)
            if instances
            else 10**9,
            uid
        )

    eligible.sort(
        key=series_sort_key
    )

    return eligible


def atomic_write_hdf5(
    patient,
    series_embeddings,
    series_metadata,
    output_path
):
    """
    Write one patient HDF5 atomically and validate it.
    """

    directory = os.path.dirname(
        output_path
    )

    fd, tmp_path = tempfile.mkstemp(
        prefix=".tmp_",
        suffix=".hdf5",
        dir=directory
    )

    os.close(fd)

    try:

        with h5py.File(
            tmp_path,
            "w"
        ) as f:

            patient_group = (
                f.create_group(patient)
            )

            for i, emb in enumerate(
                series_embeddings,
                start=1
            ):

                group = (
                    patient_group.create_group(
                        f"series_{i:03d}"
                    )
                )

                group.create_dataset(
                    "emb",
                    data=emb,
                    dtype="float32"
                )

                metadata = (
                    series_metadata[i - 1]
                )

                group.attrs[
                    "n_frames"
                ] = metadata["n_frames"]

                group.attrs[
                    "first_instance"
                ] = metadata[
                    "first_instance"
                ]

                group.attrs[
                    "last_instance"
                ] = metadata[
                    "last_instance"
                ]

            f.flush()

        # Validate before final replacement.
        with h5py.File(
            tmp_path,
            "r"
        ) as f:

            pg = f[patient]

            if len(pg.keys()) != len(
                series_embeddings
            ):
                raise RuntimeError(
                    "HDF5 validation failed: "
                    "wrong number of series."
                )

            for name in pg.keys():

                shape = pg[name][
                    "emb"
                ].shape

                if shape != (
                    NUM_CLIPS,
                    768
                ):
                    raise RuntimeError(
                        f"HDF5 validation failed: "
                        f"{name}: {shape}"
                    )

        os.replace(
            tmp_path,
            output_path
        )

    except Exception:

        if os.path.exists(
            tmp_path
        ):
            os.remove(tmp_path)

        raise


# =============================================================================
# MAIN
# =============================================================================

print("\n" + "=" * 80)
print("TARGET PATIENTS")
print("=" * 80)

for patient in TARGET_PATIENTS:
    print(" ", patient)

print("\nPilot excluded:")
print(" ", PILOT_PATIENT)


progress_fields = [
    "patient",
    "status",
    "n_series",
    "n_frames",
    "n_embeddings",
    "elapsed_min",
    "output_path",
]

error_fields = [
    "patient",
    "series",
    "error",
]

prediction_fields = [
    "patient",
    "task",
    "logit",
    "probability",
]


all_start = time.time()


for patient in TARGET_PATIENTS:

    patient_start = time.time()

    print("\n" + "=" * 80)
    print("PATIENT:", patient)
    print("=" * 80)

    patient_dir = os.path.join(
        SOURCE_DIR,
        patient
    )

    output_path = os.path.join(
        OUTPUT_DIR,
        f"{patient}_embed.hdf5"
    )

    # -------------------------------------------------------------------------
    # SAFETY CHECKS
    # -------------------------------------------------------------------------

    if patient == PILOT_PATIENT:
        print("Skipping pilot.")
        continue

    if not os.path.isdir(
        patient_dir
    ):

        print(
            "ERROR: patient directory not found:",
            patient_dir
        )

        append_csv(
            ERROR_LOG,
            {
                "patient": patient,
                "series": "",
                "error":
                    "patient directory not found",
            },
            error_fields
        )

        continue

    # Never overwrite an existing output.
    if os.path.exists(
        output_path
    ):

        print(
            "Output already exists; skipping:",
            output_path
        )

        append_csv(
            PROGRESS_LOG,
            {
                "patient": patient,
                "status":
                    "already_exists",
                "n_series": "",
                "n_frames": "",
                "n_embeddings": "",
                "elapsed_min": 0,
                "output_path":
                    output_path,
            },
            progress_fields
        )

        continue

    # -------------------------------------------------------------------------
    # DISCOVER CINE SERIES
    # -------------------------------------------------------------------------

    try:

        eligible_series = (
            discover_cine_series(
                patient_dir
            )
        )

    except Exception as e:

        print(
            "ERROR discovering series:",
            e
        )

        append_csv(
            ERROR_LOG,
            {
                "patient": patient,
                "series": "",
                "error": str(e),
            },
            error_fields
        )

        continue

    print(
        "Eligible single-frame cine series:",
        len(eligible_series)
    )

    if len(eligible_series) == 0:

        append_csv(
            ERROR_LOG,
            {
                "patient": patient,
                "series": "",
                "error":
                    "no eligible cine series",
            },
            error_fields
        )

        continue

    # -------------------------------------------------------------------------
    # PROCESS SERIES
    # -------------------------------------------------------------------------

    patient_embeddings = []
    patient_metadata = []

    successful_series = 0
    total_frames = 0

    for series_number, (
        uid,
        series_files
    ) in enumerate(
        tqdm(
            eligible_series,
            desc=patient
        ),
        start=1
    ):

        series_name = (
            f"series_{series_number:03d}"
        )

        try:

            emb, metadata = process_series(
                series_files
            )

            patient_embeddings.append(
                emb
            )

            patient_metadata.append(
                metadata
            )

            successful_series += 1

            total_frames += (
                metadata["n_frames"]
            )

        except Exception as e:

            print(
                f"\nERROR {series_name}: {e}"
            )

            append_csv(
                ERROR_LOG,
                {
                    "patient": patient,
                    "series": series_name,
                    "error": str(e),
                },
                error_fields
            )

    # -------------------------------------------------------------------------
    # WRITE HDF5
    # -------------------------------------------------------------------------

    if successful_series == 0:

        append_csv(
            PROGRESS_LOG,
            {
                "patient": patient,
                "status": "failed",
                "n_series":
                    len(eligible_series),
                "n_frames": total_frames,
                "n_embeddings": 0,
                "elapsed_min": (
                    time.time()
                    - patient_start
                ) / 60,
                "output_path": "",
            },
            progress_fields
        )

        continue

    try:

        atomic_write_hdf5(
            patient,
            patient_embeddings,
            patient_metadata,
            output_path
        )

        print(
            "\nHDF5 written:",
            output_path
        )

    except Exception as e:

        print(
            "ERROR writing HDF5:",
            e
        )

        append_csv(
            ERROR_LOG,
            {
                "patient": patient,
                "series": "",
                "error":
                    f"HDF5 write: {e}",
            },
            error_fields
        )

        continue

    # -------------------------------------------------------------------------
    # ECHOFOCUS INFERENCE
    # -------------------------------------------------------------------------

    print("\nRunning EchoFocus...")

    try:

        # One tensor per series:
        # each tensor = (16,768)
        model_input = [
            torch.tensor(
                emb,
                dtype=torch.float32,
                device=DEVICE
            )
            for emb in patient_embeddings
        ]

        with torch.no_grad():

            logits = model(
                model_input
            )

        logits = (
            logits
            .detach()
            .cpu()
            .numpy()
            .reshape(-1)
        )

        expected_outputs = len(
            echo.task_labels
        )

        if logits.shape[0] != (
            expected_outputs
        ):
            raise RuntimeError(
                f"Expected "
                f"{expected_outputs} outputs, "
                f"got {logits.shape[0]}"
            )

        if not np.isfinite(
            logits
        ).all():

            raise RuntimeError(
                "EchoFocus output contains "
                "NaN/Inf."
            )

        # EchoFocus CHD classification uses
        # sigmoid on the raw logits.
        probabilities = (
            1.0
            / (
                1.0
                + np.exp(-logits)
            )
        )

        print(
            "EchoFocus output shape:",
            logits.shape
        )

        print("\nPredictions:")

        for task, logit, prob in zip(
            echo.task_labels,
            logits,
            probabilities
        ):

            print(
                f"{task:30s} "
                f"logit={logit:9.4f} "
                f"prob={prob:.6f}"
            )

            append_csv(
                PREDICTION_LOG,
                {
                    "patient": patient,
                    "task": task,
                    "logit": float(logit),
                    "probability": float(prob),
                },
                prediction_fields
            )

    except Exception as e:

        print(
            "ERROR during EchoFocus inference:",
            e
        )

        append_csv(
            ERROR_LOG,
            {
                "patient": patient,
                "series": "",
                "error":
                    f"EchoFocus: {e}",
            },
            error_fields
        )

        continue

    # -------------------------------------------------------------------------
    # PROGRESS
    # -------------------------------------------------------------------------

    elapsed_min = (
        time.time()
        - patient_start
    ) / 60

    total_embeddings = (
        successful_series
        * NUM_CLIPS
    )

    append_csv(
        PROGRESS_LOG,
        {
            "patient": patient,
            "status": "success",
            "n_series": successful_series,
            "n_frames": total_frames,
            "n_embeddings":
                total_embeddings,
            "elapsed_min":
                round(elapsed_min, 3),
            "output_path":
                output_path,
        },
        progress_fields
    )

    print("\nPatient finished.")
    print(
        "Series:",
        successful_series
    )
    print(
        "Frames:",
        total_frames
    )
    print(
        "Embeddings:",
        total_embeddings
    )
    print(
        "Time:",
        f"{elapsed_min:.2f} min"
    )


# =============================================================================
# FINISHED
# =============================================================================

total_elapsed = (
    time.time()
    - all_start
) / 60

print("\n" + "=" * 80)
print("PRODUCTION RUN FINISHED")
print("=" * 80)

print(
    "Total elapsed:",
    f"{total_elapsed:.2f} min"
)

print("\nProgress log:")
print(PROGRESS_LOG)

print("\nError log:")
print(ERROR_LOG)

print("\nPrediction log:")
print(PREDICTION_LOG)
