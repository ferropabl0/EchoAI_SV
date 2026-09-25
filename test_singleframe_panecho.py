import os
import h5py
import numpy as np
import torch
import pydicom

from torchvision import tv_tensors
from torchvision.transforms import v2


# ============================================================
# Configuration
# ============================================================

SRC = "/run/user/1004/gvfs/smb-share:server=130.241.81.42,share=barnhjärtan/Bilder"

PATIENT = "disk_20260211_160545567_120_120_24"

SERIES_UID = (
    "1.3.6.1.4.1.9590.100.1.4."
    "319383919407526537121515644511504861475"
)

OUT = os.path.expanduser(
    "~/echofocus/ventricle_project/embeddings/"
    "singleframe_pilot.hdf5"
)

CLIP_LEN = 16
STRIDE = 16


# ============================================================
# Transformations: same preprocessing used by EchoFocus
# ============================================================

Test_Transforms = v2.Compose([
    v2.CenterCrop((224, 224)),
    v2.Normalize(
        mean=[0.485, 0.456, 0.406],
        std=[0.229, 0.224, 0.225],
    ),
])


# ============================================================
# Find frames belonging to the selected series
# ============================================================

dicom_dir = os.path.join(SRC, PATIENT, "DICOM")

frames = []

for root, _, filenames in os.walk(dicom_dir):

    for name in filenames:

        # Ignore macOS metadata files
        if name.startswith("."):
            continue

        if not name.lower().endswith(".dcm"):
            continue

        path = os.path.join(root, name)

        try:
            ds = pydicom.dcmread(
                path,
                stop_before_pixels=True
            )

            if str(
                getattr(ds, "SeriesInstanceUID", "")
            ) != SERIES_UID:
                continue

            instance = int(ds.InstanceNumber)

            frames.append({
                "path": path,
                "instance": instance,
            })

        except Exception as e:
            print("Metadata error:", path)
            print(repr(e))


frames.sort(key=lambda x: x["instance"])


print("=" * 80)
print("Single-frame → PanEcho pilot")
print("=" * 80)
print("Patient:", PATIENT)
print("Series UID:", SERIES_UID)
print("Frames:", len(frames))

if not frames:
    raise RuntimeError("No frames found.")

print(
    "Instance range:",
    frames[0]["instance"],
    "→",
    frames[-1]["instance"]
)


# ============================================================
# Load PanEcho
# ============================================================

device = torch.device(
    "cuda" if torch.cuda.is_available() else "cpu"
)

print("\nDevice:", device)

if device.type == "cuda":
    print(
        "GPU:",
        torch.cuda.get_device_name(0)
    )

print("\nLoading PanEcho...")

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
# Read frames
# ============================================================

print("\nReading frames...")

images = []

for i, item in enumerate(frames):

    ds = pydicom.dcmread(item["path"])

    arr = ds.pixel_array

    if arr.ndim != 3:
        raise RuntimeError(
            f"Unexpected frame shape: {arr.shape}"
        )

    # uint8 HWC → float32 HWC
    arr = arr.astype(np.float32) / 255.0

    # HWC → CHW
    arr = np.transpose(arr, (2, 0, 1))

    images.append(arr)

    if (i + 1) % 25 == 0 or i == 0:
        print(f"  {i + 1}/{len(frames)}")


images = np.stack(images)

print("\nLoaded image array:")
print("shape:", images.shape)
print("dtype:", images.dtype)
print("min:", images.min())
print("max:", images.max())


# ============================================================
# Create clips
# ============================================================

num_clips = (len(images) - CLIP_LEN) // STRIDE + 1

print("\nNumber of clips:", num_clips)

all_embeddings = []


# ============================================================
# PanEcho inference
# ============================================================

with torch.inference_mode():

    for clip_idx in range(num_clips):

        start = clip_idx * STRIDE
        end = start + CLIP_LEN

        clip = images[start:end]

        # T,C,H,W
        clip = torch.from_numpy(clip)

        clip = tv_tensors.Video(clip)

        # Apply EchoFocus preprocessing
        clip = Test_Transforms(clip)

        # C,T,H,W
        clip = clip.permute(1, 0, 2, 3)

        # Add batch dimension
        # B,C,T,H,W
        clip = clip.unsqueeze(0)

        clip = clip.to(device)

        embedding = model(clip)

        embedding = embedding.detach().cpu().numpy()

        print(
            f"Clip {clip_idx + 1}/{num_clips}:",
            embedding.shape
        )

        all_embeddings.append(embedding)


# ============================================================
# Combine embeddings
# ============================================================

embeddings = np.concatenate(
    all_embeddings,
    axis=0
)

print("\n" + "=" * 80)
print("RESULT")
print("=" * 80)

print("Embeddings shape:", embeddings.shape)
print("dtype:", embeddings.dtype)
print("NaN:", np.isnan(embeddings).sum())
print("Inf:", np.isinf(embeddings).sum())


# ============================================================
# Save HDF5
# ============================================================

os.makedirs(
    os.path.dirname(OUT),
    exist_ok=True
)

with h5py.File(OUT, "w") as f:

    patient_group = f.create_group(PATIENT)

    series_group = patient_group.create_group(
        "series_1"
    )

    series_group.create_dataset(
        "emb",
        data=embeddings,
        compression="gzip"
    )

    series_group.attrs["series_uid"] = SERIES_UID
    series_group.attrs["num_frames"] = len(frames)
    series_group.attrs["clip_length"] = CLIP_LEN
    series_group.attrs["stride"] = STRIDE
    series_group.attrs["num_clips"] = num_clips

    patient_group.attrs["patient"] = PATIENT


print("\nSaved:")
print(OUT)


# ============================================================
# Validate HDF5
# ============================================================

with h5py.File(OUT, "r") as f:

    emb = f[PATIENT]["series_1"]["emb"][()]

    print("\nHDF5 validation:")
    print("shape:", emb.shape)
    print("finite:", np.isfinite(emb).all())

    assert emb.shape == (num_clips, 768)
    assert np.isfinite(emb).all()

print("\n✓ Validation successful.")
