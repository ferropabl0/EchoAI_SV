# Corrected single-frame cine processor.
# This file is configured for the pilot patient first.
# Replace TARGET_PATIENT with None only after the pilot passes.

import os, random, time, csv
import cv2, h5py, numpy as np, pandas as pd, pydicom, torch
import torchvision.transforms.v2 as T
from tqdm import tqdm

SOURCE_DIR = "/run/user/1004/gvfs/smb-share:server=130.241.81.42,share=barnhjärtan/Bilder"
BASE_DIR = os.path.expanduser("~/echofocus")
METADATA_DIR = os.path.join(BASE_DIR, "ventricle_project", "metadata")
OUTPUT_DIR = os.path.join(BASE_DIR, "ventricle_project", "embeddings", "dicom")
LOG_DIR = os.path.join(BASE_DIR, "ventricle_project", "logs")
CLASSIFICATION_CSV = os.path.join(METADATA_DIR, "series_classification.csv")

TARGET_PATIENT = "disk_20260211_160545567_120_120_24"
NUM_CLIPS = 16
CLIP_LEN = 16
SEED = 0
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

Frame_Transform = T.Compose([
    T.CenterCrop((224, 224)),
    T.ToDtype(torch.float32, scale=False),
    T.Normalize(mean=[0.485, 0.456, 0.406],
                std=[0.229, 0.224, 0.225]),
])

def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

def load_panecho():
    print("\nLoading PanEcho...")
    model = torch.hub.load("CarDS-Yale/PanEcho", "PanEcho",
                           force_reload=False, backbone_only=True)
    model = model.to(DEVICE)
    model.eval()
    print("PanEcho loaded.")
    return model

def is_valid_hdf5(path):
    if not os.path.exists(path):
        return False
    try:
        with h5py.File(path, "r") as f:
            if len(f.keys()) != 1:
                return False
            patient = list(f.keys())[0]
            pg = f[patient]
            if len(pg.keys()) == 0:
                return False
            for name in pg:
                if "emb" not in pg[name]:
                    return False
                emb = pg[name]["emb"][()]
                if emb.ndim != 2 or emb.shape[1] != 768:
                    return False
                if not np.isfinite(emb).all():
                    return False
            return True
    except Exception:
        return False

def find_dicom_files(patient_dir):
    result = []
    root0 = os.path.join(patient_dir, "DICOM")
    for root, dirs, files in os.walk(root0):
        dirs[:] = [d for d in dirs if not d.startswith(".")]
        for filename in files:
            if filename.startswith(".") or not filename.lower().endswith(".dcm"):
                continue
            result.append(os.path.join(root, filename))
    return sorted(result)

def discover_series(patient_dir):
    series = {}
    for path in find_dicom_files(patient_dir):
        try:
            ds = pydicom.dcmread(path, stop_before_pixels=True, force=False)
        except Exception:
            continue
        if getattr(ds, "Modality", "") != "US":
            continue
        uid = getattr(ds, "SeriesInstanceUID", None)
        if uid is None:
            continue
        if int(getattr(ds, "NumberOfFrames", 1)) != 1:
            continue
        series.setdefault(str(uid), []).append(path)
    return series

def load_series_frames(paths):
    records = []
    for path in paths:
        try:
            ds = pydicom.dcmread(path, force=False)
            instance = getattr(ds, "InstanceNumber", None)
            if instance is None:
                continue
            records.append((int(instance), path, ds))
        except Exception:
            continue

    if len(records) < CLIP_LEN:
        return None, None

    records.sort(key=lambda x: x[0])

    unique = {}
    for instance, path, ds in records:
        if instance not in unique:
            unique[instance] = (path, ds)

    instances = sorted(unique.keys())
    if len(instances) < CLIP_LEN:
        return None, None

    records = [(i, unique[i][0], unique[i][1]) for i in instances]
    first_ds = records[0][2]
    frame_list = []

    for instance, path, ds in records:
        try:
            arr = ds.pixel_array
        except Exception:
            return None, None

        if arr.ndim != 3 or arr.shape[-1] != 3:
            return None, None

        arr = arr.astype(np.float32)
        if arr.max() > 1.0:
            arr /= 255.0
        arr = np.clip(arr, 0.0, 1.0)

        # Critical fix: standardize spatial size before stacking.
        arr = cv2.resize(arr, (256, 256), interpolation=cv2.INTER_AREA)

        # HWC -> CHW
        arr = np.transpose(arr, (2, 0, 1))
        frame_list.append(arr)

    frames = np.stack(frame_list, axis=0)

    frame_time = getattr(first_ds, "FrameTime", None)
    try:
        frame_time = float(frame_time)
    except Exception:
        frame_time = None

    fps = 1000.0 / frame_time if frame_time and frame_time > 0 else None

    metadata = {
        "frame_count": len(records),
        "instance_first": instances[0],
        "instance_last": instances[-1],
        "frame_time_ms": frame_time,
        "framerate_fps": fps,
        "rows_original": int(records[0][2].Rows),
        "columns_original": int(records[0][2].Columns),
        "photometric": str(getattr(first_ds, "PhotometricInterpretation", "")),
        "sop_class_uid": str(getattr(first_ds, "SOPClassUID", "")),
    }
    return frames, metadata

def make_clips(frames):
    n_frames = frames.shape[0]
    if n_frames < CLIP_LEN:
        return []

    clips = []
    max_start = n_frames - CLIP_LEN

    for _ in range(NUM_CLIPS):
        start = random.randint(0, max_start)
        clip = frames[start:start + CLIP_LEN]  # T,C,H,W

        transformed = []
        for frame in clip:
            x = torch.from_numpy(frame)        # C,H,W
            x = Frame_Transform(x)              # C,224,224
            transformed.append(x)

        x = torch.stack(transformed)            # T,C,H,W
        x = x.permute(1, 0, 2, 3)               # C,T,H,W
        x = x.unsqueeze(0)                      # 1,C,T,H,W
        clips.append(x)

    return clips

@torch.inference_mode()
def get_embeddings(model, clips):
    embeddings = []
    for clip in clips:
        clip = clip.to(DEVICE, non_blocking=True)
        emb = model(clip).detach().float().cpu().numpy()
        if emb.ndim == 1:
            emb = emb[None, :]
        embeddings.append(emb)

    if not embeddings:
        return None
    return np.concatenate(embeddings, axis=0)

def process_patient(patient, series_info, model):
    start_time = time.time()
    patient_dir = os.path.join(SOURCE_DIR, patient)
    output_path = os.path.join(OUTPUT_DIR, f"{patient}_embed.hdf5")
    tmp_path = output_path + ".tmp"

    print("\n" + "=" * 80)
    print(f"PATIENT: {patient}")
    print("=" * 80)
    print("DICOM directory:", os.path.join(patient_dir, "DICOM"))

    if is_valid_hdf5(output_path):
        print("\nExisting valid HDF5 found. Skipping patient.")
        return {"patient": patient, "status": "skipped_existing",
                "series": 0, "failed_series": 0, "frames": 0,
                "clips": 0, "time_min": 0, "output": output_path}

    os.makedirs(OUTPUT_DIR, exist_ok=True)
    discovered = discover_series(patient_dir)

    selected = []
    for uid, paths in discovered.items():
        if uid in series_info:
            selected.append((uid, paths, series_info[uid]["classification"]))
    selected.sort(key=lambda x: x[0])

    print("Single-frame cine series:", len(selected))

    if not selected:
        return {"patient": patient, "status": "no_series",
                "series": 0, "failed_series": 0, "frames": 0,
                "clips": 0, "time_min": 0, "output": ""}

    successful_series = 0
    failed_series = 0
    total_frames = 0
    total_clips = 0

    if os.path.exists(tmp_path):
        os.remove(tmp_path)

    with h5py.File(tmp_path, "w") as h5:
        pg = h5.create_group(patient)
        pg.attrs["source_type"] = "singleframe_cine"
        pg.attrs["clip_length"] = CLIP_LEN
        pg.attrs["num_clips_per_series"] = NUM_CLIPS
        pg.attrs["sampling_seed"] = SEED

        for number, (uid, paths, classification) in enumerate(
            tqdm(selected, desc=patient), start=1
        ):
            try:
                frames, metadata = load_series_frames(paths)
                if frames is None:
                    raise RuntimeError("Could not reconstruct valid cine sequence")

                clips = make_clips(frames)
                if len(clips) != NUM_CLIPS:
                    raise RuntimeError(f"Expected {NUM_CLIPS} clips, got {len(clips)}")

                embeddings = get_embeddings(model, clips)
                if embeddings is None:
                    raise RuntimeError("PanEcho returned no embeddings")
                if embeddings.shape != (NUM_CLIPS, 768):
                    raise RuntimeError(f"Unexpected embedding shape: {embeddings.shape}")
                if not np.isfinite(embeddings).all():
                    raise RuntimeError("Embedding contains NaN/Inf")

                name = f"series_{number:03d}"
                g = pg.create_group(name)
                g.create_dataset("emb", data=embeddings, dtype="float32")
                g.attrs["series_instance_uid"] = uid
                g.attrs["classification"] = classification
                g.attrs["frame_count"] = metadata["frame_count"]
                g.attrs["instance_first"] = metadata["instance_first"]
                g.attrs["instance_last"] = metadata["instance_last"]
                if metadata["frame_time_ms"] is not None:
                    g.attrs["frame_time_ms"] = metadata["frame_time_ms"]
                if metadata["framerate_fps"] is not None:
                    g.attrs["framerate_fps"] = metadata["framerate_fps"]
                g.attrs["rows_original"] = metadata["rows_original"]
                g.attrs["columns_original"] = metadata["columns_original"]
                g.attrs["photometric"] = metadata["photometric"]
                g.attrs["sop_class_uid"] = metadata["sop_class_uid"]
                g.attrs["source_dicom_count"] = len(paths)
                g.attrs["embedding_shape"] = str(embeddings.shape)

                total_frames += frames.shape[0]
                total_clips += embeddings.shape[0]
                successful_series += 1

            except Exception as e:
                failed_series += 1
                print(f"\nFAILED series {uid}: {e}")

    # Never silently accept a partially processed patient.
    if failed_series > 0:
        if os.path.exists(tmp_path):
            os.remove(tmp_path)
        raise RuntimeError(
            f"Patient had {failed_series} failed series. No final HDF5 was created."
        )

    if not is_valid_hdf5(tmp_path):
        if os.path.exists(tmp_path):
            os.remove(tmp_path)
        raise RuntimeError("Temporary HDF5 failed validation")

    if os.path.exists(output_path):
        print("\nValid output appeared during processing. Keeping existing output.")
        os.remove(tmp_path)
        status = "skipped_race"
    else:
        os.replace(tmp_path, output_path)
        status = "completed"

    elapsed = time.time() - start_time

    print(f"\nCompleted: {patient}")
    print(f"Successful series: {successful_series}")
    print(f"Failed series: {failed_series}")
    print(f"Frames: {total_frames}")
    print(f"Clips: {total_clips}")
    print(f"Time: {elapsed / 60:.2f} min")
    print(f"Output: {output_path}")

    return {"patient": patient, "status": status,
            "series": successful_series, "failed_series": failed_series,
            "frames": total_frames, "clips": total_clips,
            "time_min": elapsed / 60, "output": output_path}

def main():
    os.makedirs(LOG_DIR, exist_ok=True)
    set_seed(SEED)

    print("=" * 80)
    print("SINGLE-FRAME CINE -> PAN-ECHO -> HDF5")
    print("=" * 80)
    print("Source:", SOURCE_DIR)
    print("Output:", OUTPUT_DIR)
    print("Classification:", CLASSIFICATION_CSV)
    print("Device:", DEVICE)

    if DEVICE == "cuda":
        print("GPU:", torch.cuda.get_device_name(0))
        print("GPU memory:",
              f"{torch.cuda.get_device_properties(0).total_memory / 1e9:.1f} GB")

    df = pd.read_csv(CLASSIFICATION_CSV)

    required = {"patient", "series_uid", "classification"}
    missing = required - set(df.columns)
    if missing:
        raise RuntimeError(
            f"Missing columns in series_classification.csv: {missing}"
        )

    allowed = {"SINGLEFRAME_CINE_STRONG", "SINGLEFRAME_CINE_POSSIBLE"}
    df = df[df["classification"].isin(allowed)].copy()

    patient_series = {}
    for _, row in df.iterrows():
        patient = str(row["patient"])
        uid = str(row["series_uid"])
        patient_series.setdefault(patient, {})[uid] = {
            "classification": str(row["classification"])
        }

    all_patients = sorted(patient_series)
    print("Patients with eligible cine series:", len(all_patients))
    print("Total eligible series:", len(df))

    if TARGET_PATIENT is not None:
        if TARGET_PATIENT not in patient_series:
            raise RuntimeError(f"Target patient not found: {TARGET_PATIENT}")
        patients = [TARGET_PATIENT]
        print("\nTEST MODE: processing 1 patient")
    else:
        patients = all_patients
        print(f"\nFULL MODE: processing {len(patients)} patients")

    model = load_panecho()
    overall_start = time.time()
    results = []

    for patient in patients:
        results.append(
            process_patient(patient, patient_series[patient], model)
        )

    progress_path = os.path.join(
        LOG_DIR, "singleframe_cine_progress.csv"
    )
    fields = [
        "patient", "status", "series", "failed_series",
        "frames", "clips", "time_min", "output"
    ]
    write_header = not os.path.exists(progress_path)

    with open(progress_path, "a", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fields)
        if write_header:
            writer.writeheader()
        for result in results:
            writer.writerow({k: result.get(k, "") for k in fields})

    elapsed = time.time() - overall_start
    print("\n" + "=" * 80)
    print("PROCESSING FINISHED")
    print("=" * 80)
    print(f"Elapsed: {elapsed / 3600:.2f} hours")
    print("Progress log:", progress_path)
    print("Embeddings:", OUTPUT_DIR)

if __name__ == "__main__":
    main()
