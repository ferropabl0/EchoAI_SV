import h5py

path = "path/to/one_patient_embed.hdf5"

with h5py.File(path, "r") as f:
    patient = list(f.keys())[0]

    print("Patient:", patient)

    for name in f[patient]:
        group = f[patient][name]
        print(name, group["emb"].shape)
