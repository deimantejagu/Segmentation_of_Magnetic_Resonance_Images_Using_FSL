import os
import nibabel as nib
import numpy as np
import shutil
import json

from nipype.interfaces.ants import N4BiasFieldCorrection

with open("configs/fmgan.json", "r") as f:
    config = json.load(f)

phase = config["phase"]
num_mod = config["num_modalities"]
volume_shape = config["volume_shape"]

def get_mrbrains_filename(case_idx, modality, data_dir, phase):
    folder = os.path.join(data_dir, phase, str(case_idx))
    if modality.lower() in ['labels', 'label']:
        if phase.lower() == "training":
            filename = "LabelsForTraining.nii"
        else:
            filename = "LabelsForTesting.nii"
    else:
        filename = f"{modality}.nii"
    return os.path.join(folder, filename)

def read_data(case_idx, modality, data_dir, phase):
    image_path = get_mrbrains_filename(case_idx, modality, data_dir, phase)
    print("Loading:", image_path)
    return nib.load(image_path)

def read_vol(case_idx, modality, data_dir, phase):
    image_data = read_data(case_idx, modality, data_dir, phase)
    return image_data.get_fdata(dtype=np.float32)

def correct_bias(in_file, out_file):
    correct = N4BiasFieldCorrection()
    correct.inputs.input_image = in_file
    correct.inputs.output_image = out_file
    done = correct.run()
    return done.outputs.output_image

def normalise(case_idx, modality, in_dir, out_dir, copy=False):
    image_in_path = get_mrbrains_filename(case_idx, modality, in_dir)
    image_out_path = get_mrbrains_filename(case_idx, modality, out_dir)
    if copy:
        shutil.copy(image_in_path, image_out_path)
    else:
        correct_bias(image_in_path, image_out_path)
    print(image_in_path + " done.")

# Extraxt patches from 3D volume
def extract_patches(volume, patch_shape, extraction_step, datype='float32'):
    patch_h, patch_w, patch_d = patch_shape
    stride_h, stride_w, stride_d = extraction_step
    img_h, img_w, img_d = volume.shape
    N_patches_h = (img_h - patch_h) // stride_h + 1
    N_patches_w = (img_w - patch_w) // stride_w + 1
    N_patches_d = (img_d - patch_d) // stride_d + 1
    N_patches_img = N_patches_h * N_patches_w * N_patches_d
    raw_patch_matrix = np.zeros((N_patches_img, patch_h, patch_w, patch_d), dtype=datype)
    k = 0
    for h in range(N_patches_h):
        for w in range(N_patches_w):
            for d in range(N_patches_d):
                raw_patch_matrix[k] = volume[h*stride_h : h*stride_h+patch_h,
                                             w*stride_w : w*stride_w+patch_w,
                                             d*stride_d : d*stride_d+patch_d]
                k += 1
    assert(k == N_patches_img)
    return raw_patch_matrix

# Remap labels to 1, 2, 3 and all other labels to 0
def remap_labels(labels):
    return np.where(np.isin(labels, [1, 2, 3]), labels, 0)

# Get patches and labels from labeled images 
def get_patches_lab(T1_vols, T1IR_vols, FLAIR_vols, label_vols, extraction_step,
                    patch_shape, validating, testing, num_images_training):
    patch_shape_1d = patch_shape[0]
    x = np.zeros((0, patch_shape_1d, patch_shape_1d, patch_shape_1d, num_mod), dtype="float32")
    y = np.zeros((0, patch_shape_1d, patch_shape_1d, patch_shape_1d), dtype="uint8")
    for idx in range(len(T1_vols)):
        y_length = len(y)
        if testing:
            print("Extracting Patches from Image %2d ...." % (num_images_training + idx + 2))
        elif validating:
            print("Extracting Patches from Image %2d ...." % (num_images_training + idx + 1))
        else:
            print("Extracting Patches from Image %2d ...." % (1 + idx))
        label_patches = extract_patches(label_vols[idx], patch_shape, extraction_step, datype="uint8")
        # Filtrating patches based on the number of labels
        if testing or validating:
            valid_idxs = np.where(np.sum(label_patches, axis=(1, 2, 3)) != -1)
        else:
            has_gm = np.any(label_patches == 2, axis=(1, 2, 3))
            enough_foreground = np.count_nonzero(label_patches, axis=(1, 2, 3)) > 6000
            valid_idxs = np.where(has_gm | enough_foreground)
        label_patches = label_patches[valid_idxs]
        x = np.vstack((x, np.zeros((len(label_patches), patch_shape_1d, patch_shape_1d, patch_shape_1d, num_mod), dtype="float32")))
        y = np.vstack((y, np.zeros((len(label_patches), patch_shape_1d, patch_shape_1d, patch_shape_1d), dtype="uint8")))
        y[y_length:, :, :, :] = label_patches
        T1_train = extract_patches(T1_vols[idx], patch_shape, extraction_step, datype="float32")
        x[y_length:, :, :, :, 0] = T1_train[valid_idxs]
        T1IR_train = extract_patches(T1IR_vols[idx], patch_shape, extraction_step, datype="float32")
        x[y_length:, :, :, :, 1] = T1IR_train[valid_idxs]
        FLAIR_train = extract_patches(FLAIR_vols[idx], patch_shape, extraction_step, datype="float32")
        x[y_length:, :, :, :, 2] = FLAIR_train[valid_idxs]
        
    y = remap_labels(y)
    return x, y

# Seperate patches to training and validation sets
def preprocess_dynamic_lab(data_dir, seed, num_classes, extraction_step, patch_shape, 
                           num_images_training, num_images_validating, num_images_testing, 
                           validating=False, testing=False, volume_shape=volume_shape):
    
    if testing:
        phase = "testing"
        cases = sorted(os.listdir(os.path.join(data_dir, "testing")))
        cases = [int(case) for case in cases if case.isdigit()]
        cases.sort()
        n_vols = len(cases)
        print(f"Testing with {n_vols} images: {cases}")
        
    elif validating:
        phase = "training"
        cases = list(range(num_images_training + 1, num_images_training + num_images_validating + 1))
        n_vols = num_images_validating
        print(f"Validating with {n_vols} images: {cases}")

    else: 
        phase = "training"
        cases = list(range(1, num_images_training + 1))
        n_vols = num_images_training
        print(f"Training with {n_vols} images: {cases}")

    T1_vols    = np.empty((n_vols, *volume_shape), dtype="float32")
    T1IR_vols  = np.empty((n_vols, *volume_shape), dtype="float32")
    FLAIR_vols = np.empty((n_vols, *volume_shape), dtype="float32")
    label_vols = np.empty((n_vols, *volume_shape), dtype="uint8")
        
    for i, case_idx in enumerate(cases):
        print(f"Processing subject index: {case_idx}")
        T1_vols[i]    = read_vol(case_idx, 'T1',    data_dir, phase)
        T1IR_vols[i]  = read_vol(case_idx, 'T1_IR', data_dir, phase)
        FLAIR_vols[i] = read_vol(case_idx, 'FLAIR', data_dir, phase)
        label_vols[i] = read_vol(case_idx, 'Labels', data_dir, phase)

    T1_vols = (T1_vols - np.mean(T1_vols)) / np.std(T1_vols)
    T1IR_vols = (T1IR_vols - np.mean(T1IR_vols)) / np.std(T1IR_vols)
    FLAIR_vols = (FLAIR_vols - np.mean(FLAIR_vols)) / np.std(FLAIR_vols)

    for vols in [T1_vols, T1IR_vols, FLAIR_vols]:
        for i in range(vols.shape[0]):
            min_val, max_val = np.min(vols[i]), np.max(vols[i])
            vols[i] = ((vols[i] - min_val) / (max_val - min_val)) * 255
            vols[i] = vols[i] / 127.5 - 1.0

    x, y = get_patches_lab(
        T1_vols, T1IR_vols, FLAIR_vols, label_vols,
        extraction_step, patch_shape,
        validating=validating, testing=testing,
        num_images_training=num_images_training
    )

    print("Total Extracted Labeled Patches Shape:", x.shape, y.shape)

    if testing:
        return np.rollaxis(x, 4, 1), label_vols
    elif validating:
        return np.rollaxis(x, 4, 1), y, label_vols
    else:
        return np.rollaxis(x, 4, 1), y

# Extract patches from unlabeled images
def get_patches_unlab(T1_vols, T1IR_vols, FLAIR_vols, extraction_step, patch_shape, data_dir):
    patch_shape_1d = patch_shape[0]
    x = np.zeros((0, patch_shape_1d, patch_shape_1d, patch_shape_1d, num_mod), dtype="float32")
    label_ref = read_vol(1, 'Labels', data_dir, phase)
    for idx in range(len(T1_vols)):
        x_length = len(x)
        print("Processing the Unlabeled Image %2d ...." % (idx+1))
        label_patches = extract_patches(label_ref, patch_shape, extraction_step, datype="uint8")
        has_gm = np.any(label_patches == 2, axis=(1, 2, 3))
        enough_foreground = np.count_nonzero(label_patches, axis=(1, 2, 3)) > 6000
        valid_idxs = np.where(has_gm | enough_foreground)
        label_patches = label_patches[valid_idxs]
        x = np.vstack((x, np.zeros((len(label_patches), patch_shape_1d,
                                      patch_shape_1d, patch_shape_1d, num_mod), dtype="float32")))
        T1_train = extract_patches(T1_vols[idx], patch_shape, extraction_step, datype="float32")
        x[x_length:, :, :, :, 0] = T1_train[valid_idxs]
        T1IR_train = extract_patches(T1IR_vols[idx], patch_shape, extraction_step, datype="float32")
        x[x_length:, :, :, :, 1] = T1IR_train[valid_idxs]
        FLAIR_train = extract_patches(FLAIR_vols[idx], patch_shape, extraction_step, datype="float32")
        x[x_length:, :, :, :, 2] = FLAIR_train[valid_idxs]
    return x

# Generate patches for training
def preprocess_dynamic_unlab(data_dir, extraction_step, patch_shape, num_images_training_unlab, volume_shape=volume_shape):
    T1_vols    = np.empty((num_images_training_unlab, volume_shape[0], volume_shape[1], volume_shape[2]), dtype="float32")
    T1IR_vols  = np.empty((num_images_training_unlab, volume_shape[0], volume_shape[1], volume_shape[2]), dtype="float32")
    FLAIR_vols = np.empty((num_images_training_unlab, volume_shape[0], volume_shape[1], volume_shape[2]), dtype="float32")
    for case_idx in range(1, 1+num_images_training_unlab):
        T1_vols[case_idx-1, :, :, :]    = read_vol(case_idx, 'T1', data_dir, phase="testing")
        T1IR_vols[case_idx-1, :, :, :]  = read_vol(case_idx, 'T1_IR', data_dir, phase="testing")
        FLAIR_vols[case_idx-1, :, :, :] = read_vol(case_idx, 'FLAIR', data_dir, phase="testing")
    T1_mean = T1_vols.mean()
    T1_std = T1_vols.std()
    T1_vols = (T1_vols - T1_mean) / T1_std
    T1IR_mean = T1IR_vols.mean()
    T1IR_std = T1IR_vols.std()
    T1IR_vols = (T1IR_vols - T1IR_mean) / T1IR_std
    FLAIR_mean = FLAIR_vols.mean()
    FLAIR_std = FLAIR_vols.std()
    FLAIR_vols = (FLAIR_vols - FLAIR_mean) / FLAIR_std
    for i in range(T1_vols.shape[0]):
        T1_vols[i] = ((T1_vols[i] - np.min(T1_vols[i])) / (np.max(T1_vols[i])-np.min(T1_vols[i])))*255
    for i in range(T1IR_vols.shape[0]):
        T1IR_vols[i] = ((T1IR_vols[i] - np.min(T1IR_vols[i])) / (np.max(T1IR_vols[i])-np.min(T1IR_vols[i])))*255
    for i in range(FLAIR_vols.shape[0]):
        FLAIR_vols[i] = ((FLAIR_vols[i] - np.min(FLAIR_vols[i])) / (np.max(FLAIR_vols[i])-np.min(FLAIR_vols[i])))*255
    T1_vols = T1_vols/127.5 -1.
    T1IR_vols = T1IR_vols/127.5 -1.
    FLAIR_vols = FLAIR_vols/127.5 -1.
    x = get_patches_unlab(T1_vols, T1IR_vols, FLAIR_vols, extraction_step, patch_shape, data_dir)
    print("Total Extracted Unlabeled Patches Shape:", x.shape)
    return np.rollaxis(x, 4, 1)