import os
import nibabel as nib
import numpy as np
import shutil
import json

from nipype.interfaces.ants import N4BiasFieldCorrection

# Load configuration
with open("/content/drive/MyDrive/Colab Notebooks/Bakalauras/bazinis_GBT/pytorch/configs/fmgan.json", "r") as f:
    config = json.load(f)

phase = config["phase"]
num_mod = config["num_modalities"]  # 2 (T1, T2)
volume_shape = [240, 240, 48]  # Updated to match data shape

# File path creation based on your folder structure
def get_mrbrains_filename(case_idx, modality, data_dir, phase):
    """
    Create file path based on your folder structure.
    
    Args:
        case_idx (int): Subject index (e.g., 1, 11)
        modality (str): Modality (e.g., T1, T2, label)
        data_dir (str): Root data directory (dataverse_files)
        phase (str): Phase (Training, Testing, Testing (for validation))
    
    Returns:
        str: Path to the .nii file
    """
    if phase.lower() == "training":
        folder = os.path.join(data_dir, "Training")
    elif phase.lower() == "testing":
        folder = os.path.join(data_dir, "Testing")
    elif phase.lower() == "validation":
        folder = os.path.join(data_dir, "Testing (for validation)")
    else:
        raise ValueError(f"Unknown phase: {phase}")

    if modality.lower() in ['labels', 'label']:
        subfolder = "label"
        filename = f"subject-{case_idx}-label.nii"
    else:
        subfolder = modality
        filename = f"subject-{case_idx}-{modality}.nii"
    
    return os.path.join(folder, subfolder, filename)

def read_data(case_idx, modality, data_dir, phase):
    image_path = get_mrbrains_filename(case_idx, modality, data_dir, phase)
    print("Loading:", image_path)
    if not os.path.exists(image_path):
        raise FileNotFoundError(f"File not found: {image_path}")
    return nib.load(image_path), image_path

def read_vol(case_idx, modality, data_dir, phase):
    image_data, image_path = read_data(case_idx, modality, data_dir, phase)
    data = image_data.get_fdata(dtype=np.float32)
    # Verify volume shape matches expected
    if data.shape != tuple(volume_shape):
        raise ValueError(f"Expected volume shape {volume_shape}, got {data.shape} for {image_path}")
    return data

# Helper functions: bias correction, normalization, patch extraction
def correct_bias(in_file, out_file):
    correct = N4BiasFieldCorrection()
    correct.inputs.input_image = in_file
    correct.inputs.output_image = out_file
    done = correct.run()
    return done.outputs.output_image

def normalise(case_idx, modality, in_dir, out_dir, copy=False):
    image_in_path = get_mrbrains_filename(case_idx, modality, in_dir)
    image_out_path = get_mrbrains_filename(case_idx, modality, out_dir)
    os.makedirs(os.path.dirname(image_out_path), exist_ok=True)
    if copy:
        shutil.copy(image_in_path, image_out_path)
    else:
        correct_bias(image_in_path, image_out_path)
    print(image_in_path + " done.")

def extract_patches(volume, patch_shape, extraction_step, datype='float32'):
    patch_h, patch_w, patch_d = patch_shape
    stride_h, stride_w, stride_d = extraction_step
    img_h, img_w, img_d = volume.shape
    # Verify volume shape
    if (img_h, img_w, img_d) != tuple(volume_shape):
        raise ValueError(f"Volume shape {volume.shape} does not match expected {volume_shape}")
    # Check if patch extraction is feasible
    if img_h < patch_h or img_w < patch_w or img_d < patch_d:
        raise ValueError(f"Patch shape {patch_shape} too large for volume {volume.shape}")
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
    assert k == N_patches_img, f"Expected {N_patches_img} patches, got {k}"
    return raw_patch_matrix

# Patch extraction for labeled data
def remap_labels(labels):
    """
    Remap label values:
    - Values 1, 2, 3 remain unchanged
    - All other values become 0 (background)
    """
    return np.where(np.isin(labels, [1, 2, 3]), labels, 0)

def get_patches_lab(T1_vols, T2_vols, label_vols, extraction_step, patch_shape, 
                    validating, testing, num_images_training):
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
        T2_train = extract_patches(T2_vols[idx], patch_shape, extraction_step, datype="float32")
        x[y_length:, :, :, :, 1] = T2_train[valid_idxs]
        
    y = remap_labels(y)
    return x, y

# Preprocess labeled data (training/validation/testing)
def preprocess_dynamic_lab(data_dir, seed, num_classes, extraction_step, patch_shape, 
                          num_images_training, num_images_validating, num_images_testing, 
                          validating=False, testing=False, volume_shape=volume_shape):
    
    if testing:
        phase = "testing"
        cases = [11, 12, 13, 14]  # Based on Testing/T1 subjects
        n_vols = len(cases)
        print(f"Testing with {n_vols} images: {cases}")
        
    elif validating:
        phase = "validation"
        cases = [5, 6]  # Based on config num_images_validating=2
        n_vols = len(cases)
        print(f"Validating with {n_vols} images: {cases}")

    else:  # training
        phase = "training"
        cases = [1, 2, 3, 4]  # Based on Training/T1 subjects
        n_vols = len(cases)
        print(f"Training with {n_vols} images: {cases}")

    # Initialize arrays for volumes
    T1_vols = np.empty((n_vols, *volume_shape), dtype="float32")
    T2_vols = np.empty((n_vols, *volume_shape), dtype="float32")
    label_vols = np.empty((n_vols, *volume_shape), dtype="uint8")
        
    # Load the data
    for i, case_idx in enumerate(cases):
        print(f"Processing subject index: {case_idx}")
        T1_vols[i] = read_vol(case_idx, 'T1', data_dir, phase)
        T2_vols[i] = read_vol(case_idx, 'T2', data_dir, phase)
        label_vols[i] = read_vol(case_idx, 'label', data_dir, phase)

    # Normalize each modality separately
    T1_vols = (T1_vols - np.mean(T1_vols)) / np.std(T1_vols)
    T2_vols = (T2_vols - np.mean(T2_vols)) / np.std(T2_vols)

    # Rescale to [0,255] and then [-1,1]
    for vols in [T1_vols, T2_vols]:
        for i in range(vols.shape[0]):
            min_val, max_val = np.min(vols[i]), np.max(vols[i])
            vols[i] = ((vols[i] - min_val) / (max_val - min_val)) * 255
            vols[i] = vols[i] / 127.5 - 1.0

    x, y = get_patches_lab(
        T1_vols, T2_vols, label_vols,
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

# Functions for unlabeled data
def get_patches_unlab(T1_vols, T2_vols, extraction_step, patch_shape, data_dir):
    patch_shape_1d = patch_shape[0]
    x = np.zeros((0, patch_shape_1d, patch_shape_1d, patch_shape_1d, num_mod), dtype="float32")
    label_ref = read_vol(1, 'label', data_dir, phase="training")  # Reference from training
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
        T2_train = extract_patches(T2_vols[idx], patch_shape, extraction_step, datype="float32")
        x[x_length:, :, :, :, 1] = T2_train[valid_idxs]
    return x

def preprocess_dynamic_unlab(data_dir, extraction_step, patch_shape, num_images_training_unlab, 
                            volume_shape=volume_shape):
    T1_vols = np.empty((num_images_training_unlab, *volume_shape), dtype="float32")
    T2_vols = np.empty((num_images_training_unlab, *volume_shape), dtype="float32")
    cases = [11, 12, 13, 14]  # Based on Testing/T1 subjects
    for i, case_idx in enumerate(cases[:num_images_training_unlab]):
        T1_vols[i] = read_vol(case_idx, 'T1', data_dir, phase="testing")
        T2_vols[i] = read_vol(case_idx, 'T2', data_dir, phase="testing")
    
    T1_mean = T1_vols.mean()
    T1_std = T1_vols.std()
    T1_vols = (T1_vols - T1_mean) / T1_std
    T2_mean = T2_vols.mean()
    T2_std = T2_vols.std()
    T2_vols = (T2_vols - T2_mean) / T2_std
    
    for i in range(T1_vols.shape[0]):
        T1_vols[i] = ((T1_vols[i] - np.min(T1_vols[i])) / (np.max(T1_vols[i])-np.min(T1_vols[i])))*255
        T1_vols[i] = T1_vols[i] / 127.5 - 1.
    for i in range(T2_vols.shape[0]):
        T2_vols[i] = ((T2_vols[i] - np.min(T2_vols[i])) / (np.max(T2_vols[i])-np.min(T2_vols[i])))*255
        T2_vols[i] = T2_vols[i] / 127.5 - 1.
    
    x = get_patches_unlab(T1_vols, T2_vols, extraction_step, patch_shape, data_dir)
    print("Total Extracted Unlabeled Patches Shape:", x.shape)
    return np.rollaxis(x, 4, 1)