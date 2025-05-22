# utils/recompose.py
import numpy as np

def recompose3D_overlap(preds, img_h, img_w, img_d, stride_h, stride_w, stride_d):
    patch_h = preds.shape[1]
    patch_w = preds.shape[2]
    patch_d = preds.shape[3]
    N_patches_h = (img_h - patch_h) // stride_h + 1
    N_patches_w = (img_w - patch_w) // stride_w + 1
    N_patches_d = (img_d - patch_d) // stride_d + 1
    N_patches_img = N_patches_h * N_patches_w * N_patches_d
    print("N_patches_h:", N_patches_h)
    print("N_patches_w:", N_patches_w)
    print("N_patches_d:", N_patches_d)
    print("N_patches_img:", N_patches_img)

    # Check and pad if necessary
    num_patches = preds.shape[0]
    if num_patches % N_patches_img != 0:
        padding_size = N_patches_img - (num_patches % N_patches_img)
        print(f"Warning: Number of patches ({num_patches}) is not divisible by expected patches per image ({N_patches_img}). Padding with {padding_size} zero patches.")
        preds = np.pad(preds, ((0, padding_size), (0, 0), (0, 0), (0, 0)), mode='constant', constant_values=0)
        num_patches = preds.shape[0]

    N_full_imgs = num_patches // N_patches_img
    print("According to the dimension inserted, there are "
          + str(N_full_imgs) + " full images (of " + str(img_h) + "x" + str(img_w) + "x" + str(img_d) + " each)")

    # Initialize arrays
    raw_pred_martrix = np.zeros((N_full_imgs, img_h, img_w, img_d))
    raw_sum = np.zeros((N_full_imgs, img_h, img_w, img_d))
    final_matrix = np.zeros((N_full_imgs, img_h, img_w, img_d), dtype='uint16')

    k = 0
    # Iterator over all patches
    for i in range(N_full_imgs):
        for h in range(N_patches_h):
            for w in range(N_patches_w):
                for d in range(N_patches_d):
                    raw_pred_martrix[i,
                                     h * stride_h:(h * stride_h) + patch_h,
                                     w * stride_w:(w * stride_w) + patch_w,
                                     d * stride_d:(d * stride_d) + patch_d] += preds[k]
                    raw_sum[i,
                            h * stride_h:(h * stride_h) + patch_h,
                            w * stride_w:(w * stride_w) + patch_w,
                            d * stride_d:(d * stride_d) + patch_d] += 1.0
                    k += 1
    assert k == preds.shape[0], f"Processed {k} patches, expected {preds.shape[0]}"
    assert np.min(raw_sum) >= 1.0, "Some regions have zero sum"
    
    # Average overlapping regions
    final_matrix = np.around(raw_pred_martrix / raw_sum)
    return final_matrix