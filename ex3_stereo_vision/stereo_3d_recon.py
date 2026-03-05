import numpy as np
from calibration import compute_kx_ky, estimate_f_b
from extract_patches import extract_patches

def triangulate(u_left, u_right, v, calib_dict):
    kx = calib_dict['kx']
    ky = calib_dict['ky']
    f = calib_dict['f']
    b = calib_dict['b']
    o_x = calib_dict['o_x']
    o_y = calib_dict['o_y']

    ul_p = u_left - o_x
    ur_p = u_right - o_x
    v_p = v - o_y

    d = ul_p - ur_p
    d[d == 0] = 1e-9

    Z = f * kx * b / d
    X = (ul_p * Z) / (f * kx)
    Y = (v_p * Z) / (f * ky)

    return np.stack([X, Y, Z], axis=-1)

def compute_ncc(img_l, img_r, p):
    if img_l.ndim == 2:
        img_l = img_l[..., np.newaxis]
    if img_r.ndim == 2:
        img_r = img_r[..., np.newaxis]
    
    assert img_l.shape == img_r.shape, "Shape mismatch."
    H, W, C = img_l.shape

    p_size = 2*p + 1
    N = C * p_size**2

    patches_l = extract_patches(img_l, p_size)
    patches_r = extract_patches(img_r, p_size)

    mean_l = np.mean(patches_l, axis=-1, keepdims=True)
    mean_r = np.mean(patches_r, axis=-1, keepdims=True)
    std_l = np.std(patches_l, axis=-1, keepdims=True) + 1e-8
    std_r = np.std(patches_r, axis=-1, keepdims=True) + 1e-8

    patches_l = (patches_l - mean_l) / std_l
    patches_r = (patches_r - mean_r) / std_r

    # Compute correlation: for each pixel in left image, correlate with all pixels in right image
    # patches_l shape: (H, W, patch_dim)
    # patches_r shape: (H, W, patch_dim)
    # We want: corr[i, j_left, j_right] = correlation between left[i,j_left] and right[i,j_right]
    
    H_full, W_full = patches_l.shape[0], patches_l.shape[1]
    
    # Use matmul on the cropped valid region
    patches_l_crop = patches_l[p:H-p, p:W-p]  # (H_crop, W_crop, patch_dim)
    patches_r_crop = patches_r[p:H-p, p:W-p]  # (H_crop, W_crop, patch_dim)
    
    # Compute correlation between all left and right patches
    corr = np.matmul(patches_l_crop, patches_r_crop.transpose(0, 2, 1)) / N
    
    return corr

class Stereo3dReconstructor:
    def __init__(self, p=3, w_mode='peak_ratio', subpixel=True, uniqueness_ratio=0.85):
        """
        Args:
            p       ... Patch size for NCC computation (reduced default for speed)
            w_mode  ... Weighting mode. I.e. method to compute certainty scores
                        Options: 'none', 'peak_ratio', 'variance'
            subpixel ... Whether to use subpixel refinement
            uniqueness_ratio ... Ratio threshold for uniqueness constraint (best/second_best)
        """
        self.p = p
        self.w_mode = w_mode
        self.subpixel = subpixel
        self.uniqueness_ratio = uniqueness_ratio

    def fill_calib_dict(self, calib_dict, calib_points):
        """ Fill missing entries in calib dict - nothing to do here """
        calib_dict['kx'], calib_dict['ky'] = compute_kx_ky(calib_dict)
        calib_dict['f'], calib_dict['b'] = estimate_f_b(calib_dict, calib_points)
        
        return calib_dict

    def compute_certainty(self, C_masked, best_c_r_indices):
        if self.w_mode == 'peak_ratio':
            # Compare best vs second-best match
            C_sorted = np.sort(C_masked, axis=2)
            best = C_sorted[:, :, -1]
            second_best = C_sorted[:, :, -2]
            
            # Make sure we have valid values
            best = np.where(np.isfinite(best), best, 0.0)
            second_best = np.where(np.isfinite(second_best), second_best, 0.0)
            
            # Peak ratio: high when best is much better than second-best
            certainty = (best - second_best) / (np.abs(best) + 1e-8)
            certainty = np.clip(certainty, 0, 1)
            
        elif self.w_mode == 'variance':
            # Higher variance = more distinctive peak
            # Replace -inf with 0 for variance calculation
            C_valid = np.where(np.isfinite(C_masked), C_masked, 0.0)
            variance = np.var(C_valid, axis=2)
            certainty = np.tanh(variance * 10)  # Scale and normalize
            certainty = np.clip(certainty, 0, 1)
            
        else:  # 'none' or default
            # Simple normalized NCC score
            best_ncc_scores = np.max(C_masked, axis=2)
            best_ncc_scores = np.where(np.isfinite(best_ncc_scores), best_ncc_scores, 0.0)
            certainty = (best_ncc_scores + 1.0) / 2.0
            certainty = np.clip(certainty, 0, 1)
        
        return certainty

    def apply_uniqueness_constraint(self, C_masked, best_c_r_indices):
        # Use partition instead of full sort for speed
        # Filter out -inf values first
        C_valid = np.where(np.isfinite(C_masked), C_masked, -1.0)
        C_top2 = np.partition(C_valid, -2, axis=2)[:, :, -2:]
        best = C_top2[:, :, 1]
        second_best = C_top2[:, :, 0]
        
        # Uniqueness mask: best should be significantly better than second best
        ratio = best / (second_best + 1e-8)
        uniqueness_mask = ratio > (1.0 / self.uniqueness_ratio)
        
        return uniqueness_mask

    def subpixel_refinement(self, C_masked, best_indices, disparity):
        """Vectorized subpixel refinement - much faster than loop-based version"""
        H, W, D = C_masked.shape
        refined_disparity = disparity.astype(float)
        
        # Create masks for valid indices (not at boundaries)
        valid_mask = (best_indices > 0) & (best_indices < D - 1)
        
        # Get coordinates of valid pixels
        i_coords, j_coords = np.where(valid_mask)
        
        if len(i_coords) == 0:
            return refined_disparity
            
        idx = best_indices[valid_mask]
        
        # Get values at best, prev, and next indices for all valid pixels
        c_prev = C_masked[i_coords, j_coords, idx - 1]
        c_curr = C_masked[i_coords, j_coords, idx]
        c_next = C_masked[i_coords, j_coords, idx + 1]
        
        # Parabolic interpolation
        denom = 2 * (c_prev - 2*c_curr + c_next)
        valid_denom = np.abs(denom) > 1e-6
        
        delta = np.zeros_like(denom)
        if np.any(valid_denom):
            delta[valid_denom] = (c_prev[valid_denom] - c_next[valid_denom]) / denom[valid_denom]
            delta = np.clip(delta, -0.5, 0.5)
        
        # Apply refinement only to valid pixels
        refined_disparity[i_coords, j_coords] += delta
        
        return refined_disparity

    def recon_scene_3d(self, img_l, img_r, calib_dict):
        assert img_l.ndim == 3, f"Expected 3 dimensional input. Got {img_l.shape}"
        assert img_l.shape == img_r.shape, "Shape mismatch."
        
        H, W, C = img_l.shape
        p = self.p 

        print("Computing NCC volume...")
        C = compute_ncc(img_l, img_r, p)
        H_crop, W_crop = C.shape[0], C.shape[1]

        print("Applying physical constraints...")
        j = np.arange(W_crop)
        mask = j[None, None, :] > j[None, :, None]
        C_masked = np.where(mask, -np.inf, C)

        print("Finding correspondences...")
        best_c_r_indices = np.argmax(C_masked, axis=2)
        
        c_l_map = np.arange(W_crop)[None, :]
        disparity_cropped = c_l_map - best_c_r_indices

        if self.uniqueness_ratio < 1.0:
            print("Applying uniqueness constraint...")
            uniqueness_mask = self.apply_uniqueness_constraint(C_masked, best_c_r_indices)
        else:
            uniqueness_mask = np.ones((H_crop, W_crop), dtype=bool)

        if self.subpixel:
            print("Applying subpixel refinement...")
            disparity_cropped = self.subpixel_refinement(C_masked, best_c_r_indices, disparity_cropped)

        print("Computing certainty...")
        certainty_cropped = self.compute_certainty(C_masked, best_c_r_indices)
        
        # Apply uniqueness mask to certainty
        certainty_cropped = certainty_cropped * uniqueness_mask
   
        print("Triangulating...")
        
        # Create 2D coordinate maps for the *left* image,
        # corresponding to the cropped (valid) region.
        c_l_coords = np.arange(p, W - p)
        r_l_coords = np.arange(p, H - p)
        
        # 'xx' will be u_left, 'yy' will be v
        xx, yy = np.meshgrid(c_l_coords, r_l_coords)
       
        u_right_cropped = xx - disparity_cropped
        
        # Reconstruct 3D points
        XYZ_cropped = triangulate(
            xx,                 # u_left
            u_right_cropped,    # u_right
            yy,                 # v
            calib_dict
        )

        # Create the final (H, W, 4) output array
        output = np.zeros((H, W, 4))
        
        # Fill the valid center region 
        output[p:H-p, p:W-p, 0:3] = XYZ_cropped
        output[p:H-p, p:W-p, 3] = certainty_cropped
        
        print("Reconstruction complete.")
        
        return output