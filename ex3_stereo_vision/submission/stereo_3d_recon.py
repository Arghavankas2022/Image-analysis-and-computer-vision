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

    patches_l = extract_patches(img_l, 2*p+1)
    patches_r = extract_patches(img_r, 2*p+1)

    mean_l = np.mean(patches_l, axis=-1, keepdims=True)
    mean_r = np.mean(patches_r, axis=-1, keepdims=True)
    std_l = np.std(patches_l, axis=-1, keepdims=True) + 1e-8
    std_r = np.std(patches_r, axis=-1, keepdims=True) + 1e-8

    patches_l = (patches_l - mean_l) / std_l
    patches_r = (patches_r - mean_r) / std_r

    corr = np.matmul(patches_l, patches_r.transpose(0, 2, 1))/N
    return corr[p:H-p, p:W-p, p:W-p]

class Stereo3dReconstructor:
    def __init__(self, p=5, w_mode='none', subpixel=False, uniqueness_ratio=0.8):
        """
        Feel free to add hyper parameters here, but be sure to set defaults
        
        Args:
            p       ... Patch size for NCC computation
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
            # Peak ratio: high when best is much better than second-best
            certainty = (best - second_best) / (np.abs(best) + 1e-8)
            certainty = np.clip(certainty, 0, 1)
            
        elif self.w_mode == 'variance':
            # Higher variance = more distinctive peak
            variance = np.var(C_masked, axis=2)
            certainty = np.tanh(variance * 10)  # Scale and normalize
            certainty = np.clip(certainty, 0, 1)
            
        else:  # 'none' or default
            # Simple normalized NCC score
            best_ncc_scores = np.max(C_masked, axis=2)
            certainty = (best_ncc_scores + 1.0) / 2.0
        
        return certainty

    def apply_uniqueness_constraint(self, C_masked, best_c_r_indices):

        C_sorted = np.sort(C_masked, axis=2)
        best = C_sorted[:, :, -1]
        second_best = C_sorted[:, :, -2]
        
        # Uniqueness mask: best should be significantly better than second best
        uniqueness_mask = (best / (second_best + 1e-8)) > (1.0 / self.uniqueness_ratio)
        
        return uniqueness_mask

    def subpixel_refinement(self, C_masked, best_indices, disparity):

        H, W, D = C_masked.shape
        refined_disparity = disparity.astype(float)
        
        for i in range(H):
            for j in range(W):
                idx = best_indices[i, j]
                if idx > 0 and idx < D - 1:
                    c_prev = C_masked[i, j, idx - 1]
                    c_curr = C_masked[i, j, idx]
                    c_next = C_masked[i, j, idx + 1]
                    
                    denom = 2 * (c_prev - 2*c_curr + c_next)
                    if abs(denom) > 1e-6:
                        delta = (c_prev - c_next) / denom
                        delta = np.clip(delta, -0.5, 0.5)
                        refined_disparity[i, j] += delta
        
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