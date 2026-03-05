import numpy as np
import pandas as pd

def compute_kx_ky(calib_dict):
    """
    Given a calibration dictionary, compute kx and ky (in units of [px/mm]).
    
    kx -> Number of pixels per millimeter in x direction (ie width)
    ky -> Number of pixels per millimeter in y direction (ie height)
    """
    
    # Get image dimensions in pixels
    img_width_px = calib_dict['width']
    img_height_px = calib_dict['height']
    
    # Get physical sensor dimensions in millimeters
    sensor_width_mm = calib_dict['aperture_w']
    sensor_height_mm = calib_dict['aperture_h']

    # --- Check for division by zero ---
    if sensor_width_mm == 0 or sensor_height_mm == 0:
        print("Error: Sensor dimensions (aperture_w, aperture_h) cannot be zero.")
        return -1, -1

    # Compute kx and ky
    # kx = (Image Width in Pixels) / (Sensor Width in mm)
    kx = img_width_px / sensor_width_mm
    
    # ky = (Image Height in Pixels) / (Sensor Height in mm)
    ky = img_height_px / sensor_height_mm
    
    return kx, ky


def estimate_f_b(calib_dict, calib_points, n_points=None):
    """
    Estimate focal length f [mm] and baseline b [mm] from calibration points.
    
    This function uses a simultaneous least-squares method to solve for
    f_x = f * kx (focal length in pixels) and
    product = f_x * b (focal length * baseline)
    
    Parameters
    ----------
        calib_dict: dict
            Calibration dictionary. MUST contain 'o_x' (principal point) 
            and the pre-computed 'kx' (pixel density).
            
        calib_points: pd.DataFrame
            DataFrame with calibration points. This function uses the
            exact column names from the project description.
            
        n_points: int
            Number of points from the DataFrame to use for estimation.
            If None, all points are used.
        
    Returns
    -------
        f: float
            Estimated focal lenght [mm]
        b: float
            Estimated baseline [mm]
    """
    
    # Choose subset of points
    if n_points is not None:
        calib_points = calib_points.head(n_points)
    else: 
        n_points = len(calib_points) # Use all points

    if n_points == 0:
        print("Error: No calibration points selected.")
        return -1, -1

    
    try:
        kx = calib_dict['kx'] # Pixel density [px/mm]
        cx = calib_dict['o_x'] # Principal point x-coord [px]
    except KeyError as e:
        print(f"Error: Missing key {e} in calib_dict.")
        print("Please ensure you have run compute_kx_ky() and stored 'kx' in the dict.")
        return -1, -1


    A = np.zeros((2 * n_points, 2))
    B = np.zeros((2 * n_points, 1))

 
    for i in range(n_points):
        point = calib_points.iloc[i]
        
        # Get data using the exact column names
        X = point['X [mm]']
        Z = point['Z [mm]']
        ul = point['ul [px]']
        ur = point['ur [px]']
        
        # Pixel coordinates relative to the principal point
        ul_prime = ul - cx
        ur_prime = ur - cx
        
        # Left camera 
        # (ul - cx) * Z = f_x * X
        row_l = i * 2
        A[row_l, 0] = X    # Coefficient for f_x
        A[row_l, 1] = 0    # Coefficient for (f_x * b)
        B[row_l] = ul_prime * Z
        
        # Right camera equation 
        # (ur - cx) * Z = f_x * X - (f_x * b)
        row_r = i * 2 + 1
        A[row_r, 0] = X    # Coefficient for f_x
        A[row_r, 1] = -1   # Coefficient for (f_x * b)
        B[row_r] = ur_prime * Z

    # Solve 
    u_sol, _, _, _ = np.linalg.lstsq(A, B, rcond=None)
    
    f_x_est = u_sol[0][0]     # Estimated f_x (focal length in pixels)
    product_est = u_sol[1][0] # Estimated (f_x * b)

    if f_x_est == 0:
        print("Error: Estimated focal length (f_x) is zero.")
        return -1, -1

    
    # f [mm] = f_x [px] / kx [px/mm]
    f = f_x_est / kx
    
    # b [mm] = (f_x * b) / f_x
    b = product_est / f_x_est
    
    return f, b