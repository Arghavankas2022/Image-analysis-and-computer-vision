import numpy as np


def extract_patches(img, p):
    """
    Extract all patches of size <p> from image.

    Returns:
        patches ... Array of shape H x W x C*p**2, where last dimension holds flattened patches
    """

    H, W, C = img.shape
    pad = p // 2

    # Pad the image to handle border patches
    img_padded = np.pad(img, ((pad, pad), (pad, pad), (0, 0)), mode='reflect')

    # Collect patches by sliding window
    patches = np.zeros((H, W, C * p * p), dtype=img.dtype)
    idx = 0
    for dy in range(-pad, pad + 1):
        for dx in range(-pad, pad + 1):
            shifted = img_padded[pad + dy : pad + dy + H, pad + dx : pad + dx + W, :]
            patches[:, :, idx*C:(idx+1)*C] = shifted
            idx += 1
    return patches


def check_patch_extraction(extract_patches_fn):
    """ This function checks, whether patch extraction is implemented correctly
        <extract_patches_fn> is a callable function
    """
    
    # Create dummy image for debugging
    dbg_img = np.arange(1,21,1).reshape(4, 5, 1)
    
    print(f"Dummy image of shape 4 x 5 x 1")
    print(dbg_img[:, :, 0])
    print()
    
    # Extract 3x3 patches using the student's implementation
    dbg_patches = extract_patches_fn(dbg_img, p=3)
    
    # Some "ground truth" patches
    p11_true = np.array(
        [
            [ 1.,  2.,  3.],
            [ 6.,  7.,  8.],
            [11., 12., 13.]
        ]
    )
    
    p14_true = np.array(
        [
            [ 4.,  5.,  1.],
            [ 9., 10.,  6.],
            [14., 15., 11.]
        ]
    )
    
    p22_true = np.array(
        [
            [ 7.,  8.,  9.],
            [12., 13., 14.],
            [17., 18., 19.]
        ]
    )
    
    p32_true = np.array(
        [
            [12., 13., 14.],
            [17., 18., 19.],
            [ 2.,  3.,  4.]
        ]
    )
    
    # Check some extracted patches
    p11 = dbg_patches[1, 1].reshape(3, 3)
    p14 = dbg_patches[1, 4].reshape(3, 3)
    p22 = dbg_patches[2, 2].reshape(3, 3)
    p32 = dbg_patches[3, 2].reshape(3, 3)
    
    if not np.all(p11 == p11_true):
        print(
            f"3x3 Patch extraction failed at location [1, 1].",
            f"\nExpected:\n {p11_true}",
            f"\nReceived:\n {p11}"
        )
        return

    if not np.all(p14 == p14_true):
        print(
            f"3x3 Patch extraction failed at location [1, 4].",
            f"\nExpected:\n {p14_true}",
            f"\nReceived:\n {p14}"
        )
        return
    
    if not np.all(p22 == p22_true):
        print(
            f"3x3 Patch extraction failed at location [2, 2].",
            f"\nExpected:\n {p22_true}",
            f"\nReceived:\n {p22}"
        )
        return
    
    if not np.all(p32 == p32_true):
        print(
            f"3x3 Patch extraction failed at location [3, 2].",
            f"\nExpected:\n {p32_true}",
            f"\nReceived:\n {p32}"
        )
        return

    # Same test but for a 4x4 neighborhood
    dbg_patches = extract_patches_fn(dbg_img, p=4)
    
    p22 = dbg_patches[2, 2].reshape(4, 4)
    p23 = dbg_patches[2, 3].reshape(4, 4)

    print(p23)
  
    # Some "ground truth" patches
    p22_true = np.array(
        [
            [ 1.,  2.,  3., 4.],
            [ 6.,  7.,  8., 9.],
            [11., 12., 13., 14.],
            [16., 17., 18., 19.]
        ]
    )
    
    p23_true = np.array(
        [
            [2.,  3., 4., 5.],
            [7.,  8., 9., 10.],
            [12., 13., 14., 15.],
            [17., 18., 19., 20.]
        ]
    )
    
    if not np.all(p22 == p22_true):
        print(
            f"4x4 Patch extraction failed at location [2, 2].",
            f"\nExpected:\n {p22_true}",
            f"\nReceived:\n {p22}"
        )
        return

    if not np.all(p23 == p23_true):
        print(
            f"4x4 Patch extraction failed at location [2, 3].",
            f"\nExpected:\n {p23_true}",
            f"\nReceived:\n {p23}"
        )
        return
    
    print("Test completed successfully :)")
