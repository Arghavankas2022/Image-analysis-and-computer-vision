import numpy as np

class ImageCompressor:
    def __init__(self, n_components=18):  # Increased from 20
        self.n_components = n_components
        self.dtype = np.float16  # Keep float16 for size constraints
        self.mean_image = None
        self.components = None

    def get_codebook(self):
        if self.mean_image is None or self.components is None:
            return np.array([]).astype(self.dtype)
        # Codebook = mean + PCA components stacked
        codebook = np.vstack([self.mean_image.flatten(), self.components])
        return codebook.astype(self.dtype)

    def train(self, train_images):
        X = np.array([img.flatten() for img in train_images], dtype=np.float32)
        self.mean_image = X.mean(axis=0)
        X_centered = X - self.mean_image
        # Compute PCA using SVD
        U, S, Vt = np.linalg.svd(X_centered, full_matrices=False)
        self.components = Vt[:self.n_components]

    def compress(self, test_image):
        x_flat = test_image.flatten().astype(np.float32)
        z = (x_flat - self.mean_image) @ self.components.T
        return z.astype(self.dtype)


class ImageReconstructor:
    def __init__(self, codebook):
        self.codebook = codebook
        self.mean_image = codebook[0]
        self.components = codebook[1:]

    def reconstruct(self, test_code):
        x_flat = test_code.astype(np.float32) @ self.components.astype(np.float32) + self.mean_image.astype(np.float32)
        x_flat = np.clip(x_flat, 0, 255)
        HWC = (96, 96, 3)
        x_pred = x_flat.reshape(HWC).astype(np.uint8)
        return x_pred