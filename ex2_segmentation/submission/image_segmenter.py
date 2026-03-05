import numpy as np

from kmeans import (
    compute_distance,
    kmeans_fit,
    kmeans_predict_idx,
    kNN,
)

from extract_patches import extract_patches


class ImageSegmenter:
    def __init__(self, mode='kmeans',k_fg=5, k_bg=11):
        """ Feel free to add any hyper-parameters to the ImageSegmenter.
            
            But note:
            For the final submission the default hyper-parameteres will be used.
            In particular the segmetation will likely crash, if no defaults are set.
        """
        self.mode= mode
        self.k_fg = k_fg
        self.k_bg = k_bg

        # During evaluation, this will be replaced by a generator with different
        # random seeds. Use this generator, whenever you require random numbers,
        # otherwise your score might be lower due to stochasticity
        self.rng = np.random.default_rng(42)
        
    def extract_features_(self, sample_dd):
      img = sample_dd['img'].astype(np.float32)
      H, W, C = img.shape

    # Normalize color
      color_features = img.reshape(-1, C) / 255.0

    # Extract patch features
      patches = extract_patches(img, p=3).reshape(H, W, -1) / 255.0
      patch_mean = patches.mean(axis=2, keepdims=True)  # local average
      patch_var = patches.var(axis=2, keepdims=True)    # local variance

      texture_features = np.concatenate((patch_mean, patch_var), axis=2).reshape(-1, 2)

    # Add spatial coordinates
      yy, xx = np.indices((H, W))
      coords = np.stack((yy / H, xx / W), axis=-1).reshape(-1, 2)

    # Combine features
      spatial_weight = 0.7
      texture_weight = 0.5

      features = np.concatenate((
          color_features,
          texture_weight * texture_features,
          spatial_weight * coords,
      ), axis=1)

      return features

    
    def segment_image_dummy(self, sample_dd):
        return sample_dd['scribble_fg']

    def segment_image_kmeans(self, sample_dd):
        """ Segment images using k means """
        H, W, C = sample_dd['img'].shape
        features = self.extract_features_(sample_dd)
        
        #
        fg = sample_dd['scribble_fg'].flatten()>0
        bg = sample_dd['scribble_bg'].flatten()>0
        
        fg_features =features[fg]
        bg_features = features[bg]

        centroid_fg = kmeans_fit(fg_features, self.k_fg, rng = self.rng)
        centroid_bg = kmeans_fit(bg_features, self.k_bg, rng = self.rng)

        centroids = np.concatenate((centroid_fg,centroid_bg),axis=0)
        centroids_labels = np.concatenate((np.zeros(self.k_fg),np.ones(self.k_bg)),axis=0)

        predictions = kNN(centroids,centroids_labels,features,k=6)
        mask_pred = predictions ==0

        # For now return scribble
        return mask_pred.reshape(H,W)

    def segment_image(self, sample_dd):
        """ Feel free to add other methods """
        if self.mode == 'dummy':
            return self.segment_image_dummy(sample_dd)
        
        elif self.mode == 'kmeans':
            return self.segment_image_kmeans(sample_dd)
        
        else:
            raise ValueError(f"Unknown mode: {self.mode}")
