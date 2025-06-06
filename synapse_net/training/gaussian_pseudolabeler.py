import torch
import numpy as np
from typing import Optional
from skimage.feature import peak_local_max
from scipy.optimize import curve_fit
import matplotlib.pyplot as plt

def gaussian_3d(coords, amp, x0, y0, z0, sx, sy, sz):
    """3D Gaussian function."""
    x, y, z = coords
    return amp * np.exp(
        -(((x - x0) ** 2) / (2 * sx ** 2)
          + ((y - y0) ** 2) / (2 * sy ** 2)
          + ((z - z0) ** 2) / (2 * sz ** 2))
    ).ravel()

class GaussianPseudoLabeler:
    def __init__(
        self,
        activation: Optional[torch.nn.Module] = None,
        confidence_threshold: float = 0.5,
        threshold_from_both_sides: bool = True,
        patch_size: int = 9,
        peak_threshold_abs: float = 0.2,
        peak_min_distance: int = 5,
        min_density_inside: float = 0.3,
        min_amp: float = 0.2,
        max_sigma: float = 1.5,
    ):
        self.activation = activation
        self.confidence_threshold = confidence_threshold
        self.threshold_from_both_sides = threshold_from_both_sides
        self.patch_size = patch_size
        self.peak_threshold_abs = peak_threshold_abs
        self.peak_min_distance = peak_min_distance
        self.min_density_inside = min_density_inside
        self.min_amp = min_amp
        self.max_sigma = max_sigma

    def _fit_single_gaussian(self, data, peak_coord):
        z, y, x = peak_coord
        half = self.patch_size // 2
        
        # Extract patch
        z_start, z_end = max(z - half, 0), min(z + half + 1, data.shape[0])
        y_start, y_end = max(y - half, 0), min(y + half + 1, data.shape[1])
        x_start, x_end = max(x - half, 0), min(x + half + 1, data.shape[2])
        patch = data[z_start:z_end, y_start:y_end, x_start:x_end]

        #print(f"Patch at {peak_coord}: shape {patch.shape}, max {patch.max():.3f}")
        
        #if patch.size < 27 or patch.max() < self.min_density_inside:
            #print("Patch skipped: too small or too flat.")
            #return None

        zz, yy, xx = np.mgrid[0:patch.shape[0], 0:patch.shape[1], 0:patch.shape[2]]
        coords = (xx, yy, zz)

        guess = (
            patch.max(),
            patch.shape[2] // 2,
            patch.shape[1] // 2,
            patch.shape[0] // 2,
            1.0, 1.0, 1.0
        )
        

        #bounds = ([0.1, 0, 0, 0, 0.3, 0.3, 0.3],
          #[10.0, 5, 5, 5, 1000, 1000, 1000])  # amplitude upper bound increased from 2.0 to 10.0
        
        #print("Initial guess:", guess)
        #print("Bounds:", bounds)
        try:
            popt, pcov = curve_fit(
                gaussian_3d, coords, patch.ravel(),
                p0=guess, maxfev=1000,
                #bounds=(bounds)
            )
            amp, x0, y0, z0, sx, sy, sz = popt
            #print(f"Fit succeeded at {peak_coord}: amp={amp:.3f}, sigmas=({sx:.2f},{sy:.2f},{sz:.2f})")
            
            if (amp < self.min_amp or 
                sx > self.max_sigma or 
                sy > self.max_sigma or 
                sz > self.max_sigma or
                np.any(np.diag(pcov) > 0.5)):
                print("Fit rejected: poor parameters or covariance.")
                return None

            return (z_start + z0, y_start + y0, x_start + x0, sz, sy, sx)
        except (RuntimeError, ValueError) as e:
            #print(f"Fit failed at {peak_coord}: {e}")
            return None

    def _compute_gaussian_mask(self, data: np.ndarray):
        if data.ndim == 4:
            masks = []
            for b in range(data.shape[0]):
                mask_b = self._compute_gaussian_mask(data[b])
                masks.append(mask_b)
            return torch.stack(masks, dim=0)

        data = data.astype(np.float32)
        shape = data.shape

        #print(f"Input shape: {shape}, peak_min_distance: {self.peak_min_distance}, threshold_abs: {self.peak_threshold_abs}")
        
        peaks = peak_local_max(
            data,
            min_distance=self.peak_min_distance,
            threshold_abs=self.peak_threshold_abs,
            exclude_border=False
        )
        
        #print(f"Detected {len(peaks)} raw peaks.")
        
        if len(peaks) > 0:
            peak_values = data[tuple(peaks.T)]
            mask = peak_values >= self.min_density_inside
            peaks = peaks[mask]
            #print(f"{len(peaks)} peaks after intensity filtering.")

        mask = np.zeros(shape, dtype=bool)
        for p in peaks:
            result = self._fit_single_gaussian(data, tuple(p))
            if result is None:
                continue
            zc, yc, xc, sz, sy, sx = result
            Z, Y, X = np.indices(shape)
            ellipsoid = (
                ((X - xc) ** 2 / (sx ** 2) + 
                 (Y - yc) ** 2 / (sy ** 2) + 
                 (Z - zc) ** 2 / (sz ** 2)) <= 2.5
            )
            mask = np.logical_or(mask, ellipsoid)

        #print(f"Final mask sum: {mask.sum()}")
        return torch.tensor(mask, dtype=torch.float32)

    def __call__(self, teacher: torch.nn.Module, input_: torch.Tensor):
        pseudo_labels = teacher(input_)
        if self.activation is not None:
            pseudo_labels = self.activation(pseudo_labels)

        if self.confidence_threshold is None:
            label_mask = None
        else:
            pseudo_np = pseudo_labels.squeeze().detach().cpu().numpy()
            gaussian_mask = self._compute_gaussian_mask(pseudo_np)
            if self.threshold_from_both_sides:
                background_mask = (pseudo_np <= (1 - self.confidence_threshold))
            else:
                background_mask = np.zeros_like(pseudo_np, dtype=bool)
            
            label_mask = gaussian_mask.bool() | torch.from_numpy(background_mask)
            label_mask = label_mask.to(pseudo_labels.device)
            #label_mask = gaussian_mask.bool()
            while label_mask.ndim < pseudo_labels.ndim:
                label_mask = label_mask.unsqueeze(0)
        
        return pseudo_labels, label_mask
    def visualize_binary_mask(self, label_mask: torch.Tensor, axis: str = 'z', n_slices: int = 5):
        """
        Visualize the binary mask where:
        - label_mask=1 (confident) → Solid green
        - label_mask=0 (non-confident) → Black
        No transparency or raw density shown.
        """
        mask_np = label_mask.squeeze().detach().cpu().numpy().astype(bool)
        D, H, W = mask_np.shape
        
        fig, axs = plt.subplots(1, n_slices, figsize=(15, 3))
        
        for i, idx in enumerate(np.linspace(0, D-1 if axis=='z' else H-1 if axis=='y' else W-1, n_slices, dtype=int)):
            # Get slice
            if axis == 'z':
                mask_slice = mask_np[idx, :, :]
            elif axis == 'y':
                mask_slice = mask_np[:, idx, :]
            else:
                mask_slice = mask_np[:, :, idx]
            
            # Plot binary mask (green=1, black=0)
            axs[i].imshow(mask_slice, cmap='Greens', vmin=0, vmax=1)
            axs[i].set_title(f'Mask slice {axis}={idx}')
        
        plt.tight_layout()
        plt.show()
    def visualize_masked_density_separate(self, pseudo_labels, label_mask, axis='z', n_slices=5):
        """Visualize confident regions (green=foreground, blue=background) on white background."""
        pseudo_np = pseudo_labels.squeeze().detach().cpu().numpy()
        mask_np = label_mask.squeeze().detach().cpu().numpy().astype(bool)
        
        # Get background mask separately
        if self.threshold_from_both_sides:
            background_mask = (pseudo_np <= (1 - self.confidence_threshold))
        else:
            background_mask = np.zeros_like(pseudo_np, dtype=bool)
        
        D, H, W = pseudo_np.shape
        fig, axs = plt.subplots(3, n_slices, figsize=(20, 12))
        
        for i, idx in enumerate(np.linspace(0, D-1 if axis=='z' else H-1 if axis=='y' else W-1, n_slices, dtype=int)):
            # Get slices
            if axis == 'z':
                slc = pseudo_np[idx,:,:]
                mask_slc = mask_np[idx,:,:]
                bg_slc = background_mask[idx,:,:]
            elif axis == 'y':
                slc = pseudo_np[:,idx,:]
                mask_slc = mask_np[:,idx,:]
                bg_slc = background_mask[:,idx,:]
            else:
                slc = pseudo_np[:,:,idx]
                mask_slc = mask_np[:,:,idx]
                bg_slc = background_mask[:,:,idx]
            
            # Row 1: Raw density
            axs[0,i].imshow(slc, cmap='viridis')
            axs[0,i].set_title(f'Density {axis}={idx}')
        
            # Row 2: Particles (green) + Background (blue) on WHITE
            # Create white background
            white_bg = np.ones_like(slc)
            axs[1,i].imshow(white_bg, cmap='gray', vmin=0, vmax=1)  # White background
            
            # Overlay confident regions (solid colors)
            particles = np.ma.masked_where(~(mask_slc & ~bg_slc), np.ones_like(slc))
            background = np.ma.masked_where(~bg_slc, np.ones_like(slc))
            axs[1,i].imshow(particles, cmap='Greens', alpha=1.0, vmin=0, vmax=1)  # Solid green
            axs[1,i].imshow(background, cmap='Blues', alpha=0.7, vmin=0, vmax=1)  # Solid blue
            axs[1,i].set_title('Confidence map')
            
            # Row 3: Intensity histogram
            axs[2,i].hist(slc.ravel(), bins=50, color='gray')
            axs[2,i].axvline(self.confidence_threshold, color='green', linestyle='--', label='Foreground thresh')
            if self.threshold_from_both_sides:
                axs[2,i].axvline(1-self.confidence_threshold, color='blue', linestyle='--', label='Background thresh')
            axs[2,i].legend()
        
        plt.tight_layout()
        plt.show()
    def step(self, metric, epoch):
        pass
