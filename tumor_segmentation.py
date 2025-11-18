"""
Tumor Segmentation Pipeline using Frequency-Domain Filtering
BME 271D Final Project - Fall 2025
Team: Ege Ozemek, Max Bazan, Sasha Nikiforov

This module implements tumor segmentation from MRI/CT images using:
1. FFT-based high-pass and band-pass filtering
2. Canny edge detection
3. Comparison to baseline Otsu thresholding methods
"""

import numpy as np
import matplotlib.pyplot as plt
from scipy import ndimage
from skimage import filters, feature, color, morphology
from skimage.io import imread, imsave
import pandas as pd
from pathlib import Path
import warnings
warnings.filterwarnings('ignore')


# ============================================================================
# DATA LOADING MODULE
# ============================================================================

def load_grayscale_image(path):
    """
    Load and normalize an MRI/CT image to grayscale [0,1] range.
    
    Parameters:
    -----------
    path : str
        Path to the image file
        
    Returns:
    --------
    img : np.ndarray
        Normalized grayscale image
    """
    img = imread(path)
    if img.ndim == 3:  # RGB -> grayscale
        img = color.rgb2gray(img)
    img = img.astype(np.float64)
    img /= (img.max() + 1e-8)
    return img


def load_binary_mask(path):
    """
    Load a ground truth tumor segmentation mask.
    
    Parameters:
    -----------
    path : str
        Path to the mask image file
        
    Returns:
    --------
    mask : np.ndarray
        Binary mask (0 = background, 1 = tumor)
    """
    mask = imread(path)
    if mask.ndim == 3:
        mask = color.rgb2gray(mask)
    mask = mask > (0.5 * mask.max())
    return mask.astype(np.uint8)


def load_dataset(image_dir, mask_dir=None):
    """
    Load multiple images and optionally their corresponding masks.
    
    Parameters:
    -----------
    image_dir : str
        Directory containing image files
    mask_dir : str, optional
        Directory containing mask files (must have same filenames as images)
        
    Returns:
    --------
    images : list
        List of loaded images
    masks : list or None
        List of loaded masks (None if mask_dir not provided)
    filenames : list
        List of image filenames
    """
    image_path = Path(image_dir)
    image_files = sorted(list(image_path.glob('*.png')) + 
                        list(image_path.glob('*.jpg')) + 
                        list(image_path.glob('*.jpeg')))
    
    images = []
    masks = []
    filenames = []
    
    for img_file in image_files:
        try:
            img = load_grayscale_image(str(img_file))
            images.append(img)
            filenames.append(img_file.name)
            
            if mask_dir is not None:
                mask_path = Path(mask_dir) / img_file.name
                if mask_path.exists():
                    mask = load_binary_mask(str(mask_path))
                    masks.append(mask)
                else:
                    print(f"Warning: No mask found for {img_file.name}")
                    masks.append(None)
        except Exception as e:
            print(f"Error loading {img_file.name}: {e}")
    
    return images, (masks if mask_dir else None), filenames


# ============================================================================
# FREQUENCY DOMAIN ANALYSIS MODULE
# ============================================================================

def compute_fft_spectrum(image):
    """
    Compute 2D FFT and return shifted spectrum.
    
    Parameters:
    -----------
    image : np.ndarray
        Input image
        
    Returns:
    --------
    F_shift : np.ndarray (complex)
        Shifted frequency spectrum (low frequencies at center)
    magnitude : np.ndarray
        Magnitude spectrum for visualization
    """
    F = np.fft.fft2(image)
    F_shift = np.fft.fftshift(F)
    magnitude = np.abs(F_shift)
    return F_shift, magnitude


def visualize_frequency_spectrum(image, F_shift, title="FFT Analysis"):
    """
    Plot original image alongside its log magnitude spectrum.
    
    Parameters:
    -----------
    image : np.ndarray
        Original image
    F_shift : np.ndarray
        Shifted frequency spectrum
    title : str
        Figure title
    """
    magnitude = np.abs(F_shift)
    
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    
    axes[0].imshow(image, cmap='gray')
    axes[0].set_title('Original MRI/CT Slice')
    axes[0].axis('off')
    
    axes[1].imshow(np.log1p(magnitude), cmap='gray')
    axes[1].set_title('Log Magnitude Spectrum')
    axes[1].axis('off')
    
    plt.suptitle(title, fontsize=14, fontweight='bold')
    plt.tight_layout()
    return fig


def reconstruct_without_phase(magnitude, shape):
    """
    Reconstruct image from magnitude spectrum only (zero phase).
    Demonstrates the importance of phase information.
    
    Parameters:
    -----------
    magnitude : np.ndarray
        Magnitude spectrum
    shape : tuple
        Original image shape
        
    Returns:
    --------
    reconstructed : np.ndarray
        Reconstructed image (will look scrambled without phase)
    """
    # Create complex spectrum with zero phase
    F_no_phase = magnitude + 0j
    F_no_phase = np.fft.ifftshift(F_no_phase)
    reconstructed = np.fft.ifft2(F_no_phase)
    reconstructed = np.abs(reconstructed)
    return reconstructed


# ============================================================================
# FILTER DESIGN MODULE
# ============================================================================

def make_hp_mask(shape, cutoff_radius):
    """
    Create high-pass filter mask (blocks low frequencies).
    Useful for emphasizing edges and boundaries.
    
    Parameters:
    -----------
    shape : tuple
        Image shape (height, width)
    cutoff_radius : int
        Radius of the blocked central region
        
    Returns:
    --------
    mask : np.ndarray
        High-pass filter mask
    """
    h, w = shape
    cy, cx = h // 2, w // 2
    Y, X = np.ogrid[:h, :w]
    dist = np.sqrt((Y - cy)**2 + (X - cx)**2)
    mask = dist > cutoff_radius
    return mask.astype(float)


def make_bp_mask(shape, r1, r2):
    """
    Create band-pass filter mask (keeps frequencies in a ring).
    Useful for isolating texture information.
    
    Parameters:
    -----------
    shape : tuple
        Image shape (height, width)
    r1 : int
        Inner radius (blocks lower frequencies)
    r2 : int
        Outer radius (blocks higher frequencies)
        
    Returns:
    --------
    mask : np.ndarray
        Band-pass filter mask
    """
    h, w = shape
    cy, cx = h // 2, w // 2
    Y, X = np.ogrid[:h, :w]
    dist = np.sqrt((Y - cy)**2 + (X - cx)**2)
    mask = (dist > r1) & (dist < r2)
    return mask.astype(float)


def make_lp_mask(shape, cutoff_radius):
    """
    Create low-pass filter mask (blocks high frequencies).
    Useful for smoothing.
    
    Parameters:
    -----------
    shape : tuple
        Image shape (height, width)
    cutoff_radius : int
        Radius of the kept central region
        
    Returns:
    --------
    mask : np.ndarray
        Low-pass filter mask
    """
    h, w = shape
    cy, cx = h // 2, w // 2
    Y, X = np.ogrid[:h, :w]
    dist = np.sqrt((Y - cy)**2 + (X - cx)**2)
    mask = dist <= cutoff_radius
    return mask.astype(float)


def visualize_filters(hp_mask, bp_mask, lp_mask=None):
    """
    Display filter masks for documentation.
    
    Parameters:
    -----------
    hp_mask : np.ndarray
        High-pass filter mask
    bp_mask : np.ndarray
        Band-pass filter mask
    lp_mask : np.ndarray, optional
        Low-pass filter mask
    """
    n_filters = 3 if lp_mask is not None else 2
    fig, axes = plt.subplots(1, n_filters, figsize=(12, 4))
    
    axes[0].imshow(hp_mask, cmap='gray')
    axes[0].set_title('High-Pass Filter')
    axes[0].axis('off')
    
    axes[1].imshow(bp_mask, cmap='gray')
    axes[1].set_title('Band-Pass Filter')
    axes[1].axis('off')
    
    if lp_mask is not None:
        axes[2].imshow(lp_mask, cmap='gray')
        axes[2].set_title('Low-Pass Filter')
        axes[2].axis('off')
    
    plt.suptitle('Frequency Domain Filters', fontsize=14, fontweight='bold')
    plt.tight_layout()
    return fig


# ============================================================================
# FILTER APPLICATION & RECONSTRUCTION MODULE
# ============================================================================

def apply_frequency_filter(F_shift, filter_mask):
    """
    Apply filter in frequency domain and reconstruct to spatial domain.
    
    Parameters:
    -----------
    F_shift : np.ndarray
        Shifted frequency spectrum
    filter_mask : np.ndarray
        Filter mask to apply
        
    Returns:
    --------
    filtered_image : np.ndarray
        Filtered image in spatial domain
    """
    # Apply filter
    F_filtered = F_shift * filter_mask
    
    # Inverse FFT
    F_filtered_unshift = np.fft.ifftshift(F_filtered)
    filtered = np.fft.ifft2(F_filtered_unshift)
    filtered_image = np.abs(filtered)
    
    # Normalize
    filtered_image /= (filtered_image.max() + 1e-8)
    
    return filtered_image


def filter_pipeline(image, filter_type='hp', **params):
    """
    Complete filtering pipeline: FFT → filter → inverse FFT → normalize.
    
    Parameters:
    -----------
    image : np.ndarray
        Input image
    filter_type : str
        Type of filter: 'hp', 'bp', or 'lp'
    **params : dict
        Filter parameters:
        - For 'hp': cutoff_radius
        - For 'bp': r1, r2
        - For 'lp': cutoff_radius
        
    Returns:
    --------
    filtered_image : np.ndarray
        Filtered image
    F_shift : np.ndarray
        Frequency spectrum (for visualization)
    filter_mask : np.ndarray
        Applied filter mask (for visualization)
    """
    # Compute FFT
    F_shift, _ = compute_fft_spectrum(image)
    
    # Create appropriate filter
    if filter_type == 'hp':
        cutoff = params.get('cutoff_radius', 25)
        filter_mask = make_hp_mask(image.shape, cutoff)
    elif filter_type == 'bp':
        r1 = params.get('r1', 10)
        r2 = params.get('r2', 40)
        filter_mask = make_bp_mask(image.shape, r1, r2)
    elif filter_type == 'lp':
        cutoff = params.get('cutoff_radius', 50)
        filter_mask = make_lp_mask(image.shape, cutoff)
    else:
        raise ValueError(f"Unknown filter type: {filter_type}")
    
    # Apply filter
    filtered_image = apply_frequency_filter(F_shift, filter_mask)
    
    return filtered_image, F_shift, filter_mask


# ============================================================================
# SEGMENTATION MODULE
# ============================================================================

def otsu_segmentation(image):
    """
    Apply Otsu thresholding with morphological cleanup.
    
    Parameters:
    -----------
    image : np.ndarray
        Input image
        
    Returns:
    --------
    mask : np.ndarray
        Binary segmentation mask
    """
    # Otsu threshold
    threshold = filters.threshold_otsu(image)
    mask = image > threshold
    
    # Morphological cleanup
    mask = ndimage.binary_opening(mask, np.ones((3, 3)))
    mask = ndimage.binary_closing(mask, np.ones((5, 5)))
    
    # Keep largest connected component
    labeled, n_components = ndimage.label(mask)
    if n_components > 0:
        sizes = ndimage.sum(mask, labeled, range(1, n_components + 1))
        largest_component = sizes.argmax() + 1
        mask = (labeled == largest_component)
    
    return mask.astype(np.uint8)


def canny_segmentation(image, sigma=1.0):
    """
    Canny edge detection → dilation → hole filling → largest component.
    
    Parameters:
    -----------
    image : np.ndarray
        Input image
    sigma : float
        Gaussian smoothing parameter for Canny
        
    Returns:
    --------
    mask : np.ndarray
        Binary segmentation mask
    """
    # Canny edge detection
    edges = feature.canny(image, sigma=sigma)
    
    # Dilate edges to close gaps
    edges = ndimage.binary_dilation(edges, np.ones((3, 3)))
    
    # Fill holes
    filled = ndimage.binary_fill_holes(edges)
    
    # Keep largest component
    labeled, n_components = ndimage.label(filled)
    if n_components > 0:
        sizes = ndimage.sum(filled, labeled, range(1, n_components + 1))
        largest_component = sizes.argmax() + 1
        mask = (labeled == largest_component)
    else:
        mask = filled
    
    return mask.astype(np.uint8)


def baseline_segmentation_raw(image):
    """
    Baseline 1: Direct Otsu thresholding on original image.
    
    Parameters:
    -----------
    image : np.ndarray
        Input image
        
    Returns:
    --------
    mask : np.ndarray
        Binary segmentation mask
    """
    return otsu_segmentation(image)


def baseline_segmentation_smooth(image, sigma=1.0):
    """
    Baseline 2: Gaussian smoothing then Otsu thresholding.
    
    Parameters:
    -----------
    image : np.ndarray
        Input image
    sigma : float
        Gaussian smoothing parameter
        
    Returns:
    --------
    mask : np.ndarray
        Binary segmentation mask
    """
    smoothed = ndimage.gaussian_filter(image, sigma)
    return otsu_segmentation(smoothed)


# ============================================================================
# EVALUATION METRICS MODULE
# ============================================================================

def dice_coefficient(pred_mask, gt_mask):
    """
    Compute Dice coefficient (F1 score for segmentation).
    Most common metric in medical image segmentation.
    
    Parameters:
    -----------
    pred_mask : np.ndarray
        Predicted binary mask
    gt_mask : np.ndarray
        Ground truth binary mask
        
    Returns:
    --------
    dice : float
        Dice coefficient [0, 1], higher is better
    """
    pred = pred_mask.astype(bool)
    gt = gt_mask.astype(bool)
    
    intersection = np.logical_and(pred, gt).sum()
    dice = 2 * intersection / (pred.sum() + gt.sum() + 1e-8)
    
    return dice


def iou_score(pred_mask, gt_mask):
    """
    Compute Intersection over Union (Jaccard Index).
    More strict than Dice coefficient.
    
    Parameters:
    -----------
    pred_mask : np.ndarray
        Predicted binary mask
    gt_mask : np.ndarray
        Ground truth binary mask
        
    Returns:
    --------
    iou : float
        IoU score [0, 1], higher is better
    """
    pred = pred_mask.astype(bool)
    gt = gt_mask.astype(bool)
    
    intersection = np.logical_and(pred, gt).sum()
    union = np.logical_or(pred, gt).sum()
    iou = intersection / (union + 1e-8)
    
    return iou


def boundary_accuracy(pred_mask, gt_mask, tolerance=2):
    """
    Measure how close predicted boundary is to ground truth boundary.
    
    Parameters:
    -----------
    pred_mask : np.ndarray
        Predicted binary mask
    gt_mask : np.ndarray
        Ground truth binary mask
    tolerance : int
        Distance tolerance in pixels
        
    Returns:
    --------
    accuracy : float
        Boundary accuracy [0, 1], higher is better
    """
    # Extract boundaries
    pred_boundary = pred_mask.astype(bool) ^ ndimage.binary_erosion(pred_mask.astype(bool))
    gt_boundary = gt_mask.astype(bool) ^ ndimage.binary_erosion(gt_mask.astype(bool))
    
    # Distance transform from GT boundary
    dist_from_gt = ndimage.distance_transform_edt(~gt_boundary)
    
    # Count predicted boundary pixels within tolerance
    pred_boundary_coords = np.where(pred_boundary)
    if len(pred_boundary_coords[0]) == 0:
        return 0.0
    
    distances = dist_from_gt[pred_boundary_coords]
    within_tolerance = (distances <= tolerance).sum()
    accuracy = within_tolerance / len(distances)
    
    return accuracy


def evaluate_all_metrics(pred_mask, gt_mask):
    """
    Compute all evaluation metrics.
    
    Parameters:
    -----------
    pred_mask : np.ndarray
        Predicted binary mask
    gt_mask : np.ndarray
        Ground truth binary mask
        
    Returns:
    --------
    metrics : dict
        Dictionary containing all metric scores
    """
    return {
        'dice': dice_coefficient(pred_mask, gt_mask),
        'iou': iou_score(pred_mask, gt_mask),
        'boundary_acc': boundary_accuracy(pred_mask, gt_mask)
    }


# ============================================================================
# VOLUME ESTIMATION MODULE
# ============================================================================

def estimate_tumor_volume(mask, pixel_spacing=(1.0, 1.0), slice_thickness=1.0):
    """
    Estimate tumor volume from segmentation mask.
    
    Parameters:
    -----------
    mask : np.ndarray
        Binary segmentation mask
    pixel_spacing : tuple
        (row_spacing, col_spacing) in mm
    slice_thickness : float
        Slice thickness in mm
        
    Returns:
    --------
    volume_mm3 : float
        Estimated volume in mm³
    volume_cm3 : float
        Estimated volume in cm³
    """
    # Count tumor pixels
    tumor_pixels = mask.sum()
    
    # Compute pixel area
    pixel_area_mm2 = pixel_spacing[0] * pixel_spacing[1]
    
    # Compute volume
    volume_mm3 = tumor_pixels * pixel_area_mm2 * slice_thickness
    volume_cm3 = volume_mm3 / 1000.0  # Convert to cm³
    
    return volume_mm3, volume_cm3


def display_volume_measurement(image, mask, volume_mm3, volume_cm3):
    """
    Visualize mask with volume annotation.
    
    Parameters:
    -----------
    image : np.ndarray
        Original image
    mask : np.ndarray
        Segmentation mask
    volume_mm3 : float
        Volume in mm³
    volume_cm3 : float
        Volume in cm³
    """
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    
    axes[0].imshow(image, cmap='gray')
    axes[0].set_title('Original Image')
    axes[0].axis('off')
    
    axes[1].imshow(image, cmap='gray')
    axes[1].imshow(mask, cmap='Reds', alpha=0.5)
    axes[1].set_title(f'Segmentation\nVolume: {volume_mm3:.1f} mm³ ({volume_cm3:.3f} cm³)')
    axes[1].axis('off')
    
    plt.tight_layout()
    return fig


# ============================================================================
# VISUALIZATION MODULE
# ============================================================================

def plot_segmentation_comparison(image, masks_dict, gt_mask=None):
    """
    Create subplot grid showing all segmentation results.
    
    Parameters:
    -----------
    image : np.ndarray
        Original image
    masks_dict : dict
        Dictionary of masks: {'method_name': mask_array}
    gt_mask : np.ndarray, optional
        Ground truth mask
    """
    n_methods = len(masks_dict) + 1 + (1 if gt_mask is not None else 0)
    n_cols = 3
    n_rows = int(np.ceil(n_methods / n_cols))
    
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(15, 5 * n_rows))
    axes = axes.flatten() if n_rows > 1 else np.array([axes]).flatten()
    
    # Original image
    axes[0].imshow(image, cmap='gray')
    axes[0].set_title('Original Image', fontweight='bold')
    axes[0].axis('off')
    
    # Ground truth
    idx = 1
    if gt_mask is not None:
        axes[idx].imshow(image, cmap='gray')
        axes[idx].imshow(gt_mask, cmap='Greens', alpha=0.5)
        axes[idx].set_title('Ground Truth', fontweight='bold')
        axes[idx].axis('off')
        idx += 1
    
    # All segmentation methods
    for method_name, mask in masks_dict.items():
        axes[idx].imshow(image, cmap='gray')
        axes[idx].imshow(mask, cmap='Reds', alpha=0.5)
        axes[idx].set_title(method_name, fontweight='bold')
        axes[idx].axis('off')
        idx += 1
    
    # Hide unused subplots
    for i in range(idx, len(axes)):
        axes[i].axis('off')
    
    plt.suptitle('Segmentation Method Comparison', fontsize=16, fontweight='bold')
    plt.tight_layout()
    return fig


def plot_filtering_steps(image, hp_filtered, bp_filtered):
    """
    Show original → HP filtered → BP filtered.
    
    Parameters:
    -----------
    image : np.ndarray
        Original image
    hp_filtered : np.ndarray
        High-pass filtered image
    bp_filtered : np.ndarray
        Band-pass filtered image
    """
    fig, axes = plt.subplots(1, 3, figsize=(15, 5))
    
    axes[0].imshow(image, cmap='gray')
    axes[0].set_title('Original Image')
    axes[0].axis('off')
    
    axes[1].imshow(hp_filtered, cmap='gray')
    axes[1].set_title('High-Pass Filtered\n(Edge Enhancement)')
    axes[1].axis('off')
    
    axes[2].imshow(bp_filtered, cmap='gray')
    axes[2].set_title('Band-Pass Filtered\n(Texture Isolation)')
    axes[2].axis('off')
    
    plt.suptitle('Frequency Domain Filtering Effects', fontsize=14, fontweight='bold')
    plt.tight_layout()
    return fig


def create_overlay_visualization(image, pred_mask, gt_mask, method_name="Method"):
    """
    Overlay predicted (red) and ground truth (green) contours on image.
    
    Parameters:
    -----------
    image : np.ndarray
        Original image
    pred_mask : np.ndarray
        Predicted mask
    gt_mask : np.ndarray
        Ground truth mask
    method_name : str
        Name of the method for title
    """
    # Extract contours
    pred_contour = pred_mask.astype(bool) ^ ndimage.binary_erosion(pred_mask.astype(bool))
    gt_contour = gt_mask.astype(bool) ^ ndimage.binary_erosion(gt_mask.astype(bool))
    
    # Create RGB overlay
    overlay = np.stack([image, image, image], axis=-1)
    overlay[pred_contour] = [1, 0, 0]  # Red for prediction
    overlay[gt_contour] = [0, 1, 0]    # Green for ground truth
    
    fig, ax = plt.subplots(1, 1, figsize=(8, 8))
    ax.imshow(overlay)
    ax.set_title(f'{method_name}\nRed=Predicted, Green=Ground Truth', fontweight='bold')
    ax.axis('off')
    
    plt.tight_layout()
    return fig


def plot_metrics_comparison(results_df, metric='dice'):
    """
    Create bar plot comparing methods by a specific metric.
    
    Parameters:
    -----------
    results_df : pd.DataFrame
        Results dataframe with methods as columns
    metric : str
        Metric to plot ('dice', 'iou', or 'boundary_acc')
    """
    # Get mean and std for each method
    methods = [col for col in results_df.columns if col not in ['image', 'filename']]
    means = [results_df[method][metric].mean() for method in methods]
    stds = [results_df[method][metric].std() for method in methods]
    
    fig, ax = plt.subplots(figsize=(12, 6))
    x = np.arange(len(methods))
    bars = ax.bar(x, means, yerr=stds, capsize=5, alpha=0.7, 
                   color=['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728', '#9467bd'])
    
    ax.set_xlabel('Method', fontsize=12, fontweight='bold')
    ax.set_ylabel(f'{metric.upper()} Score', fontsize=12, fontweight='bold')
    ax.set_title(f'Segmentation Performance Comparison ({metric.upper()})', 
                 fontsize=14, fontweight='bold')
    ax.set_xticks(x)
    ax.set_xticklabels(methods, rotation=45, ha='right')
    ax.set_ylim(0, 1.0)
    ax.grid(axis='y', alpha=0.3)
    
    # Add value labels on bars
    for i, (bar, mean, std) in enumerate(zip(bars, means, stds)):
        height = bar.get_height()
        ax.text(bar.get_x() + bar.get_width()/2., height + std + 0.02,
                f'{mean:.3f}±{std:.3f}',
                ha='center', va='bottom', fontsize=9)
    
    plt.tight_layout()
    return fig


# ============================================================================
# MAIN EXPERIMENTAL PIPELINE
# ============================================================================

def run_single_image_experiment(image, gt_mask, params, verbose=True):
    """
    Run all segmentation methods on a single image.
    
    Parameters:
    -----------
    image : np.ndarray
        Input image
    gt_mask : np.ndarray
        Ground truth mask
    params : dict
        Experiment parameters
    verbose : bool
        Print progress messages
        
    Returns:
    --------
    results : dict
        Dictionary containing all masks and metrics
    """
    if verbose:
        print("Running segmentation experiment...")
    
    results = {}
    
    # 1. Baseline methods
    if verbose:
        print("  - Baseline 1: Raw Otsu...")
    mask_raw_otsu = baseline_segmentation_raw(image)
    results['Baseline_Raw_Otsu'] = {
        'mask': mask_raw_otsu,
        'metrics': evaluate_all_metrics(mask_raw_otsu, gt_mask)
    }
    
    if verbose:
        print("  - Baseline 2: Smoothed Otsu...")
    mask_smooth_otsu = baseline_segmentation_smooth(image, params.get('gaussian_sigma', 1.0))
    results['Baseline_Smooth_Otsu'] = {
        'mask': mask_smooth_otsu,
        'metrics': evaluate_all_metrics(mask_smooth_otsu, gt_mask)
    }
    
    # 2. FFT High-Pass method
    if verbose:
        print("  - FFT High-Pass...")
    hp_filtered, _, _ = filter_pipeline(image, 'hp', cutoff_radius=params.get('hp_radius', 25))
    mask_hp = otsu_segmentation(hp_filtered)
    results['FFT_HighPass'] = {
        'mask': mask_hp,
        'filtered_image': hp_filtered,
        'metrics': evaluate_all_metrics(mask_hp, gt_mask)
    }
    
    # 3. FFT Band-Pass method
    if verbose:
        print("  - FFT Band-Pass...")
    bp_filtered, _, _ = filter_pipeline(image, 'bp', 
                                        r1=params.get('bp_r1', 10), 
                                        r2=params.get('bp_r2', 40))
    mask_bp = otsu_segmentation(bp_filtered)
    results['FFT_BandPass'] = {
        'mask': mask_bp,
        'filtered_image': bp_filtered,
        'metrics': evaluate_all_metrics(mask_bp, gt_mask)
    }
    
    # 4. Canny method
    if verbose:
        print("  - Canny Edge Detection...")
    mask_canny = canny_segmentation(image, params.get('canny_sigma', 1.0))
    results['Canny_Edges'] = {
        'mask': mask_canny,
        'metrics': evaluate_all_metrics(mask_canny, gt_mask)
    }
    
    if verbose:
        print("Done!")
    
    return results


def run_batch_experiment(images, masks, filenames, params, output_dir=None):
    """
    Run experiments on multiple images.
    
    Parameters:
    -----------
    images : list
        List of images
    masks : list
        List of ground truth masks
    filenames : list
        List of image filenames
    params : dict
        Experiment parameters
    output_dir : str, optional
        Directory to save figures
        
    Returns:
    --------
    results_df : pd.DataFrame
        DataFrame with all results for statistical analysis
    """
    print(f"Running batch experiment on {len(images)} images...")
    
    all_results = []
    
    for idx, (image, gt_mask, filename) in enumerate(zip(images, masks, filenames)):
        print(f"\n[{idx+1}/{len(images)}] Processing {filename}...")
        
        # Run experiment
        result = run_single_image_experiment(image, gt_mask, params, verbose=False)
        
        # Store results
        row = {'filename': filename, 'image_idx': idx}
        for method_name, method_data in result.items():
            for metric_name, metric_value in method_data['metrics'].items():
                row[f'{method_name}_{metric_name}'] = metric_value
        all_results.append(row)
        
        # Save visualizations if output directory provided
        if output_dir is not None:
            output_path = Path(output_dir)
            output_path.mkdir(exist_ok=True, parents=True)
            
            # Create comparison figure
            masks_dict = {name: data['mask'] for name, data in result.items()}
            fig = plot_segmentation_comparison(image, masks_dict, gt_mask)
            fig.savefig(output_path / f'comparison_{filename}', dpi=150, bbox_inches='tight')
            plt.close(fig)
    
    # Convert to DataFrame
    results_df = pd.DataFrame(all_results)
    
    print("\nBatch experiment complete!")
    return results_df


def compare_methods_statistically(results_df, output_dir=None):
    """
    Compute statistics and create comparison plots.
    
    Parameters:
    -----------
    results_df : pd.DataFrame
        Results dataframe from batch experiment
    output_dir : str, optional
        Directory to save figures
    """
    print("\n=== STATISTICAL ANALYSIS ===\n")
    
    # Extract method names (those with _dice suffix)
    method_cols = [col for col in results_df.columns if col.endswith('_dice')]
    methods = [col.replace('_dice', '') for col in method_cols]
    
    # Compute statistics for each metric
    metrics = ['dice', 'iou', 'boundary_acc']
    
    for metric in metrics:
        print(f"\n{metric.upper()} Scores:")
        print("-" * 60)
        
        for method in methods:
            col_name = f'{method}_{metric}'
            if col_name in results_df.columns:
                mean = results_df[col_name].mean()
                std = results_df[col_name].std()
                print(f"{method:25s}: {mean:.4f} ± {std:.4f}")
        
        # Create bar plot
        if output_dir is not None:
            stats_data = []
            for method in methods:
                col_name = f'{method}_{metric}'
                if col_name in results_df.columns:
                    stats_data.append({
                        'method': method,
                        'mean': results_df[col_name].mean(),
                        'std': results_df[col_name].std()
                    })
            
            stats_df = pd.DataFrame(stats_data)
            
            fig, ax = plt.subplots(figsize=(12, 6))
            x = np.arange(len(stats_df))
            bars = ax.bar(x, stats_df['mean'], yerr=stats_df['std'], 
                         capsize=5, alpha=0.7,
                         color=['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728', '#9467bd'])
            
            ax.set_xlabel('Method', fontsize=12, fontweight='bold')
            ax.set_ylabel(f'{metric.upper()} Score', fontsize=12, fontweight='bold')
            ax.set_title(f'Segmentation Performance ({metric.upper()})', 
                        fontsize=14, fontweight='bold')
            ax.set_xticks(x)
            ax.set_xticklabels(stats_df['method'], rotation=45, ha='right')
            ax.set_ylim(0, 1.0)
            ax.grid(axis='y', alpha=0.3)
            
            # Add value labels
            for i, (bar, row) in enumerate(zip(bars, stats_df.itertuples())):
                height = bar.get_height()
                ax.text(bar.get_x() + bar.get_width()/2., height + row.std + 0.02,
                       f'{row.mean:.3f}±{row.std:.3f}',
                       ha='center', va='bottom', fontsize=9)
            
            plt.tight_layout()
            fig.savefig(Path(output_dir) / f'stats_{metric}.png', dpi=150, bbox_inches='tight')
            plt.close(fig)
    
    # Determine best method
    print("\n" + "=" * 60)
    print("BEST METHODS:")
    print("=" * 60)
    for metric in metrics:
        best_method = None
        best_score = -1
        for method in methods:
            col_name = f'{method}_{metric}'
            if col_name in results_df.columns:
                mean = results_df[col_name].mean()
                if mean > best_score:
                    best_score = mean
                    best_method = method
        print(f"{metric.upper():15s}: {best_method} ({best_score:.4f})")


# ============================================================================
# MAIN EXECUTION
# ============================================================================

def main(image_dir, mask_dir, output_dir='results', params=None):
    """
    Complete workflow for tumor segmentation project.
    
    Parameters:
    -----------
    image_dir : str
        Directory containing images
    mask_dir : str
        Directory containing ground truth masks
    output_dir : str
        Directory to save results
    params : dict, optional
        Experiment parameters
    """
    # Default parameters
    if params is None:
        params = {
            'hp_radius': 25,
            'bp_r1': 10,
            'bp_r2': 40,
            'canny_sigma': 1.0,
            'gaussian_sigma': 1.0
        }
    
    print("=" * 70)
    print(" TUMOR SEGMENTATION PIPELINE - BME 271D Final Project")
    print(" Team: Ege Ozemek, Max Bazan, Sasha Nikiforov")
    print("=" * 70)
    
    # Create output directory
    output_path = Path(output_dir)
    output_path.mkdir(exist_ok=True, parents=True)
    
    # Load dataset
    print("\n1. Loading dataset...")
    images, masks, filenames = load_dataset(image_dir, mask_dir)
    print(f"   Loaded {len(images)} images with masks")
    
    # Run batch experiment
    print("\n2. Running batch experiments...")
    results_df = run_batch_experiment(images, masks, filenames, params, output_dir)
    
    # Save results
    print("\n3. Saving results...")
    results_df.to_csv(output_path / 'segmentation_results.csv', index=False)
    print(f"   Results saved to {output_path / 'segmentation_results.csv'}")
    
    # Statistical analysis
    print("\n4. Statistical analysis...")
    compare_methods_statistically(results_df, output_dir)
    
    # Create summary figure from first image
    if len(images) > 0:
        print("\n5. Creating summary visualizations...")
        img = images[0]
        gt = masks[0]
        
        # Run single experiment for detailed visualizations
        result = run_single_image_experiment(img, gt, params, verbose=False)
        
        # Frequency spectrum visualization
        F_shift, mag = compute_fft_spectrum(img)
        fig = visualize_frequency_spectrum(img, F_shift)
        fig.savefig(output_path / 'fft_analysis.png', dpi=150, bbox_inches='tight')
        plt.close(fig)
        
        # Filter visualization
        hp_mask = make_hp_mask(img.shape, params['hp_radius'])
        bp_mask = make_bp_mask(img.shape, params['bp_r1'], params['bp_r2'])
        fig = visualize_filters(hp_mask, bp_mask)
        fig.savefig(output_path / 'filter_masks.png', dpi=150, bbox_inches='tight')
        plt.close(fig)
        
        # Filtering effects
        hp_img = result['FFT_HighPass']['filtered_image']
        bp_img = result['FFT_BandPass']['filtered_image']
        fig = plot_filtering_steps(img, hp_img, bp_img)
        fig.savefig(output_path / 'filtering_effects.png', dpi=150, bbox_inches='tight')
        plt.close(fig)
        
        print(f"   Visualizations saved to {output_dir}/")
    
    print("\n" + "=" * 70)
    print(" EXPERIMENT COMPLETE!")
    print("=" * 70)
    print(f"\nAll results saved to: {output_dir}/")
    print("\nKey files:")
    print(f"  - segmentation_results.csv: Quantitative results")
    print(f"  - comparison_*.png: Visual comparison for each image")
    print(f"  - stats_*.png: Statistical comparison plots")
    print(f"  - fft_analysis.png: Frequency domain analysis")
    print(f"  - filter_masks.png: Filter visualization")
    print(f"  - filtering_effects.png: Filtering effects demonstration")


if __name__ == "__main__":
    # Example usage
    print(__doc__)
    print("\nTo run the pipeline, use:")
    print("  main(image_dir='path/to/images', mask_dir='path/to/masks')")
    print("\nOr for a single image:")
    print("  img = load_grayscale_image('image.png')")
    print("  mask = load_binary_mask('mask.png')")
    print("  params = {'hp_radius': 25, 'bp_r1': 10, 'bp_r2': 40}")
    print("  results = run_single_image_experiment(img, mask, params)")
