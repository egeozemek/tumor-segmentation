"""
Quick Test Script for Tumor Segmentation Pipeline
Run this to verify everything is working correctly
"""

import numpy as np
import matplotlib.pyplot as plt
from skimage import data, filters
from skimage.io import imsave
import tumor_segmentation as ts
from pathlib import Path

def create_synthetic_tumor_image(size=256):
    """
    Create a synthetic tumor image for testing.
    
    Returns:
    --------
    image : np.ndarray
        Synthetic medical image
    mask : np.ndarray
        Ground truth tumor mask
    """
    # Create base image
    image = np.random.randn(size, size) * 0.1 + 0.5
    
    # Add tumor (bright ellipse)
    cy, cx = size // 2, size // 2
    Y, X = np.ogrid[:size, :size]
    
    # Create elliptical tumor
    tumor_mask = ((Y - cy) / 40)**2 + ((X - cx) / 60)**2 < 1
    
    # Add tumor intensity
    image[tumor_mask] = image[tumor_mask] + 0.4
    
    # Add some texture to tumor
    texture = np.random.randn(size, size) * 0.05
    image[tumor_mask] = image[tumor_mask] + texture[tumor_mask]
    
    # Smooth the image
    image = filters.gaussian(image, sigma=1.5)
    
    # Normalize
    image = (image - image.min()) / (image.max() - image.min())
    
    return image, tumor_mask.astype(np.uint8)


def test_basic_functionality():
    """Test basic functions"""
    print("="*70)
    print(" TUMOR SEGMENTATION PIPELINE - FUNCTIONALITY TEST")
    print("="*70)
    
    # Create synthetic data
    print("\n1. Creating synthetic tumor image...")
    image, gt_mask = create_synthetic_tumor_image()
    print(f"   ✓ Created {image.shape} image with {gt_mask.sum()} tumor pixels")
    
    # Test FFT
    print("\n2. Testing FFT computation...")
    F_shift, magnitude = ts.compute_fft_spectrum(image)
    print(f"   ✓ FFT spectrum computed: {F_shift.shape}")
    
    # Test filters
    print("\n3. Testing filter creation...")
    hp_mask = ts.make_hp_mask(image.shape, 25)
    bp_mask = ts.make_bp_mask(image.shape, 10, 40)
    print(f"   ✓ High-pass filter: {hp_mask.sum()}/{hp_mask.size} pixels pass")
    print(f"   ✓ Band-pass filter: {bp_mask.sum()}/{bp_mask.size} pixels pass")
    
    # Test filtering
    print("\n4. Testing frequency-domain filtering...")
    hp_filtered, _, _ = ts.filter_pipeline(image, 'hp', cutoff_radius=25)
    bp_filtered, _, _ = ts.filter_pipeline(image, 'bp', r1=10, r2=40)
    print(f"   ✓ High-pass filtered image: range [{hp_filtered.min():.3f}, {hp_filtered.max():.3f}]")
    print(f"   ✓ Band-pass filtered image: range [{bp_filtered.min():.3f}, {bp_filtered.max():.3f}]")
    
    # Test segmentation
    print("\n5. Testing segmentation methods...")
    mask_otsu = ts.otsu_segmentation(image)
    mask_hp = ts.otsu_segmentation(hp_filtered)
    mask_canny = ts.canny_segmentation(image)
    print(f"   ✓ Otsu segmentation: {mask_otsu.sum()} pixels")
    print(f"   ✓ HP+Otsu segmentation: {mask_hp.sum()} pixels")
    print(f"   ✓ Canny segmentation: {mask_canny.sum()} pixels")
    
    # Test metrics
    print("\n6. Testing evaluation metrics...")
    dice = ts.dice_coefficient(mask_hp, gt_mask)
    iou = ts.iou_score(mask_hp, gt_mask)
    boundary = ts.boundary_accuracy(mask_hp, gt_mask)
    print(f"   ✓ Dice coefficient: {dice:.4f}")
    print(f"   ✓ IoU score: {iou:.4f}")
    print(f"   ✓ Boundary accuracy: {boundary:.4f}")
    
    # Test volume estimation
    print("\n7. Testing volume estimation...")
    vol_mm3, vol_cm3 = ts.estimate_tumor_volume(mask_hp, (0.5, 0.5), 5.0)
    print(f"   ✓ Estimated volume: {vol_mm3:.1f} mm³ ({vol_cm3:.3f} cm³)")
    
    # Create visualizations
    print("\n8. Creating test visualizations...")
    
    # FFT visualization
    fig = ts.visualize_frequency_spectrum(image, F_shift, "Test: FFT Analysis")
    plt.savefig('test_fft.png', dpi=150, bbox_inches='tight')
    plt.close(fig)
    print("   ✓ Saved: test_fft.png")
    
    # Filter visualization
    fig = ts.visualize_filters(hp_mask, bp_mask)
    plt.savefig('test_filters.png', dpi=150, bbox_inches='tight')
    plt.close(fig)
    print("   ✓ Saved: test_filters.png")
    
    # Filtering effects
    fig = ts.plot_filtering_steps(image, hp_filtered, bp_filtered)
    plt.savefig('test_filtering.png', dpi=150, bbox_inches='tight')
    plt.close(fig)
    print("   ✓ Saved: test_filtering.png")
    
    # Segmentation comparison
    masks_dict = {
        'Raw Otsu': mask_otsu,
        'HP+Otsu': mask_hp,
        'Canny': mask_canny
    }
    fig = ts.plot_segmentation_comparison(image, masks_dict, gt_mask)
    plt.savefig('test_segmentation.png', dpi=150, bbox_inches='tight')
    plt.close(fig)
    print("   ✓ Saved: test_segmentation.png")
    
    # Overlay
    fig = ts.create_overlay_visualization(image, mask_hp, gt_mask, "HP+Otsu Method")
    plt.savefig('test_overlay.png', dpi=150, bbox_inches='tight')
    plt.close(fig)
    print("   ✓ Saved: test_overlay.png")
    
    print("\n" + "="*70)
    print(" ALL TESTS PASSED! ✓")
    print("="*70)
    print("\nGenerated test images:")
    print("  - test_fft.png: FFT spectrum visualization")
    print("  - test_filters.png: Filter mask visualization")
    print("  - test_filtering.png: Filtering effects")
    print("  - test_segmentation.png: Segmentation comparison")
    print("  - test_overlay.png: Prediction vs ground truth overlay")
    print("\nYou're ready to run the full pipeline on real TCIA data!")
    
    return image, gt_mask


def create_test_dataset(n_images=5, output_dir='test_data'):
    """
    Create a small test dataset for batch processing.
    
    Parameters:
    -----------
    n_images : int
        Number of test images to create
    output_dir : str
        Directory to save test data
    """
    print(f"\nCreating test dataset with {n_images} images...")
    
    # Create directories
    image_dir = Path(output_dir) / 'images'
    mask_dir = Path(output_dir) / 'masks'
    image_dir.mkdir(parents=True, exist_ok=True)
    mask_dir.mkdir(parents=True, exist_ok=True)
    
    # Generate images
    for i in range(n_images):
        # Create image with varying tumor characteristics
        size = np.random.choice([200, 256, 300])
        image, mask = create_synthetic_tumor_image(size)
        
        # Save
        img_path = image_dir / f'tumor_{i:02d}.png'
        mask_path = mask_dir / f'tumor_{i:02d}.png'
        
        imsave(str(img_path), (image * 255).astype(np.uint8))
        imsave(str(mask_path), (mask * 255).astype(np.uint8))
        
        print(f"  ✓ Created tumor_{i:02d}.png")
    
    print(f"\nTest dataset created in '{output_dir}/'")
    print(f"  - {n_images} images in '{image_dir}'")
    print(f"  - {n_images} masks in '{mask_dir}'")
    
    return str(image_dir), str(mask_dir)


def run_mini_experiment():
    """Run a mini batch experiment on synthetic data"""
    print("\n" + "="*70)
    print(" RUNNING MINI BATCH EXPERIMENT")
    print("="*70)
    
    # Create test dataset
    img_dir, mask_dir = create_test_dataset(n_images=3)
    
    # Load dataset
    images, masks, filenames = ts.load_dataset(img_dir, mask_dir)
    print(f"\nLoaded {len(images)} images")
    
    # Define parameters
    params = {
        'hp_radius': 25,
        'bp_r1': 10,
        'bp_r2': 40,
        'canny_sigma': 1.0,
        'gaussian_sigma': 1.0
    }
    
    # Run batch experiment
    results_df = ts.run_batch_experiment(images, masks, filenames, params, 'test_results')
    
    # Show statistics
    print("\n" + "="*70)
    print(" MINI EXPERIMENT RESULTS")
    print("="*70)
    ts.compare_methods_statistically(results_df, 'test_results')
    
    print("\nResults saved to 'test_results/' directory")


if __name__ == "__main__":
    print(__doc__)
    
    # Run basic tests
    image, mask = test_basic_functionality()
    
    # Ask if user wants to run mini experiment
    print("\n" + "="*70)
    response = input("\nRun mini batch experiment? (y/n): ").strip().lower()
    if response == 'y':
        run_mini_experiment()
    
    print("\n" + "="*70)
    print(" TEST COMPLETE!")
    print("="*70)
    print("\nNext steps:")
    print("1. Download real tumor images from TCIA")
    print("2. Organize them into data/images/ and data/masks/")
    print("3. Run the Jupyter notebook: Tumor_Segmentation_Final_Project.ipynb")
    print("4. Or use: ts.main('data/images', 'data/masks', 'results')")
