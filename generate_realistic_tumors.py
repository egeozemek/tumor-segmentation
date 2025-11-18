"""
Generate Realistic Tumor Images for Testing
This creates images that mimic real MRI/CT tumor characteristics:
- Irregular boundaries
- Heterogeneous texture
- Variable contrast
- Noise
"""

import numpy as np
import matplotlib.pyplot as plt
from scipy import ndimage
from skimage import filters
from skimage.io import imsave
from pathlib import Path


def create_realistic_tumor(size=256, tumor_type='brain'):
    """
    Create a realistic tumor image with complex characteristics.
    
    Parameters:
    -----------
    size : int
        Image size
    tumor_type : str
        'brain', 'liver', or 'lung' - affects texture and contrast
        
    Returns:
    --------
    image : np.ndarray
        Realistic tumor image
    mask : np.ndarray
        Ground truth tumor mask
    """
    np.random.seed(None)  # Different each time
    
    # Create base background - DARKER for better tumor contrast
    background_base = 0.25 + np.random.randn(size, size) * 0.04  # Changed from 0.3 and 0.05
    
    # Add structured noise (mimics tissue texture)
    noise_freq = np.fft.fft2(np.random.randn(size, size))
    noise_freq = ndimage.fourier_gaussian(noise_freq, sigma=10)
    structured_noise = np.real(np.fft.ifft2(noise_freq))
    structured_noise = (structured_noise - structured_noise.min()) / (structured_noise.max() - structured_noise.min())
    
    background = background_base + structured_noise * 0.1
    
    # Create irregular tumor shape
    cy, cx = size // 2 + np.random.randint(-20, 20), size // 2 + np.random.randint(-20, 20)
    Y, X = np.ogrid[:size, :size]
    
    # Base ellipse with random orientation
    angle = np.random.uniform(0, np.pi)
    a, b = np.random.uniform(30, 50), np.random.uniform(40, 60)
    
    X_rot = (X - cx) * np.cos(angle) - (Y - cy) * np.sin(angle)
    Y_rot = (X - cx) * np.sin(angle) + (Y - cy) * np.cos(angle)
    
    ellipse = (X_rot / a) ** 2 + (Y_rot / b) ** 2 < 1
    
    # Add irregularities to boundary
    theta = np.linspace(0, 2*np.pi, 100)
    irregularity = np.random.randn(100) * 0.15
    irregularity = ndimage.gaussian_filter1d(irregularity, 5, mode='wrap')
    
    # Create irregular mask
    mask = np.zeros((size, size), dtype=bool)
    for i in range(len(theta)):
        t = theta[i]
        r = (a + b) / 2 * (1 + irregularity[i])
        x_pos = int(cx + r * np.cos(t + angle))
        y_pos = int(cy + r * np.sin(t + angle))
        if 0 <= x_pos < size and 0 <= y_pos < size:
            mask[y_pos, x_pos] = True
    
    # Fill and smooth the mask
    mask = ndimage.binary_fill_holes(mask)
    mask = ndimage.binary_dilation(mask, iterations=2)
    mask = ndimage.binary_erosion(mask, iterations=1)
    mask_smooth = ndimage.gaussian_filter(mask.astype(float), 2) > 0.5
    
    # Create tumor with heterogeneous texture
    # INCREASED CONTRAST - tumors are now more visible
    if tumor_type == 'brain':
        tumor_base_intensity = 0.75  # Increased from 0.7
        texture_strength = 0.10      # Reduced from 0.15 (less noisy)
    elif tumor_type == 'liver':
        tumor_base_intensity = 0.70  # Increased from 0.6
        texture_strength = 0.08      # Reduced from 0.12
    else:  # lung
        tumor_base_intensity = 0.72  # Increased from 0.65
        texture_strength = 0.08      # Reduced from 0.1
    
    # Add heterogeneous texture to tumor
    tumor_texture = np.random.randn(size, size) * texture_strength
    tumor_texture = ndimage.gaussian_filter(tumor_texture, 3)
    
    # Add some necrotic core (darker region in center)
    if np.random.rand() > 0.5:
        necrotic_core = ((X - cx)**2 + (Y - cy)**2) < (min(a, b) * 0.3)**2
        necrotic_intensity = -0.2
    else:
        necrotic_core = np.zeros((size, size), dtype=bool)
        necrotic_intensity = 0
    
    # Combine everything
    image = background.copy()
    image[mask_smooth] = tumor_base_intensity + tumor_texture[mask_smooth]
    image[necrotic_core] += necrotic_intensity
    
    # Add edge enhancement (tumor boundary is often enhanced in MRI)
    boundary = mask_smooth.astype(float) - ndimage.binary_erosion(mask_smooth, iterations=3).astype(float)
    image = image + boundary * 0.15
    
    # Add overall gaussian smoothing (mimics MRI/CT blur)
    image = ndimage.gaussian_filter(image, 1.5)
    
    # Add some imaging artifacts
    # Intensity inhomogeneity (bias field)
    x_bias = np.linspace(-1, 1, size)
    y_bias = np.linspace(-1, 1, size)
    X_bias, Y_bias = np.meshgrid(x_bias, y_bias)
    bias_field = 1 + 0.1 * (X_bias**2 + Y_bias**2)
    image = image * bias_field
    
    # Add noise - REDUCED for clearer tumor visibility
    noise_level = 0.02  # Reduced from 0.03
    image = image + np.random.randn(size, size) * noise_level
    
    # Normalize
    image = np.clip(image, 0, 1)
    image = (image - image.min()) / (image.max() - image.min())
    
    return image, mask_smooth.astype(np.uint8)


def generate_tumor_dataset(n_images=10, output_dir='data', tumor_type='brain'):
    """
    Generate a complete dataset of realistic tumor images.
    
    Parameters:
    -----------
    n_images : int
        Number of images to generate
    output_dir : str
        Output directory
    tumor_type : str
        Type of tumor to generate
    """
    print(f"Generating {n_images} realistic {tumor_type} tumor images...")
    
    # Create directories
    image_dir = Path(output_dir) / 'images'
    mask_dir = Path(output_dir) / 'masks'
    image_dir.mkdir(parents=True, exist_ok=True)
    mask_dir.mkdir(parents=True, exist_ok=True)
    
    for i in range(n_images):
        # Generate image
        image, mask = create_realistic_tumor(size=256, tumor_type=tumor_type)
        
        # Save
        img_filename = f'{tumor_type}_tumor_{i:03d}.png'
        img_path = image_dir / img_filename
        mask_path = mask_dir / img_filename
        
        imsave(str(img_path), (image * 255).astype(np.uint8))
        imsave(str(mask_path), (mask * 255).astype(np.uint8))
        
        print(f"  ✓ Created {img_filename}")
    
    print(f"\n✅ Dataset created in '{output_dir}/'")
    print(f"   - {n_images} images in '{image_dir}'")
    print(f"   - {n_images} masks in '{mask_dir}'")
    
    return str(image_dir), str(mask_dir)


def preview_generated_images(image_dir='data/images', mask_dir='data/masks', n_preview=3):
    """
    Show a preview of generated images.
    """
    import tumor_segmentation as ts
    
    image_path = Path(image_dir)
    mask_path = Path(mask_dir)
    
    image_files = sorted(list(image_path.glob('*.png')))[:n_preview]
    
    fig, axes = plt.subplots(n_preview, 2, figsize=(10, 5*n_preview))
    if n_preview == 1:
        axes = axes.reshape(1, -1)
    
    for i, img_file in enumerate(image_files):
        # Load image and mask
        img = ts.load_grayscale_image(str(img_file))
        mask = ts.load_binary_mask(str(mask_path / img_file.name))
        
        # Plot
        axes[i, 0].imshow(img, cmap='gray')
        axes[i, 0].set_title(f'Image: {img_file.name}')
        axes[i, 0].axis('off')
        
        axes[i, 1].imshow(img, cmap='gray')
        axes[i, 1].imshow(mask, cmap='Reds', alpha=0.5)
        axes[i, 1].set_title('Ground Truth Overlay')
        axes[i, 1].axis('off')
    
    plt.suptitle('Generated Tumor Images Preview', fontsize=16, fontweight='bold')
    plt.tight_layout()
    plt.savefig('generated_images_preview.png', dpi=150, bbox_inches='tight')
    plt.show()
    
    print(f"\n✓ Preview saved as 'generated_images_preview.png'")


if __name__ == "__main__":
    print("="*70)
    print(" REALISTIC TUMOR IMAGE GENERATOR")
    print("="*70)
    
    # Ask user what type
    print("\nWhat type of tumors would you like to generate?")
    print("1. Brain tumors (MRI)")
    print("2. Liver tumors (CT)")
    print("3. Lung tumors (CT)")
    
    choice = input("\nEnter choice (1/2/3) or press Enter for brain [default=1]: ").strip()
    
    tumor_types = {
        '1': 'brain',
        '2': 'liver',
        '3': 'lung',
        '': 'brain'
    }
    
    tumor_type = tumor_types.get(choice, 'brain')
    
    # Ask how many
    n_str = input(f"\nHow many images to generate? [default=10]: ").strip()
    n_images = int(n_str) if n_str else 10
    
    # Generate
    print()
    img_dir, mask_dir = generate_tumor_dataset(n_images, 'data', tumor_type)
    
    # Preview
    print("\n" + "="*70)
    preview = input("\nShow preview? (y/n) [default=y]: ").strip().lower()
    if preview != 'n':
        preview_generated_images(img_dir, mask_dir, min(3, n_images))
    
    print("\n" + "="*70)
    print(" GENERATION COMPLETE!")
    print("="*70)
    print(f"\nYour images are ready in: data/")
    print(f"  - Images: data/images/")
    print(f"  - Masks: data/masks/")
    print("\nNext step: Run your tumor segmentation analysis!")
    print("  python -c \"import tumor_segmentation as ts; ts.main('data/images', 'data/masks', 'results')\"")
