"""
Test script for the FaceletSegmenter component.

This script demonstrates usage and includes a synthetic test image generator.
"""

import cv2
import numpy as np
import os
from facelet_segmenter import FaceletSegmenter, BoundingBox, segment_cube_face


def create_synthetic_cube_face(
    image_size: tuple = (480, 640),
    face_size: int = 300,
    offset: tuple = (170, 90)
) -> np.ndarray:
    """
    Create a synthetic Rubik's cube face image for testing.

    Args:
        image_size: (height, width) of the output image
        face_size: Size of the cube face in pixels
        offset: (x, y) offset of the face from top-left

    Returns:
        Synthetic image with a colored 3x3 grid
    """
    # Rubik's cube standard colors (BGR format for OpenCV)
    colors = {
        'white': (255, 255, 255),
        'yellow': (0, 255, 255),
        'red': (0, 0, 255),
        'orange': (0, 165, 255),
        'blue': (255, 0, 0),
        'green': (0, 255, 0),
    }

    # Create a scrambled face pattern (3x3 grid)
    color_grid = [
        ['red', 'blue', 'yellow'],
        ['green', 'white', 'orange'],
        ['yellow', 'red', 'blue']
    ]

    # Create background image (dark gray)
    image = np.full((image_size[0], image_size[1], 3), 50, dtype=np.uint8)

    # Calculate facelet size
    facelet_size = face_size // 3
    gap = 4  # Gap between facelets

    # Draw each facelet
    for row in range(3):
        for col in range(3):
            color = colors[color_grid[row][col]]

            x = offset[0] + col * facelet_size + gap
            y = offset[1] + row * facelet_size + gap

            # Draw filled rectangle
            cv2.rectangle(
                image,
                (x, y),
                (x + facelet_size - 2 * gap, y + facelet_size - 2 * gap),
                color,
                -1  # Filled
            )

            # Draw border
            cv2.rectangle(
                image,
                (x, y),
                (x + facelet_size - 2 * gap, y + facelet_size - 2 * gap),
                (0, 0, 0),
                2  # Border thickness
            )

    # Draw outer border of the cube face
    cv2.rectangle(
        image,
        (offset[0], offset[1]),
        (offset[0] + face_size, offset[1] + face_size),
        (30, 30, 30),
        3
    )

    return image


def visualize_facelets(facelets: np.ndarray, title: str = "Segmented Facelets") -> np.ndarray:
    """
    Create a visualization of the 3x3 facelet grid.

    Args:
        facelets: numpy array of shape (3, 3, 64, 64, 3)
        title: Title for the visualization

    Returns:
        Combined visualization image
    """
    # Get facelet size
    size = facelets.shape[2]
    gap = 5

    # Create output image
    total_size = 3 * size + 4 * gap
    viz = np.full((total_size + 30, total_size, 3), 255, dtype=np.uint8)

    # Place each facelet
    for row in range(3):
        for col in range(3):
            y = gap + row * (size + gap)
            x = gap + col * (size + gap)
            viz[y:y+size, x:x+size] = facelets[row, col]

    # Add title
    cv2.putText(viz, title, (10, total_size + 22),
                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 1)

    return viz


def test_basic_segmentation():
    """Test basic segmentation with a synthetic image."""
    print("=" * 50)
    print("Test: Basic Segmentation")
    print("=" * 50)

    # Create synthetic image
    image = create_synthetic_cube_face()
    print(f"Created synthetic image: {image.shape}")

    # Create segmenter
    segmenter = FaceletSegmenter(output_size=64)

    # Segment with known bounding box
    bbox = BoundingBox(x=170, y=90, width=300, height=300)
    facelets = segmenter.segment(image, bbox=bbox)

    print(f"Output shape: {facelets.shape}")
    print(f"  - 3x3 grid of facelets")
    print(f"  - Each facelet: {facelets[0, 0].shape}")

    # Verify output shape
    assert facelets.shape == (3, 3, 64, 64, 3), \
        f"Expected shape (3, 3, 64, 64, 3), got {facelets.shape}"

    # Test accessing individual facelets
    top_left = facelets[0, 0]
    center = facelets[1, 1]
    bottom_right = facelets[2, 2]
    print(f"  - Top-left [0,0]: {top_left.shape}")
    print(f"  - Center [1,1]: {center.shape}")
    print(f"  - Bottom-right [2,2]: {bottom_right.shape}")

    print("PASSED: Basic segmentation test")
    return image, facelets


def test_auto_detection():
    """Test auto-detection of the cube face region."""
    print("\n" + "=" * 50)
    print("Test: Auto Detection")
    print("=" * 50)

    # Create synthetic image
    image = create_synthetic_cube_face()

    # Create segmenter
    segmenter = FaceletSegmenter(output_size=64)

    # Segment without bounding box (auto-detect)
    facelets = segmenter.segment(image)

    print(f"Auto-detected and segmented")
    print(f"Output shape: {facelets.shape}")

    assert facelets.shape[:2] == (3, 3), "Should produce 3x3 grid"

    print("PASSED: Auto detection test")
    return facelets


def test_functional_interface():
    """Test the functional interface."""
    print("\n" + "=" * 50)
    print("Test: Functional Interface")
    print("=" * 50)

    # Create synthetic image
    image = create_synthetic_cube_face()

    # Use functional interface
    facelets = segment_cube_face(image, output_size=64)

    print(f"Functional interface output shape: {facelets.shape}")

    assert facelets.shape == (3, 3, 64, 64, 3), "Should produce (3, 3, 64, 64, 3)"

    print("PASSED: Functional interface test")
    return facelets


def test_different_output_sizes():
    """Test with different output sizes."""
    print("\n" + "=" * 50)
    print("Test: Different Output Sizes")
    print("=" * 50)

    image = create_synthetic_cube_face()

    for size in [32, 64, 128]:
        segmenter = FaceletSegmenter(output_size=size)
        bbox = BoundingBox(x=170, y=90, width=300, height=300)
        facelets = segmenter.segment(image, bbox=bbox)

        expected_shape = (3, 3, size, size, 3)
        assert facelets.shape == expected_shape, \
            f"Expected {expected_shape}, got {facelets.shape}"
        print(f"  Output size {size}x{size}: shape {facelets.shape} OK")

    print("PASSED: Different output sizes test")


def test_save_facelets():
    """Test saving facelets to disk."""
    print("\n" + "=" * 50)
    print("Test: Save Facelets")
    print("=" * 50)

    image = create_synthetic_cube_face()
    segmenter = FaceletSegmenter(output_size=64)
    bbox = BoundingBox(x=170, y=90, width=300, height=300)
    facelets = segmenter.segment(image, bbox=bbox)

    # Save to output directory
    output_dir = "test_output"
    saved_paths = segmenter.save_facelets(facelets, output_dir, prefix="test_facelet")

    print(f"Saved {len(saved_paths)} facelets to {output_dir}/")
    for path in saved_paths:
        assert os.path.exists(path), f"File should exist: {path}"
        print(f"  - {path}")

    print("PASSED: Save facelets test")
    return output_dir


def main():
    """Run all tests and create demonstration outputs."""
    print("Rubik's Cube Facelet Segmenter - Test Suite")
    print("=" * 50)

    # Run tests
    original_image, facelets = test_basic_segmentation()
    test_auto_detection()
    test_functional_interface()
    test_different_output_sizes()
    output_dir = test_save_facelets()

    # Create visualization
    print("\n" + "=" * 50)
    print("Creating Visualizations")
    print("=" * 50)

    # Save original synthetic image
    cv2.imwrite(os.path.join(output_dir, "original_synthetic.png"), original_image)
    print(f"Saved original image to {output_dir}/original_synthetic.png")

    # Save visualization
    viz = visualize_facelets(facelets)
    cv2.imwrite(os.path.join(output_dir, "facelets_visualization.png"), viz)
    print(f"Saved visualization to {output_dir}/facelets_visualization.png")

    print("\n" + "=" * 50)
    print("All tests PASSED!")
    print("=" * 50)

    # Print usage example
    print("\nUsage Example:")
    print("-" * 50)
    print("""
from facelet_segmenter import FaceletSegmenter, BoundingBox

# Create segmenter
segmenter = FaceletSegmenter(output_size=64)

# Segment an image
facelets = segmenter.segment_from_file("cube_face.jpg")
# facelets.shape == (3, 3, 64, 64, 3)

# Access individual facelets by grid position:
top_left = facelets[0, 0]      # shape: (64, 64, 3)
center = facelets[1, 1]        # shape: (64, 64, 3)
bottom_right = facelets[2, 2]  # shape: (64, 64, 3)

# Grid layout:
#   [0,0] | [0,1] | [0,2]
#   ----------------------
#   [1,0] | [1,1] | [1,2]
#   ----------------------
#   [2,0] | [2,1] | [2,2]

# Save individual facelets
segmenter.save_facelets(facelets, "output/", prefix="facelet")
# Creates: output/facelet_0_0.png through output/facelet_2_2.png
""")


if __name__ == "__main__":
    main()
