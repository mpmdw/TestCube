"""
Rubik's Cube Face Segmentation Component

Takes an image containing a Rubik's cube face and outputs a 3x3 grid of
segmented 64x64 facelet images as a numpy array.

Output shape: (3, 3, 64, 64, 3)
    - First two dimensions: row, col position in the 3x3 grid
    - Next two dimensions: 64x64 pixel image
    - Last dimension: RGB/BGR color channels

Grid layout:
    [0,0] | [0,1] | [0,2]
    ----------------------
    [1,0] | [1,1] | [1,2]
    ----------------------
    [2,0] | [2,1] | [2,2]
"""

import cv2
import numpy as np
from typing import Optional
from dataclasses import dataclass


@dataclass
class BoundingBox:
    """Represents a bounding box for the cube face region."""
    x: int
    y: int
    width: int
    height: int


class FaceletSegmenter:
    """
    Segments a Rubik's cube face image into a 3x3 grid of facelet images.

    Usage:
        segmenter = FaceletSegmenter(output_size=64)
        facelets = segmenter.segment(image)
        # facelets.shape == (3, 3, 64, 64, 3)
        # Access center facelet: facelets[1, 1]
    """

    def __init__(self, output_size: int = 64):
        """
        Initialize the segmenter.

        Args:
            output_size: Size of output facelet images (default 64x64)
        """
        self.output_size = output_size

    def segment(
        self,
        image: np.ndarray,
        bbox: Optional[BoundingBox] = None
    ) -> np.ndarray:
        """
        Segment a cube face image into a 3x3 grid of facelets.

        Args:
            image: Input image (BGR format from OpenCV or RGB)
            bbox: Optional bounding box for the cube face region.
                  If None, attempts to auto-detect or uses full image.

        Returns:
            numpy array of shape (3, 3, 64, 64, 3)
            - result[row, col] gives the facelet at grid position (row, col)
            - result[0, 0] = top-left, result[1, 1] = center, result[2, 2] = bottom-right
        """
        if bbox is None:
            bbox = self._detect_face_region(image)

        # Extract the face region
        face_region = self._extract_region(image, bbox)

        # Split into 3x3 grid of facelets
        facelets = self._split_into_facelets(face_region)

        return facelets

    def _detect_face_region(self, image: np.ndarray) -> BoundingBox:
        """
        Attempt to auto-detect the cube face region in the image.

        This implementation tries to find the largest square-like contour.
        Falls back to using the largest centered square that fits.

        Args:
            image: Input image

        Returns:
            BoundingBox for the detected face region
        """
        height, width = image.shape[:2]

        # Convert to grayscale for edge detection
        if len(image.shape) == 3:
            gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        else:
            gray = image

        # Apply Gaussian blur to reduce noise
        blurred = cv2.GaussianBlur(gray, (5, 5), 0)

        # Edge detection
        edges = cv2.Canny(blurred, 50, 150)

        # Dilate edges to connect nearby lines
        kernel = np.ones((3, 3), np.uint8)
        edges = cv2.dilate(edges, kernel, iterations=2)

        # Find contours
        contours, _ = cv2.findContours(edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        best_bbox = None
        best_score = 0

        for contour in contours:
            # Approximate the contour to a polygon
            epsilon = 0.02 * cv2.arcLength(contour, True)
            approx = cv2.approxPolyDP(contour, epsilon, True)

            # Look for quadrilaterals (4 corners)
            if len(approx) == 4:
                x, y, w, h = cv2.boundingRect(approx)

                # Score based on size and squareness
                area = w * h
                squareness = min(w, h) / max(w, h) if max(w, h) > 0 else 0

                # Prefer larger, more square regions
                score = area * squareness

                if score > best_score and squareness > 0.7:
                    best_score = score
                    best_bbox = BoundingBox(x, y, w, h)

        # Fallback: use centered square region
        if best_bbox is None:
            # Use the largest centered square that fits
            size = min(width, height)
            margin = int(size * 0.1)  # 10% margin
            size -= 2 * margin

            x = (width - size) // 2
            y = (height - size) // 2
            best_bbox = BoundingBox(x, y, size, size)

        return best_bbox

    def _extract_region(self, image: np.ndarray, bbox: BoundingBox) -> np.ndarray:
        """
        Extract and square-crop the face region from the image.

        Args:
            image: Input image
            bbox: Bounding box for extraction

        Returns:
            Extracted region as numpy array
        """
        x, y, w, h = bbox.x, bbox.y, bbox.width, bbox.height

        # Ensure bounds are within image
        height, width = image.shape[:2]
        x = max(0, min(x, width - 1))
        y = max(0, min(y, height - 1))
        w = min(w, width - x)
        h = min(h, height - y)

        # Extract the region
        region = image[y:y+h, x:x+w]

        # Make it square by cropping to the smaller dimension
        min_dim = min(region.shape[0], region.shape[1])

        # Center crop to square
        start_y = (region.shape[0] - min_dim) // 2
        start_x = (region.shape[1] - min_dim) // 2

        square_region = region[start_y:start_y+min_dim, start_x:start_x+min_dim]

        return square_region

    def _split_into_facelets(self, face_region: np.ndarray) -> np.ndarray:
        """
        Split the square face region into a 3x3 grid of facelets.

        Args:
            face_region: Square image of the cube face

        Returns:
            numpy array of shape (3, 3, output_size, output_size, 3)
        """
        height, width = face_region.shape[:2]
        channels = face_region.shape[2] if len(face_region.shape) == 3 else 1

        # Calculate facelet dimensions
        facelet_h = height // 3
        facelet_w = width // 3

        # Pre-allocate output array
        facelets = np.zeros(
            (3, 3, self.output_size, self.output_size, channels),
            dtype=face_region.dtype
        )

        # Extract and resize each facelet
        for row in range(3):
            for col in range(3):
                y_start = row * facelet_h
                y_end = (row + 1) * facelet_h if row < 2 else height
                x_start = col * facelet_w
                x_end = (col + 1) * facelet_w if col < 2 else width

                facelet = face_region[y_start:y_end, x_start:x_end]

                # Resize to output size
                resized = cv2.resize(
                    facelet,
                    (self.output_size, self.output_size),
                    interpolation=cv2.INTER_AREA
                )

                facelets[row, col] = resized

        return facelets

    def segment_from_file(
        self,
        image_path: str,
        bbox: Optional[BoundingBox] = None
    ) -> np.ndarray:
        """
        Convenience method to segment directly from an image file.

        Args:
            image_path: Path to the input image
            bbox: Optional bounding box for the cube face region

        Returns:
            numpy array of shape (3, 3, 64, 64, 3)
        """
        image = cv2.imread(image_path)
        if image is None:
            raise ValueError(f"Could not load image from: {image_path}")
        return self.segment(image, bbox)

    def save_facelets(
        self,
        facelets: np.ndarray,
        output_dir: str,
        prefix: str = "facelet"
    ) -> list:
        """
        Save facelet images to files.

        Args:
            facelets: numpy array of shape (3, 3, 64, 64, 3)
            output_dir: Directory to save images
            prefix: Filename prefix (default "facelet")

        Returns:
            List of saved file paths (in row-major order)
        """
        import os
        os.makedirs(output_dir, exist_ok=True)

        saved_paths = []
        for row in range(3):
            for col in range(3):
                idx = row * 3 + col
                path = os.path.join(output_dir, f"{prefix}_{row}_{col}.png")
                cv2.imwrite(path, facelets[row, col])
                saved_paths.append(path)

        return saved_paths


def segment_cube_face(
    image: np.ndarray,
    bbox: Optional[BoundingBox] = None,
    output_size: int = 64
) -> np.ndarray:
    """
    Functional interface for facelet segmentation.

    Args:
        image: Input image containing a Rubik's cube face
        bbox: Optional bounding box for the face region
        output_size: Size of output facelet images (default 64)

    Returns:
        numpy array of shape (3, 3, output_size, output_size, 3)
    """
    segmenter = FaceletSegmenter(output_size=output_size)
    return segmenter.segment(image, bbox)
