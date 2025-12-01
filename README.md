# TestCube - Rubik's Cube Facelet Segmenter

A component that takes an image containing a Rubik's cube face and outputs a `(3, 3, 64, 64, 3)` numpy array of segmented facelet images.

## Overview

This component processes a camera/image input (typically 640x480) containing a Rubik's cube face and:

1. Detects or accepts a bounding box for the cube face region
2. Extracts and squares the face region
3. Divides it into a 3x3 grid of 9 facelets
4. Outputs a numpy array of shape `(3, 3, 64, 64, 3)` ready for the next processing stage

## Output Format

**Shape:** `(3, 3, 64, 64, 3)`
- First two dimensions: row, col position in the 3x3 grid
- Next two dimensions: 64x64 pixel image
- Last dimension: RGB/BGR color channels

**Grid layout:**
```
[0,0] | [0,1] | [0,2]
----------------------
[1,0] | [1,1] | [1,2]
----------------------
[2,0] | [2,1] | [2,2]
```

## Installation

```bash
pip install -r requirements.txt
```

## Usage

### Basic Usage

```python
from facelet_segmenter import FaceletSegmenter, BoundingBox

# Create segmenter (outputs 64x64 images by default)
segmenter = FaceletSegmenter(output_size=64)

# Segment an image file
facelets = segmenter.segment_from_file("cube_face.jpg")
# facelets.shape == (3, 3, 64, 64, 3)

# Access individual facelets by grid position
top_left = facelets[0, 0]      # shape: (64, 64, 3)
center = facelets[1, 1]        # shape: (64, 64, 3)
bottom_right = facelets[2, 2]  # shape: (64, 64, 3)

# With explicit bounding box
bbox = BoundingBox(x=100, y=50, width=300, height=300)
facelets = segmenter.segment_from_file("cube_face.jpg", bbox=bbox)

# Process numpy array directly
import cv2
image = cv2.imread("cube_face.jpg")
facelets = segmenter.segment(image, bbox=bbox)
```

### Functional Interface

```python
from facelet_segmenter import segment_cube_face

# Simple one-liner
facelets = segment_cube_face(image, output_size=64)
# Returns: numpy array of shape (3, 3, 64, 64, 3)
```

### Saving Results

```python
# Save all facelets to a directory
saved_paths = segmenter.save_facelets(facelets, "output/", prefix="facelet")
# Creates: output/facelet_0_0.png through output/facelet_2_2.png
```

## API Reference

### FaceletSegmenter

```python
class FaceletSegmenter:
    def __init__(self, output_size: int = 64)
    def segment(self, image: np.ndarray, bbox: Optional[BoundingBox] = None) -> np.ndarray
        # Returns: shape (3, 3, output_size, output_size, 3)
    def segment_from_file(self, image_path: str, bbox: Optional[BoundingBox] = None) -> np.ndarray
    def save_facelets(self, facelets: np.ndarray, output_dir: str, prefix: str = "facelet") -> List[str]
```

### BoundingBox

```python
@dataclass
class BoundingBox:
    x: int      # Left edge x coordinate
    y: int      # Top edge y coordinate
    width: int  # Width of the region
    height: int # Height of the region
```

## Testing

Run the test suite:

```bash
python test_segmenter.py
```

This creates a `test_output/` directory with:
- `original_synthetic.png` - A synthetic test image
- `test_facelet_0.png` through `test_facelet_8.png` - Individual facelets
- `facelets_visualization.png` - Visual grid of all 9 facelets

## Integration

This component is designed to fit into a processing pipeline:

```
Camera/Image Input (640x480)
         |
         v
  FaceletSegmenter
         |
         v
   9 x 64x64 images
         |
         v
  Color Recognition
  (next component)
```
