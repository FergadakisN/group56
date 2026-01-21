"""
Image feature extraction for data drift detection.

Since data drift detection tools like Evidently work on structured/tabular data,
we need to extract numerical features from images. This module extracts features
such as brightness, contrast, sharpness, and color statistics that can be used
to detect distribution shifts in image data over time.
"""

from __future__ import annotations

import numpy as np
from PIL import Image, ImageStat, ImageFilter


def extract_image_features(image: Image.Image) -> dict[str, float]:
    """
    Extract structured features from an image for drift detection.

    Features extracted:
    - mean_brightness: Average pixel intensity across all channels
    - std_brightness: Standard deviation of pixel intensity
    - contrast: Difference between max and min pixel values
    - sharpness: Laplacian variance (measure of edge strength)
    - red_mean, green_mean, blue_mean: Average values per color channel
    - red_std, green_std, blue_std: Standard deviation per color channel
    - aspect_ratio: Width / Height ratio
    - file_size_kb: Approximate file size

    Args:
        image: PIL Image object (RGB)

    Returns:
        Dictionary of feature name to float value
    """
    # Ensure RGB
    if image.mode != "RGB":
        image = image.convert("RGB")

    # Get image statistics
    stat = ImageStat.Stat(image)

    # Mean and std for each channel
    red_mean, green_mean, blue_mean = stat.mean
    red_std, green_std, blue_std = stat.stddev

    # Overall brightness (average across channels)
    mean_brightness = sum(stat.mean) / len(stat.mean)
    std_brightness = sum(stat.stddev) / len(stat.stddev)

    # Convert to numpy for additional calculations
    img_array = np.array(image)

    # Contrast: difference between max and min intensities
    contrast = float(img_array.max() - img_array.min())

    # Sharpness: use Laplacian filter to measure edge strength
    gray_image = image.convert("L")
    laplacian = gray_image.filter(ImageFilter.FIND_EDGES)
    laplacian_array = np.array(laplacian)
    sharpness = float(np.var(laplacian_array))

    # Aspect ratio
    width, height = image.size
    aspect_ratio = width / height if height > 0 else 1.0

    # Approximate file size (number of pixels * 3 bytes per pixel / 1024)
    approx_size_kb = (width * height * 3) / 1024

    return {
        "mean_brightness": mean_brightness,
        "std_brightness": std_brightness,
        "contrast": contrast,
        "sharpness": sharpness,
        "red_mean": red_mean,
        "green_mean": green_mean,
        "blue_mean": blue_mean,
        "red_std": red_std,
        "green_std": green_std,
        "blue_std": blue_std,
        "aspect_ratio": aspect_ratio,
        "approx_size_kb": approx_size_kb,
    }


def features_to_csv_row(features: dict[str, float], timestamp: str, prediction: str) -> str:
    """
    Convert features to CSV row format.

    Args:
        features: Dictionary from extract_image_features()
        timestamp: ISO format timestamp string
        prediction: Predicted class name

    Returns:
        Comma-separated string ready for CSV file
    """
    values = [
        timestamp,
        str(features["mean_brightness"]),
        str(features["std_brightness"]),
        str(features["contrast"]),
        str(features["sharpness"]),
        str(features["red_mean"]),
        str(features["green_mean"]),
        str(features["blue_mean"]),
        str(features["red_std"]),
        str(features["green_std"]),
        str(features["blue_std"]),
        str(features["aspect_ratio"]),
        str(features["approx_size_kb"]),
        prediction,
    ]
    return ",".join(values)


def get_csv_header() -> str:
    """Get CSV header for prediction database."""
    return (
        "timestamp,mean_brightness,std_brightness,contrast,sharpness,"
        "red_mean,green_mean,blue_mean,red_std,green_std,blue_std,"
        "aspect_ratio,approx_size_kb,prediction"
    )
