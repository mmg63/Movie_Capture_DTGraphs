
import os
import numpy as np
import torch
from PIL import Image
import jpype
from jpype.types import JArray, JDouble
import cv2

from utils import reduce_superpixels_via_cosine_merge

# === Java Integration ===
if not jpype.isJVMStarted():
    jpype.startJVM(classpath=[os.path.abspath("graphbuilder.jar")])

Segmenter = jpype.JClass("graphbuilder.Segmenter")
DelaunayGraphBuilder = jpype.JClass("graphbuilder.DelaunayGraphBuilder")
ImageUtils = jpype.JClass("graphbuilder.ImageUtils")
GraphData = jpype.JClass("graphbuilder.GraphData")
RGBToLabConverter = jpype.JClass("snic.RGBToLabConverter")

# === Load input image path from environment variable ===
input_path = os.getenv("INPUT_PATH", "./Captured movies and processed/test_image.png")

if not os.path.exists(input_path):
    raise FileNotFoundError(f"Input image not found at {input_path}")

# === Load and process image ===
image = cv2.imread(input_path)
if image is None:
    raise ValueError("Failed to load image")

height, width, _ = image.shape
flat_image = image.reshape(-1, 3)

# Convert RGB to LAB using Java helper
rgb2lab = RGBToLabConverter()
lab_flat = rgb2lab.convert(flat_image[:, 0], flat_image[:, 1], flat_image[:, 2])

# Reshape LAB to original image shape
lab_image = np.stack(lab_flat).reshape(3, height, width)

# Call your superpixel processing logic here
print("[INFO] Image loaded and processed into LAB format.")

# === Example call to your custom logic ===
# You can plug your ViT or GNN model here if needed
# reduced_result = reduce_superpixels_via_cosine_merge(...)

print("[INFO] Processing completed for:", input_path)
