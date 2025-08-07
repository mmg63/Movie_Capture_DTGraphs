
# import os
# import numpy as np
# import torch
# from PIL import Image
# import jpype
# from jpype.types import JArray, JDouble
# import cv2

# from utils import reduce_superpixels_via_cosine_merge

# # === Java Integration ===
# if not jpype.isJVMStarted():
#     jpype.startJVM(classpath=[os.path.abspath("graphbuilder.jar")])

# Segmenter = jpype.JClass("graphbuilder.Segmenter")
# DelaunayGraphBuilder = jpype.JClass("graphbuilder.DelaunayGraphBuilder")
# ImageUtils = jpype.JClass("graphbuilder.ImageUtils")
# GraphData = jpype.JClass("graphbuilder.GraphData")
# RGBToLabConverter = jpype.JClass("snic.RGBToLabConverter")

# # === Load input image path from environment variable ===
# input_path = os.getenv("INPUT_PATH", "./Captured movies and processed/test_image.png")

# if not os.path.exists(input_path):
#     raise FileNotFoundError(f"Input image not found at {input_path}")

# # === Load and process image ===
# image = cv2.imread(input_path)
# if image is None:
#     raise ValueError("Failed to load image")

# height, width, _ = image.shape
# flat_image = image.reshape(-1, 3)

# # Convert RGB to LAB using Java helper
# rgb2lab = RGBToLabConverter()
# lab_flat = rgb2lab.convert(flat_image[:, 0], flat_image[:, 1], flat_image[:, 2])

# # Reshape LAB to original image shape
# lab_image = np.stack(lab_flat).reshape(3, height, width)

# # Call your superpixel processing logic here
# print("[INFO] Image loaded and processed into LAB format.")

# # === Example call to your custom logic ===
# # You can plug your ViT or GNN model here if needed
# # reduced_result = reduce_superpixels_via_cosine_merge(...)

# print("[INFO] Processing completed for:", input_path)



import jpype
from jpype.types import JArray, JDouble
import numpy as np
import torch
from torch_geometric.data import Data
from multiprocessing import Pool, cpu_count
from PIL import Image
import cv2
import os
import time

from utils import reduce_superpixels_via_cosine_merge

# utilizing the ViT model for superpixel segmentation
# and Delaunay triangulation for graph construction
# Ensure you have the required Java classes in the classpath
# graphbuilder.Segmenter, graphbuilder.DelaunayGraphBuilder,
# graphbuilder.ImageUtils, graphbuilder.GraphData, snic.RGBToLabConverter
import timm
import torch.nn.functional as F
import torchvision.transforms as T


# vit_model_name = 'vit_base_patch16_224'
# vit_model = timm.create_model(vit_model_name, pretrained=True)
# vit_model.eval()
# -------------------------------------
#  use for facial recognition or other tasks
# from transformers import AutoImageProcessor, AutoModelForImageClassification
# import torch.nn.functional as F
# processor = AutoImageProcessor.from_pretrained('trpakov/vit-face-expression')
# fer_vit = AutoModelForImageClassification.from_pretrained('trpakov/vit-face-expression')
# fer_vit.eval()
# --------------------------------------

# Set the number of Delaunay nodes (superpixels)
import threading
delaunayNodes_k = 64
superpixel_compactness = 50.0
k_lock = threading.Lock()
c_lock = threading.Lock()


# Start JVM
if not jpype.isJVMStarted():
    jpype.startJVM(classpath=[os.path.abspath(".")])


# === Java Imports ===
Segmenter = jpype.JClass("graphbuilder.Segmenter")
DelaunayGraphBuilder = jpype.JClass("graphbuilder.DelaunayGraphBuilder")
ImageUtils = jpype.JClass("graphbuilder.ImageUtils")
GraphData = jpype.JClass("graphbuilder.GraphData")
RGBToLabConverter = jpype.JClass("snic.RGBToLabConverter")


def read_config_from_input():
    global delaunayNodes_k, superpixel_compactness
    while True:
        try:
            user_input = input("Enter 'k=...' or 'c=...': ").strip()
            if user_input.startswith("k="):
                value = user_input.split("=")[1]
                if value.isdigit():
                    with k_lock:
                        delaunayNodes_k = int(value)
                    print(f"[INFO] Updated k to {delaunayNodes_k}")
            elif user_input.startswith("c="):
                value = user_input.split("=")[1]
                compact = float(value)
                with c_lock:
                    superpixel_compactness = compact
                print(f"[INFO] Updated compactness to {superpixel_compactness}")
            else:
                print("[WARN] Use format 'k=...' or 'c=...' only.")
        except Exception as e:
            print(f"[ERROR] Invalid input: {e}")


def java_image_to_numpy(image_path):
    img = Image.open(image_path).convert("RGB")
    np_img = np.array(img)

    H, W, C = np_img.shape
    size = H * W

    r = np_img[:, :, 0].astype(np.float64).flatten()
    g = np_img[:, :, 1].astype(np.float64).flatten()
    b = np_img[:, :, 2].astype(np.float64).flatten()

    # Create Java arrays
    jr = jpype.JArray(jpype.JDouble)(r.tolist())
    jg = jpype.JArray(jpype.JDouble)(g.tolist())
    jb = jpype.JArray(jpype.JDouble)(b.tolist())

    jl = jpype.JArray(jpype.JDouble)([0.0] * size)
    ja = jpype.JArray(jpype.JDouble)([0.0] * size)
    jb_ = jpype.JArray(jpype.JDouble)([0.0] * size)

    # Call SNIC lab converter
    RGBToLabConverter.convert(jr, jg, jb, size, jl, ja, jb_)

    lab = np.zeros((H, W, 3), dtype=np.float64)
    lab[:, :, 0] = np.reshape(jl, (H, W)) / 100.0
    lab[:, :, 1] = (np.reshape(ja, (H, W)) + 128.0) / 255.0
    lab[:, :, 2] = (np.reshape(jb_, (H, W)) + 128.0) / 255.0

    return lab

def convert_image_to_graph(image, k=16, compactness=50.0):
    try:
        # img = Image.open(image_path).convert("RGB")
        np_img = np.array(image).astype(np.float64)  # shape [H, W, 3]

        H, W, C = np_img.shape
        jimg = JArray(JArray(JArray(JDouble)))([
        [JArray(JDouble)(np_img[y][x].tolist()) for x in range(W)]
        for y in range(H)
        ])

        seg = Segmenter.segment(jimg, k, compactness, True)
        # seg = Segmenter.segment(jimg, 16, 0.5, True)
        graph = DelaunayGraphBuilder.build(seg)

        # Convert GraphData back to Python
        x = np.array(graph.x)
        pos = np.array(graph.pos)
        edge_index = np.array(graph.edgeIndex)

        data = Data(
            x=torch.tensor(x, dtype=torch.float),
            edge_index=torch.tensor(edge_index, dtype=torch.long),
            pos=torch.tensor(pos, dtype=torch.float)
        )

        # Placeholder label map (you can modify later)
        label_map = np.array(seg.labels, dtype=np.int64).reshape(H, W)
        image_name = 'frame'
        num_nodes = x.shape[0]

        return (data, label_map, image_name, num_nodes), seg

    except Exception as e:
        print(f"Error processing {image}: {e}")
        return None


def colorize_label_map(label_map, num_labels=None, seed=42):
    """
    Colorizes a superpixel label map (2D array) with random colors.
    
    Args:
        label_map (np.ndarray): 2D array of superpixel labels [H x W]
        num_labels (int): optionally pass max label + 1 if known
        seed (int): for reproducibility
    
    Returns:
        np.ndarray: RGB image [H x W x 3] with colored superpixels
    """
    unique_labels = np.unique(label_map)
    max_label = num_labels if num_labels is not None else unique_labels.max() + 1

    # Generate reproducible random colors
    rng = np.random.default_rng(seed)
    palette = rng.integers(0, 256, size=(max_label, 3), dtype=np.uint8)

    # Apply palette
    colored = palette[label_map.astype(np.int32)]
    return colored


def visualize_superpixel_means(label_map, data_x):
    """
    Given a label map and a feature matrix (mean values per superpixel), generate an RGB image.

    Args:
        label_map (np.ndarray): 2D array (H x W) of superpixel labels
        data_x (torch.Tensor or np.ndarray): shape (num_superpixels, 3), RGB means

    Returns:
        np.ndarray: H x W x 3 uint8 image
    """
    # Convert torch tensor to numpy if needed
    if isinstance(data_x, torch.Tensor):
        data_x = data_x[:, 0:3].cpu().numpy()

    # Clip to [0, 255] and convert to uint8 if values are in [0, 1]
    if data_x.max() <= 1.0:
        data_x = (data_x * 255).astype(np.uint8)
    else:
        data_x = data_x.astype(np.uint8)

    # Map each label to its corresponding RGB mean
    color_image = data_x[label_map]

    return color_image


def visualize_superpixel_means(label_map, data_x):
    if isinstance(data_x, torch.Tensor):
        data_x = data_x.cpu().numpy()

    if data_x.max() <= 1.0:
        data_x = (data_x * 255).astype(np.uint8)
    else:
        data_x = data_x.astype(np.uint8)

    return data_x[label_map]

def draw_graph_on_image(image, data, node_color=(0, 255, 0), edge_color=(255, 0, 0)):
    image = image.copy()
    pos = data.pos.cpu().numpy().astype(np.int32)
    edge_index = data.edge_index.cpu().numpy()

    for i in range(edge_index.shape[1]):
        pt1 = tuple(pos[edge_index[0, i]])
        pt2 = tuple(pos[edge_index[1, i]])
        cv2.line(image, pt1, pt2, edge_color, 1, lineType=cv2.LINE_AA)

    for p in pos:
        cv2.circle(image, tuple(p), 2, node_color, -1)

    return image


if __name__ == "__main__":

    # webcam setup
    # cap = cv2.VideoCapture(0)
    
    # read the movie.mov file using opencv
    cap = cv2.VideoCapture('Captured movies and processed/sample_movie.mov')
    if not cap.isOpened():
        print("Error: Could not open video.")
        exit()
    
    fps = cap.get(cv2.CAP_PROP_FPS)
    print(f"[INFO] Video FPS: {fps}")

    skip_frame = fps / 5  # target 5 FPS processing
    if skip_frame < 1:
        skip_frame = 1

    resize_width, resize_height = 320, 320


    # Set initial Delaunay Nodes count using multithreading
    input_thread = threading.Thread(target=read_config_from_input, daemon=True)
    input_thread.start()


    # ----------- vit section -----------
    vit_transform = T.Compose([
        T.Resize((224, 224)),
        T.ToTensor(),
        T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])
    # -----------------------------------

    frame_id = 0
    
    while True:

        cv2.skipFrame = int(cap.get(cv2.CAP_PROP_FPS) / skip_frame)
        
        cap.set(cv2.CAP_PROP_POS_FRAMES, frame_id * skip_frame)  # set to the target frame position
        print(f"[INFO] Target FPS: {skip_frame}, Frame skip: {cv2.skipFrame}")
        ret, frame = cap.read()
        if not ret:
            break
        frame_id += 1
        print(f"[INFO] Processing frame {frame_id}")

        # Display original frame
        frame_Raw_resized = cv2.resize(frame,(resize_width, resize_height))# (640, 480))
        cv2.imwrite(f'Captured movies and processed/OriginalFrame/Original Webcam Feed Resized_{frame_id}.jpg', frame_Raw_resized)
        # cv2.imshow("Original Webcam Feed", frame_Raw_resized)

        # Resize + Process
        frame_resized = cv2.resize(frame, (resize_width, resize_height))
        with k_lock:
            current_k = delaunayNodes_k
        with c_lock:
            current_c = superpixel_compactness
        graph_data, segment = convert_image_to_graph(frame_resized, k=current_k, compactness=current_c)

        # reduce the number of superpixels based on the cosine similarity
        H, W = frame_resized.shape[:2]
        label_map = np.array(segment.labels, dtype=np.int32).reshape(H, W)
        kc_flat = np.array(segment.kc_flat).reshape((segment.numSuperpixels, -1))
        kx = np.array(segment.kx)
        ky = np.array(segment.ky)

        new_label_map, new_kc, new_kx, new_ky = reduce_superpixels_via_cosine_merge(
            label_map, kc_flat, kx, ky, target_P=8)
        # ----------------------------------------------------------------

        if graph_data is not None:
            data, label_map, image_name, num_nodes = graph_data


        # label_map = output of your segmentation model (values 0 to k)
        colored_labelmap = colorize_label_map(label_map)
        # graph_img = draw_graph_on_image(colored_superpixels, data)
        cv2.imwrite(f'Captured movies and processed/LabelMap/Label Map Resized_{frame_id}.jpg', colored_labelmap)
        # cv2.imshow("Label map", colored_labelmap)


        # New reduced label_map = output of your segmentation model (values 0 to k)
        colored_new_labelmap = colorize_label_map(new_label_map)
        # graph_img = draw_graph_on_image(colored_superpixels, data)
        cv2.imwrite(f'Captured movies and processed/NewLabelMap/New Label Map Resized_{frame_id}.jpg', colored_new_labelmap)
        # cv2.imshow("New Reduced Label map", colored_new_labelmap)

        overlay = cv2.addWeighted(frame_resized, 0.7, colored_labelmap, 0.2, 0)
        graph_img = draw_graph_on_image(overlay, data)
        cv2.imwrite(f'Captured movies and processed/Overlay/Overlay Resized_{frame_id}.jpg', graph_img)
        # cv2.imshow("Label map with overlay", graph_img)

        colorized_means = visualize_superpixel_means(label_map, data.x[:, 0:3])
        graph_img = draw_graph_on_image(colorized_means, data)
        cv2.imwrite(f'Captured movies and processed/Means/Means Resized_{frame_id}.jpg', graph_img)
        # cv2.imshow("Delaunay's nodes values Colors", graph_img)


    cap.release()
    cv2.destroyAllWindows()
