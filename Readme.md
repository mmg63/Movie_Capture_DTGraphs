# Real-Time Webcam Superpixel Graph Processor with Dynamic Controls

This project is a real-time, webcam-based computer vision system that performs superpixel segmentation and graph construction using a hybrid of Java and Python technologies. It integrates Java-based SNIC segmentation and Delaunay triangulation with PyTorch Geometric to build graphs on-the-fly, offering a highly interactive and visual experience for graph-based image processing.

It is particularly suited for researchers, students, or developers exploring graph neural networks, superpixel techniques, or real-time visual processing.

---

## 🚀 Features

* 📷 Real-time webcam frame capture using OpenCV
* 🔄 Live SNIC (Simple Non-Iterative Clustering) segmentation via Java integration
* 🔺 Delaunay triangulation-based graph construction from superpixels
* 📊 PyTorch Geometric `Data` object construction from visual features
* 🎨 Live visualization of:

  * Superpixel label maps
  * Mean RGB color reconstructions of superpixels
  * Graph overlays (nodes + edges)
* 🎛️ Terminal-based live parameter adjustment:

  * `k=128` → updates the number of superpixels
  * `c=25.0` → updates the SNIC compactness parameter
* ⚙️ Multi-threaded input system so your GUI remains fully responsive

---

## 📚 Background Concepts

### SNIC (Simple Non-Iterative Clustering)

SNIC is an efficient superpixel segmentation algorithm that clusters pixels based on both color and spatial proximity. It is a variant of SLIC (Simple Linear Iterative Clustering) that avoids iterative refinement, making it faster and suitable for real-time use. SNIC produces label maps where each region (superpixel) is roughly uniform in appearance and compact in shape.

### Delaunay Triangulation

Delaunay triangulation is a geometric technique used to connect a set of points into triangles such that no point lies inside the circumcircle of any triangle. In this project, the centroids of superpixels are used as nodes, and edges are constructed using Delaunay triangulation. This forms a well-shaped planar graph suitable for message passing in GNNs.

---

## 📦 Requirements

* Python 3.8+
* Java 21 (for SNIC and Delaunay Java classes)
* JPype 1.5+
* PyTorch & PyTorch Geometric
* OpenCV
* Pillow
* NumPy

Install dependencies:

```bash
pip install torch torchvision torchaudio
pip install torch-geometric
pip install opencv-python pillow jpype1 numpy
```

---

## 🛠 Folder Structure

```
project_root/
├── main.py                         # Main real-time processing script
├── graphbuilder/                  # Java compiled classes (Segmenter, GraphBuilder, etc.)
├── snic/                          # Java SNIC converter classes
├── graphbuilder.jar               # Optional Java JAR containing the classes
├── README.md                      # Project documentation
```

---

## ▶️ Running the App

```bash
python main.py
```

While the app is running, open your terminal and type:

```bash
k=128      # Update superpixel count
c=30.0     # Update SNIC compactness
```

Press `q` in any OpenCV window to quit.

---

## 📌 Reference Publication

This system was developed as part of our research on graph neural networks applied to visual data. It was published in the 38th Canadian Conference on Artificial Intelligence:

> **"Fast Graph Neural Network for Image Classification"**
> Mustafa Mohammadi Gharasuie, 2025
> 📄 [Read the paper here](https://caiac.pubpub.org/pub/u1wgain4/release/1)

This paper introduces a fast and effective graph construction pipeline for image classification using Delaunay-based topology and SNIC-generated superpixels. The real-time pipeline presented here is based on that framework.

---

## 🧠 Powered By

* **JPype** – Java-Python bridge
* **Java SNIC** – For real-time, non-iterative superpixel segmentation
* **DelaunayGraphBuilder** – Java implementation of triangulated graph generation
* **PyTorch Geometric** – Efficient graph representation and modeling
* **OpenCV** – Camera feed capture and window visualization

---

## 📈 Roadmap

* [x] Real-time graph construction from webcam frames
* [x] Live parameter editing via threaded input (`k`, `compactness`)
* [ ] 🔜 Integration of Vision Transformers (ViT) for semantic analysis
* [ ] 🔜 Save graph/image/frame outputs
* [ ] 🔜 Streamlit or HuggingFace interface for demo and cloud access

---

## 📝 License

This project is currently released under the MIT License. Please ensure you comply with any licensing terms of the Java libraries used if distributing this project.

---

## 🙌 Contributions

Feel free to fork, adapt, or improve this project for your own use cases. PRs and suggestions are welcome!
