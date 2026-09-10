## Application of Hybrid U-Net Model in Skin Lesion Boundary Segmentation
A deep learning computer vision pipeline built with PyTorch to automate the analysis and semantic segmentation of skin lesion images for medical diagnosis and skin cancer detection.

🔍 Overview
Skin lesion segmentation is a critical step in computer-aided diagnosis systems for skin cancer. This project implements a robust deep learning pipeline leveraging hybrid U-Net architectures to precisely extract lesion boundaries from dermoscopic images. The pipeline covers end-to-end processing from advanced image augmentation, model training with specialized loss functions, to real-time inference benchmarking and experiment tracking via Weights & Biases (WandB).

✨ Key Features
Advanced Image Preprocessing & Augmentation: Utilizes Albumentations to handle geometric transformations, color adjustments, and noise injection to enhance dataset diversity and model generalization.

State-of-the-Art Architectures: Implements sophisticated neural network models (such as U-Net and FPN variants) powered by the segmentation-models-pytorch library.

Specialized Loss & Evaluation Metrics: Employs a combination of loss functions (e.g., Dice Loss, Binary Cross-Entropy) and spatial overlap metrics (Intersection over Union - IoU, Dice Coefficient) to handle class imbalance and fine boundary details.

Comprehensive Experiment Tracking: Fully integrated with Weights & Biases (WandB) to monitor training curves, visualize predicted segmentation masks, and tune hyperparameters.

Inference Optimization: Benchmarks inference speed (Frames Per Second - FPS) to ensure potential real-time application suitability.

🛠 Tech Stack
Language: Python

Deep Learning Framework: PyTorch, segmentation-models-pytorch

Data Augmentation: Albumentations

Experiment Tracking: Weights & Biases (WandB)

Domain: Computer Vision, Medical Image Segmentation


📊 Dataset
Dataset Used: ISIC (International Skin Imaging Collaboration) dataset.

Preprocessing: Resizing, normalization, and heavy data augmentation to prevent overfitting on medical image variations.


🚀 Installation & Setup
Clone the repository:

Bash
git clone https://github.com/DataVuong/phandoananh.git
cd phandoananh
Create and activate a virtual environment:

Bash

python -m venv venv
source venv/bin/activate  # On Windows use: venv\Scripts\activate
Install dependencies:

Bash

pip install -r requirements.txt
📈 Training & Evaluation
To train the model with default configurations and log metrics to WandB:

Bash

python train.py --epochs 50 --batch_size 16 --lr 1e-4
To evaluate the trained model weights on the test set:

Bash

python evaluate.py --checkpoint weights/best_model.pth


⚡ Inference & Performance
Run inference on single images or batches to generate binary segmentation masks:

Bash

python infer.py --input data/test_images --output outputs/masks
The pipeline measures and logs FPS (Frames Per Second) to evaluate computational efficiency during inference.
