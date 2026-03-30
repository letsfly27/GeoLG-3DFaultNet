# GeoLG-3DFaultNet

This repository contains the official PyTorch implementation and the quick-test dataset for the manuscript:
**"GeoLG-3DFaultNet: The 3D Fault Segmentation Method Based on Local-Global Feature Enhancement and Geometric Constraints"**.

## 1. Environment Requirements
This code has been tested on Linux (Ubuntu) with the following specific environment:
- **Python:** 3.9.21
- **PyTorch:** 2.8.0+cu128 (CUDA Available: True)

To replicate the environment and install all necessary dependencies, please run:
```bash
pip install -r requirements.txt
```

## 2. Repository Structure
- `model.py`: The core implementation of the GeoLG-3DFaultNet architecture, including the **Local-Global Feature Enhancement (LG-Module)** and **Attention-guided Continuity Optimization (ACO-Module)**.
- `inference.py`: The evaluation script for quick testing and generating 3D visualization results.
- `train.py`: The script used for training the network with the proposed joint geometric constraint loss.
- `utils.py`: Contains the implementations of the 4 combined loss functions (BCE, Dice, clDice, Edge Smoothing) and evaluation metrics.
- `best_model.pth`: The pre-trained model weights for quick inference.
- `data/`: Contains a sample 3D seismic patch and its corresponding ground truth for the quick test.

## 3. Quick Test (For Reviewers)
To verify the functionality of our code, we provide a foolproof quick-test script. You do not need to configure complex paths or train the model from scratch.

Simply run the following command in your terminal:
```bash
python inference.py
```

**What will happen?**
1. The script will automatically load the pre-trained weights (`best_model.pth`).
2. It will perform 3D inference on the sample data located in `./data/seis/`.
3. Upon completion, a high-resolution comparison image (Seismic vs. Ground Truth vs. GeoLG-3DFaultNet) will be automatically generated and saved in the newly created `./predict_images/` directory.

## 4. Training (Optional)
If you wish to train the model from scratch on your own dataset, please organize your 3D seismic data in `.dat` or `.npy` format, update the data paths in `train.py`, and run:
```bash
python train.py
```

## 5. Citation
If you find this code or our method useful in your research, please consider citing our paper (Citation details will be updated upon acceptance).

## 6. License
This project is released under the MIT License.
