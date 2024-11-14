# PointCloud-Classifier
This repository consists of a PointNet-like-model to classify 3D point cloud data. 


## Features
- **PointNet Architecture** for 3D point cloud data
- **ModelNet10 Dataset** for 3D object classification
- Modular, well-organized codebase for easy modifications and improvements

## Prerequisites

- Python 3.8 or higher
- CUDA-compatible GPU (optional but recommended)

## Setup

### 1. Clone the repository

```bash
git clone https://github.com/your-username/PointCloud-Classifier.git
cd PointCloud-Classifier
```

### 2. Create and activate a virtual environment

It's recommended to use a virtual environment to manage dependencies for this project.

```bash
python3 -m venv my-venv
source my-venv/bin/activate  # On Windows: venv\Scripts\activate
```

### 3. Install required packages

```bash
pip install -r requirements.txt
```

### 4. Download the ModelNet10 Dataset

Download and extract the ModelNet10 dataset from [here](https://modelnet.cs.princeton.edu/). After extracting, organize the data with separate folders for `train` and `test` sets within the dataset folder. Update the path to this dataset in `main.py`:

```python
train_loader = get_dataloader(root_dir='path/to/ModelNet10', batch_size=Config.BATCH_SIZE, num_points=Config.NUM_POINTS, split='train')
test_loader = get_dataloader(root_dir='path/to/ModelNet10', batch_size=Config.BATCH_SIZE, num_points=Config.NUM_POINTS, split='test')
```

### 5. Run the project

To train and evaluate the model:

```bash
python main.py
```

The script will begin training the model on the ModelNet10 dataset and print accuracy metrics after each epoch.



### Configuration

You can modify `config.py` to change hyperparameters like `BATCH_SIZE`, `NUM_POINTS`, and `EPOCHS`.

### Additional Notes

- **CUDA Support**: If you're using a GPU, ensure that your PyTorch installation is CUDA-enabled.
- **Virtual Environment**: Remember to activate the virtual environment each time you work on this project.

Happy training!

sudo docker run -it --name test3d \
    --gpus all \
    -v /home/btp_2/PointCloud-Classifier:/workspace \
    pytorch/pytorch:1.13.1-cuda11.6-cudnn8-devel /bin/bash

sudo docker start test3d

sudo docker attach test3d

Detach: Ctrl + P + Q

sudo docker stop test3d


Install python-dev

pip install --no-build-isolation "git+https://github.com/facebookresearch/pytorch3d.git@stable"
