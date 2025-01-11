# xCAPT5Service

This repository contains the source code for the **xCAPT5-service**.

---

## Environment requirements
- **Operating system**: Ubuntu
- **Python**: 3.10
- **Conda**: Latest version (for environment management)
- **CUDA**: 11.8
- **GPU available**

## Setup Instructions

### Step 1: Clone the repository
```bash
git clone https://github.com/anhducpn67/xCAPT5-service.git
cd xCAPT5-service/
```

### Step 2: Create conda environment
Create a new Conda environment with Python 3.10
```bash
conda create -n service python=3.10 -y
conda activate service
```
Install dependencies
```bash
pip3 install -r requirements.txt
pip3 install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118
```
Note: To verify if TensorFlow is using the GPU, run the following command:
```bash
python3 -c "import tensorflow as tf; print(tf.config.list_physical_devices('GPU'))"
```
If TensorFlow does not detect the GPU, ensure you have the necessary drivers installed. (e.g. CUDA version 11.8)

### Step 3: Download Model checkpoints
Run the provided script `backend/download_checkpoints.sh` to download the necessary model checkpoints
```bash
cd backend
bash download_checkpoints.sh
```
Note: Ensure the script has executable permissions. You can set it using
```bash
chmod +x download_checkpoints.sh
```


### Step 4: Start the service
Run the service using `uvicorn` with port 8091
```bash
uvicorn backend.main:app --port 8091 --reload
```




