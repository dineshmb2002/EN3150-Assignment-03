import sys
import torch
import numpy as np
import matplotlib
import matplotlib.pyplot as plt
from PIL import Image
import sklearn
import pandas as pd
import seaborn as sns
import imageio

print("Python version:", sys.version)
print("PyTorch version:", torch.__version__)
print("CUDA available:", torch.cuda.is_available())
if torch.cuda.is_available():
    print("CUDA device name:", torch.cuda.get_device_name(0))
print("NumPy version:", np.__version__)
print("Matplotlib version:", matplotlib.__version__)
print("Pillow version:", Image.__version__)
print("Scikit-learn version:", sklearn.__version__)
print("Pandas version:", pd.__version__)
print("Seaborn version:", sns.__version__)
print("Imageio version:", imageio.__version__)

# Test loading a dummy image
try:
    img = Image.new('L', (64, 64))  # Create a dummy grayscale image
    arr = np.array(img)
    print("Image loaded successfully, shape:", arr.shape)
except Exception as e:
    print("Error loading image:", e)
