# EN3150 Assignment 03 – Convolutional Neural Network for Image Classification

This repository contains the implementation and report for **EN3150 Assignment 03**.  
The assignment focuses on building a **simple Convolutional Neural Network (CNN)** for image classification and comparing it with state-of-the-art pre-trained models using **transfer learning**.

---

## 📌 Assignment Overview

1. **Custom CNN Implementation**
   - Build a simple CNN from scratch using TensorFlow/Keras or PyTorch.
   - Dataset selected from **UCI Machine Learning Repository** (not CIFAR-10).
   - Dataset split into **70% training, 15% validation, 15% testing**.
   - Train the model for 20 epochs.
   - Evaluate using accuracy, confusion matrix, precision, recall.
   - Justify choices of activation functions, kernel sizes, filters, fully connected layer size, and dropout rate.

2. **Optimizers**
   - Train the network using a selected optimizer (e.g., Adam, RMSprop).
   - Compare performance with:
     - Standard **Stochastic Gradient Descent (SGD)**
     - **SGD with Momentum**
   - Discuss the effect of learning rate and momentum.

3. **Transfer Learning**
   - Fine-tune **two state-of-the-art pre-trained models** (e.g., ResNet, VGG, AlexNet, GoogLeNet, DenseNet).
   - Train on the same dataset splits as the custom CNN.
   - Record training/validation loss and evaluate on test data.
   - Compare custom CNN with pre-trained models.

4. **Discussion**
   - Analyze trade-offs, advantages, and limitations of using custom CNN vs. transfer learning.
   - Provide performance comparison using chosen metrics.
   


