# Deep Learning (DL) Interview Preparation

This folder contains comprehensive notes on Deep Learning fundamentals, covering loss functions, activation functions, and various neural network architectures essential for ML interview preparation.

## 📚 Contents

### Core Components
- **[1-Loss_Functions.md](./1-Loss_Functions.md)** - Comprehensive Guide to Loss Functions
  - Regression Losses (MSE, MAE, Huber, etc.)
  - Classification Losses (Cross-Entropy, Focal Loss, etc.)
  - Ranking & Metric Learning Losses
  - Sequence Modeling Losses
  - Image Losses (SSIM, Perceptual Loss)
  - Generative & Autoencoder Losses
  - Reinforcement Learning Losses
  - Quick Reference Guide

- **[2-Activation.md](./2-Activation.md)** - Activation Functions
  - Common activation functions (ReLU, Sigmoid, Tanh, etc.)
  - Properties and use cases
  - Gradient flow considerations

### Neural Network Architectures
- **[3-DNN.md](./3-DNN.md)** - Deep Neural Networks
  - Feedforward networks
  - Backpropagation
  - Training deep networks
  - Regularization techniques

- **[4-RNN.md](./4-RNN.md)** - Recurrent Neural Networks
  - RNN architecture and variants
  - LSTM and GRU
  - Sequence modeling
  - Vanishing gradient problem

- **[5-CNN.md](./5-CNN.md)** - Convolutional Neural Networks
  - Convolution operations
  - Pooling layers
  - CNN architectures
  - Image classification and computer vision

### Additional Resources
- **[6-Appendix](./6-Appendix)** - Supplementary materials and references

## 🎯 Key Topics Covered

### Loss Functions
- **Regression**: MSE, MAE, Huber Loss, Quantile Loss
- **Classification**: Cross-Entropy, Focal Loss, Label Smoothing
- **Ranking**: Triplet Loss, Contrastive Loss, Margin Loss
- **Sequence**: CTC Loss, Connectionist Temporal Classification
- **Image**: SSIM, Perceptual Loss, Style Loss
- **Generative**: GAN Loss, VAE Loss, Wasserstein Distance

### Activation Functions
- **Sigmoid**: Binary classification, smooth gradients
- **Tanh**: Centered outputs, better than sigmoid for hidden layers
- **ReLU**: Most common, addresses vanishing gradients
- **Leaky ReLU**: Prevents dying ReLU problem
- **Swish/GELU**: Modern alternatives with smooth gradients

### Deep Neural Networks (DNN)
- Feedforward architecture
- Backpropagation algorithm
- Gradient descent variants
- Regularization (Dropout, Batch Normalization, L1/L2)
- Initialization strategies

### Recurrent Neural Networks (RNN)
- Sequential data processing
- LSTM (Long Short-Term Memory)
- GRU (Gated Recurrent Unit)
- Bidirectional RNNs
- Sequence-to-sequence models
- Attention mechanisms

### Convolutional Neural Networks (CNN)
- Convolution operations
- Pooling (Max, Average, Global)
- Feature maps and filters
- Popular architectures (LeNet, AlexNet, VGG, ResNet)
- Transfer learning
- Object detection and segmentation

## 💡 Interview Focus Areas

### Loss Functions
- When to use which loss function
- Properties and mathematical formulations
- Gradient behavior and optimization
- Loss function selection for different tasks

### Activation Functions
- Why activation functions are needed
- Comparison of different activations
- Gradient flow and vanishing/exploding gradients
- Modern activation functions

### Architecture Design
- Choosing the right architecture for the task
- Depth vs Width trade-offs
- Regularization strategies
- Training deep networks effectively

### Optimization
- Gradient descent variants (SGD, Adam, RMSprop)
- Learning rate scheduling
- Batch normalization effects
- Initialization strategies

## 📖 How to Use This Folder

1. **Start with Loss Functions**: Understand different loss functions and when to use them
2. **Learn Activation Functions**: Know the properties and trade-offs of each activation
3. **Study Architectures**: Deep dive into DNN, RNN, and CNN architectures
4. **Practice Applications**: Understand how these components work together in real models

## 🔗 Related Folders

- **[NLP](../NLP/)** - NLP models using deep learning (Transformers, BERT, GPT)
- **[GenAI](../GenAI/)** - Generative AI models and LLMs
- **[ML Concepts & Theory](../1.%20ML%20Concepts%20%26%20Theory.md)** - Machine learning fundamentals

## 📝 Study Tips

1. **Understand the math**: Know the mathematical formulations of loss functions and activations
2. **Know the trade-offs**: Understand when to use which component and why
3. **Visualize concepts**: Draw architectures and understand data flow
4. **Practice problems**: Be ready to explain backpropagation, gradient flow, and architecture choices

---

*Last Updated: 2024*

