# Multimodal Chest X-ray Analysis with Dual VAE and Classifier

## Overview
This project implements a multimodal deep learning system for chest X-ray analysis, combining dual-view image processing (frontal and lateral) with text data (captions). It uses a Variational Autoencoder (VAE) with a classifier to perform image classification, reconstruction, and synthetic image generation from text findings.

## Features
- **Dual-View Processing**: Encodes frontal and lateral X-ray images using ResNet-18.
- **Variational Autoencoder**: Reconstructs images and learns a latent space representation.
- **Classification**: Predicts four diagnostic categories: Normal, Pulmonary, Cardiovascular, Musculoskeletal.
- **Text-to-Image Generation**: Uses GPT-2 and a regression layer to generate synthetic X-rays from captions.
- **Evaluation Metrics**: Includes SSIM, PSNR, and classification accuracy for synthetic images.

## Dataset
- **Chest X-rays Indiana University**: Sourced from [Kaggle](https://www.kaggle.com/datasets/raddar/chest-xrays-indiana-university).
- Contains paired frontal/lateral X-ray images, metadata (`projection.csv`), and reports (`reports.csv`).

## Requirements
- Python 3.8+
- PyTorch
- Torchvision
- Pytorch-Lightning
- Transformers (Hugging Face)
- OpenCV
- Pandas
- Scikit-learn
- Matplotlib
- TQDM
- TIMM

Install dependencies:
```bash
pip install torch torchvision pytorch-lightning transformers opencv-python pandas scikit-learn matplotlib tqdm timm pillow
```

## Usage
1. **Prepare the Dataset**:
   - Download the dataset and place it in `/kaggle/input/chest-xrays-indiana-university/`.
   - Ensure `images_normalized/`, `indiana_projections.csv`, and `indiana_reports.csv` are accessible.
2. **Run the Code**:
   - Execute the notebook:
     ```bash
     jupyter notebook deeplearniningcode.ipynb
     ```
     or run in Google Colab.
3. **Output**:
   - Training/validation loss and accuracy plots.
   - Reconstructed and synthetic X-ray images.
   - Evaluation metrics (SSIM, PSNR, accuracy).

## Results
- **Classification**: Achieves high accuracy on test set (specific value depends on training).
- **Reconstruction**: Visualizes original vs. reconstructed images for frontal and lateral views.
- **Synthetic Images**: Generates X-rays from captions, evaluated with SSIM/PSNR.
- **Plots**:
  ![Training History](results/training_history.png)
  ![Confusion Matrix](results/confusion_matrix.png)
  ![Reconstructions](results/reconstructions.png)

## Project Structure
- `deeplearniningcode.ipynb`: Main notebook with data processing, model, and training.
- `saved_model/`: Directory for saved model weights (`best_model.pth`, `text_model.pth`).
- `results/`: Output plots and visualizations (e.g., training history, confusion matrix).

## How It Works
1. **Data Prep**: Processes X-ray images and metadata, categorizes problems into four classes.
2. **Model**: DualVAEWithClassifier encodes images into a latent space, reconstructs them, and classifies categories.
3. **Text-to-Latent**: GPT-2 maps captions to latent vectors, decoded into synthetic images.
4. **Training**: Uses MSE, KL divergence, and BCE losses with Adam optimizer over 30 epochs.
5. **Evaluation**: Assesses reconstruction quality and classification performance.

## License
MIT License - free to use and adapt!

