# COVID-19 Detection from Chest X-Ray using CNN and Transfer Learning

This project explores the use of deep learning models for classifying chest X-ray images into three categories: `NORMAL`, `BACTERIA`, and `COVID-19`.

I designed, trained, and evaluated various models, including:
- A **custom CNN** classifier trained from scratch
- Models trained with **undersampling**, **oversampling**, and **class weighting** strategies to handle dataset imbalance
- A **pretrained ResNet18** model using **Transfer Learning** with frozen layers
- Visualization of learned representations using **t-SNE**

## Dataset
I used a labeled dataset of chest X-ray images with the following distribution:

| Category   | Train Samples | Test Samples |
|------------|---------------|--------------|
| COVID-19   | 125           | 40           |
| BACTERIA   | 1079          | 242          |
| NORMAL     | 2030          | 234          |

Due to severe class imbalance, multiple balancing strategies were tested and compared.

## Key Findings
- **Oversampling** with tailored augmentations provided the best performance with:
  - Accuracy: ~99% (train), ~96% (validation)
  - F1-score: **0.8872**, Precision: **0.8765**, Recall: **0.9070**
- **Transfer Learning with ResNet18** yielded solid performance but was slightly outperformed by the oversampled CNN.
- t-SNE visualizations show that the CNN with oversampling better separates the three classes in feature space.

## Technologies
- Python, PyTorch
- Google Colab
- CNN, ResNet18, Transfer Learning
- Data Augmentation
- t-SNE visualization
