# Fashion MNIST Image Augmentation Effects on Model Accuracy

**Overview:**
  - This project investigates the effects of image augmentation on the accuracy of various models trained on the Fashion MNIST dataset. The models evaluated include a Multi-Layer Perceptron Artificial Neural Network (MLP ANN), Support Vector Machine (SVM), and K-Nearest Neighbors (kNN). The analysis focuses on how different image augmentations impact model performance and robustness.

**Problem Statement:**
  - The objective of this project is to determine:
    - Which model best predicts the Fashion MNIST images.
    - Which model can best predict augmented images they were never trained on.
    - Which model is the most robust as levels of noise are added to the images.
    - Whether changing the loss function can make the ANN more robust given its speed advantage over SVM and kNN.
    
**Dataset:**
  - Dataset Used: Fashion MNIST
  - Number of Classes: 10
  - Classes: T-shirt/top, Trouser, Pullover, Dress, Coat, Sandal, Shirt, Sneaker, Bag, Ankle boot
  - Training Samples: 50,000
  - Validation Samples: 10,000
  - Test Samples: 10,000
  - Image Dimensions: 28x28 grayscale images, flattened into 784-dimensional vectors

**Methodology:**
  - Data Preprocessing
    - Normalization: Pixel values were normalized to be between 0 and 1.
    - One-Hot Encoding: Class labels were one-hot encoded.
  - Image Augmentation
    - The following augmentations were applied to the dataset:
      - Inverted Pixels
      - Resizing
      - Horizontal Flip
      - Adding Noise

**Model Architectures:**
  - Support Vector Machine (SVM): Linear kernel, trained on a smaller subset of the training data.
  - K-Nearest Neighbors (kNN): k=5, trained on a smaller subset of the training data.
  - Multi-Layer Perceptron Artificial Neural Network (ANN):
    - Layers: Dense layers with ReLU activation and Dropout for regularization.
    - Loss Functions: Binary Cross-Entropy, Hinge, Mean Squared Error (MSE), Mean Absolute Error (MAE).
  
**Training and Evaluation:**
  - Training: Models were trained on the preprocessed and augmented datasets.
  - Evaluation: Models were evaluated on the original and augmented test datasets. Metrics included accuracy, precision, recall, and F1-score.
  - Visualization: Confusion matrices and learning curves were plotted to visualize model performance.

**Experiments and Evaluation:**
  - Experiment 1: Model Comparison
    - Objective: Determine which model best predicts the Fashion MNIST images. Hypothesis: ANN will have the highest accuracy due to its ability to capture the complexity of image recognition.
    - Results:
      - ANN: 88.4%
      - SVM: 87.5%
      - kNN: 81.8%
  - Experiment 2: Augmented Image Prediction
    - Objective: Determine which model can best predict augmented images they were never trained on. Hypothesis: All models will equally struggle to predict image augmentation with inverted and resizing being the most difficult to predict.
    - Findings: All models struggled to predict image augmentations at relatively equal proportions. Noise and Blur had the best accuracy across models, while inverted, resized, and flipped had a large drop in prediction accuracy. Horizontally flipped images did not predict any shoe type well.

  - Experiment 3: Robustness to Noise
    - Objective: Determine which model is the most robust as levels of noise are added to the images. Hypothesis: Deep learning methods such as ANN should be the most robust to added noise.
    - Findings:
      - kNN: Performed the best with increasing noise levels, but took an extremely long time to run.
      - SVM and ANN: Showed a significant drop in accuracy as noise levels increased.

  - Experiment 4: ANN Loss Function Comparison
    - Objective: Determine if changing the loss function can make the ANN more robust. Hypothesis: Cross Entropy will have a higher accuracy with minimal noise, but other loss functions could have higher accuracy when a lot of noise is added.
    - Findings: Cross Entropy had the highest accuracy with no noise and as noise was added, all loss functions performed similarly.

**Key Results:**
  - Best Model: The ANN achieved the highest accuracy on the original dataset with 88.4%.
- Augmentation Impact: All models experienced a decline in accuracy with augmented images, particularly with inverted pixels, resizing, and horizontal flips.
- Noise Robustness: kNN was the most robust to increasing noise levels, maintaining better performance compared to SVM and ANN.
- Loss Function Comparison: Different loss functions for ANN (Cross Entropy, Hinge, MSE, MAE) showed similar performance as noise levels increased, with Cross Entropy having the highest accuracy without noise.

**Conclusion:**
The ANN model demonstrated the highest accuracy on the Fashion MNIST dataset, followed closely by SVM. kNN was the most robust to noise. Image augmentations generally decreased model accuracy, with certain augmentations having a more pronounced effect. Future work could explore more complex architectures, hyperparameter tuning, and additional data augmentation techniques to further improve model performance.

**Future Work:**
  - Experiment with Different Architectures: Try deeper or more complex models.
  - Hyperparameter Tuning: Optimize batch size, learning rate, and other parameters.
  - Transfer Learning: Utilize pre-trained models and fine-tune them on the Fashion MNIST dataset.
  - Ensemble Methods: Combine predictions from multiple models to improve accuracy.
