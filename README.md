# Fashion MNIST Image Augmentation Effects on Model Accuracy

This notebook contains data processing, models, experiements and evaluation of the Fashion MNIST data. The analysis in the notebook is focused on the effects image augmentation has on various models and loss types.

**High level summary of results:**
* MLP ANN had the strongest accuracy followed by SVM (nearly identical accuracy) and then kNN.
* All models had similiar reactions to the image augmentations with inverted pixels, resizing, and horizontal flip causing the biggest decline in accuracy.
* The default levels of noise had minimal impact on model accuracy, but as noise levels increased, kNN performed the best. SVM and MLP ANN had the largest drop in accuracy as noise was added.
* MLP ANN was by far the fastest to train and predict.
* Different loss functions from Cross Entropy (Hinge, MSE, MAE) had a lower overall accuracy with small amount of noise and as noise increased, all loss functions performed similar.
