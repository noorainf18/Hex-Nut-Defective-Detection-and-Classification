<p align="center"><img width=100% src="https://github.com/noorainf18/noorainf18/blob/main/Noorain%20Fathima%20-%20Banner.png"></p>

&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;
![Python](https://img.shields.io/badge/python-v3.11+-blue.svg)
![Dependencies](https://img.shields.io/badge/dependencies-up%20to%20date-brightgreen.svg)
![Contributions welcome](https://img.shields.io/badge/contributions-welcome-orange.svg)
[![License](https://img.shields.io/badge/license-MIT-blue.svg)](https://opensource.org/licenses/MIT)


<h1 align="center">
  HEX NUT DEFECT DETECTION AND CLASSIFICATION
</h1>

Hex nut defect detection and classification play a critical role in manufacturing and quality control processes. Ensuring the quality of hex nuts is essential for several reasons. First, it directly impacts safety and reliability. Defective nuts can compromise the integrity of assembled structures, leading to accidents or failures. Second, early defect detection helps maintain high-quality standards, preventing faulty nuts from reaching end-users. Third, cost savings result from minimizing waste and rework. By catching defects during production, manufacturers avoid costly recalls or replacements. Finally, adhering to industry standards ensures that nuts meet performance requirements and fit seamlessly into existing systems.


### Table of Contents

1. Data Description
2. Methodology
3. Model Training
4. Results
5. License


## Data Description

- Type of data: 4000 images of hex nuts.
- Data format: JPG.
- Source: Kaggle


## Methodology

The methodology involves using a combination of Xception and Random Forest Classifier models. 

The Xception model and the Random Forest classifier are both widely used in machine learning for different tasks, including hex nut defect detection and classification.

Xception is a convolutional neural network (CNN) architecture designed for image classification tasks. It is known for its depthwise separable convolutions, which significantly reduce the number of parameters compared to traditional convolutional layers. This reduction in parameters allows Xception to achieve better performance while maintaining computational efficiency. In this task, the Xception model is used as a feature extractor. By setting include_top=False, the fully connected layers at the top of the network are excluded, and only the convolutional base is retained. The model is pretrained on the ImageNet dataset, which contains a vast number of images across various categories. By leveraging transfer learning, the pretrained Xception model can extract meaningful features from input images, which are then used as input to the Random Forest classifier.

On the other hand, the Random Forest classifier is an ensemble learning method that operates by constructing multiple decision trees during training and outputting the mode of the classes (classification) or the mean prediction (regression) of the individual trees. Each tree in the forest is trained on a random subset of the training data and a random subset of features. This randomness helps prevent overfitting and improves generalization performance. In our task, the Random Forest classifier is trained on the features extracted by the Xception model. These features represent high-level representations of the input images learned by the Xception model. By training the Random Forest classifier on these features, it learns to classify hex nut images into either being ‘defective’ or ‘non-defective’ based on the extracted features.


## Model Training

* Loading the Xception Base Model:
  * We initialize the base model using the Xception architecture.
  * The input shape is set to (224, 224, 3) for RGB images.
  * Pre-trained weights from the ImageNet dataset are used.
  * The top (classification) layers are excluded (include_top=False).

* Freezing Layers:
  * All layers in the base model are set to non-trainable (layer.trainable = False).
  * This ensures that the pre-trained weights are not updated during subsequent training.
    
* Feature Extraction:
  * Features are extracted from the base model for the training, validation, and test data.
  * train_features, validation_features, and test_features store the extracted features.

* Flattening Features:
  * The extracted features are reshaped into a flattened format.
  * train_features_flatten, validation_features_flatten, and test_features_flatten are created.

* Ground Truth Labels:
  * We retrieve the ground truth labels for the training, validation, and test data.
  * train_labels, validation_labels, and test_labels store the class labels.

* Random Forest Classifier:
  * A Random Forest Classifier is initialized with 100 trees (n_estimators=100).
  * The classifier is trained using the flattened features and corresponding labels.

* Predictions:
  * The trained classifier makes predictions on the training, validation, and test data.
  * train_predictions, validation_predictions, and test_predictions store the predicted labels. 


## Results

- Train Accuracy: 100.00%
- Validation Accuracy: 100.00%
- Test Accuracy: 98.88%


## License

MIT License

Copyright (c) 2024 Noorain Fathima

Permission is hereby granted, free of charge, to any person obtaining a copy of this software and associated documentation files (the "Software"), to deal in the Software without restriction, including without limitation the rights to use, copy, modify, merge, publish, distribute, sublicense, and/or sell copies of the Software, and to permit persons to whom the Software is furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE SOFTWARE.
