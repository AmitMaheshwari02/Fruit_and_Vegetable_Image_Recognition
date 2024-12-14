# Fruit & Vegetable Image Recognition

This project focuses on image recognition of fruits and vegetables using the MobileNetV2 model. It includes feature extraction and training for accurate classification. The model and corresponding Jupyter Notebook provide a workflow for recognizing different categories of fruits and vegetables from images.

## Features
- Pre-trained MobileNetV2 for feature extraction.
- Fine-tuning the model for fruit and vegetable classification.
- Input pipeline and preprocessing for image data.
- Output predictions with high accuracy.

## Files in the Repository
1. **f_&_v_feature_extraction.ipynb**
   - A Jupyter Notebook containing the steps for data preprocessing, feature extraction, and model training.
2. **mobilenet_v2__f_&_v_image_recognition_model.h5**
   - A trained MobileNetV2-based model saved in HDF5 format.

## Requirements
Ensure the following libraries and tools are installed:
- Python 3.8+
- TensorFlow 2.15
- Numpy
- Matplotlib
- OpenCV
- Jupyter Notebook

Install the requirements using:
```bash
pip install tensorflow numpy matplotlib opencv-python jupyter
```

## How to Run
1. Clone the repository:
   ```bash
   git clone https://github.com/AmitMaheshwari02/Fruit_and_Vegetable_Image_Recognition.git
   cd Fruit_and_Vegetable_Image_Recognition
   ```

2. Open the Jupyter Notebook:
   ```bash
   jupyter notebook f_&_v_feature_extraction.ipynb
   ```

3. Follow the steps in the Notebook to preprocess data, extract features, and run predictions using the trained model.

4. To load the pre-trained model and run predictions:
   ```python
   from tensorflow.keras.models import load_model
   model = load_model('mobilenet_v2__f_&_v_image_recognition_model.h5')
   ```

## Model Architecture
- **Backbone:** MobileNetV2
- **Input Shape:** (224, 224, 3)
- **Number of Classes:** 36

## Results
- Training Accuracy :- 84.19%
- Validation Accuracy :- 85.57%

## Acknowledgments
- The project utilizes the [MobileNetV2](https://arxiv.org/abs/1801.04381) architecture.
- TensorFlow/Keras for model development and training.

