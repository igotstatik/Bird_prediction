# Bird Species Classifier

This project implements a machine learning model to classify bird species based on images. It uses a high-quality dataset of 20 bird species and is built with Python, TensorFlow, and supporting libraries.

---

## Features
- **Classification of 20 bird species** using a deep learning model.
- Preprocessing pipeline for resizing and normalizing images.
- Training and validation workflows with real-time monitoring of metrics.
- Functionality to predict bird species for custom images.

---

## Dataset
The dataset is sourced from Kaggle: **BIRDS 20 SPECIES - IMAGE CLASSIFICATION**
https://www.kaggle.com/datasets/umairshahpirzada/birds-20-species-image-classification/data

- **Training Images**: 3208
- **Validation Images**: 100 (5 per species)
- **Test Images**: 100 (5 per species)
- **Image Format**: JPEG, 224x224x3
- **Key Feature**: Single bird in each image, occupying at least 50% of the pixels.

---

## Requirements
- Python 3.8+
- macOS (or any OS with compatible Python setup)
- Libraries:
  - `numpy`
  - `pandas`
  - `tensorflow` (>=2.12)
  - `opencv-python`
  - `matplotlib`

Install dependencies:
```bash
pip install -r requirements.txt
```

---

## Project Structure
```
.
├── bird_classifier.ipynb     # Jupyter Notebook with full implementation
├── requirements.txt          # Required libraries
├── README.md                 # Project documentation
└── sample_images/            # Folder for testing your images
```

---

## Usage

### Training the Model
1. Open the `bird_classifier.ipynb` notebook in Google Colab.
2. Upload the dataset.
3. Run the notebook cells to:
   - Preprocess the data.
   - Train the model.
   - Save the best-performing model.

### Testing the Model
1. Load the saved model using the `load_model` function.
2. Use the function `predict_species` to classify new bird images:

```python
predicted_species = predict_species('path_to_image.jpg', model, class_indices)
print(f"Predicted species: {predicted_species}")
```

---

## Example Prediction
Input image:
![Example Bird](sample_images/example_bird.jpg)

Output:
```
Predicted species: Cardinal
```

---

## Results
- Achieved **X% accuracy** on the test dataset.
- Model generalizes well to unseen data.

---

## Future Improvements
- Implement advanced data augmentation.
- Test with more complex architectures like EfficientNet.
- Expand the dataset with more species and images.

---

## License
This project is licensed under the MIT License. See `LICENSE` for details.

