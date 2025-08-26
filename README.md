# Airbnb Price Prediction with TabNet

This project implements a **TabNet-based regression model** to predict Airbnb listing prices using a preprocessed dataset. The code preprocesses the data, extracts features, trains a TabNet model, and evaluates its performance using various metrics.

## Table of Contents
- [Overview](#overview)
- [Dataset](#dataset)
- [Features](#features)
- [Installation](#installation)
- [Usage](#usage)
- [Preprocessing Steps](#preprocessing-steps)
- [Model Training](#model-training)
- [Evaluation Metrics](#evaluation-metrics)
- [Dependencies](#dependencies)
- [Contributing](#contributing)
- [License](#license)

## Overview
This project uses the **TabNet** model (a deep learning architecture for tabular data) to predict Airbnb listing prices based on various features such as location, property type, amenities, and more. The dataset is preprocessed to handle missing values, encode categorical variables, and extract meaningful features from text fields like `bathrooms_text` and `amenities`.

## Dataset
The dataset is expected to be a CSV file (`data.csv`) containing Airbnb listing information. Key columns include:
- `price`: Target variable (listing price in USD).
- `neighbourhood_cleansed`, `property_type`, `room_type`: Categorical features.
- `bathrooms_text`, `amenities`: Text fields used to extract numerical and categorical features.
- Numerical features like `bedrooms`, `minimum_nights`, `maximum_nights`, etc.

The dataset should be placed in the specified file path (e.g., `C:\\Users\\***\\data.csv`).

## Features
- **Numerical Features**: `bedrooms`, `minimum_nights`, `maximum_nights`, etc., with missing values filled using the column mean.
- **Categorical Features**: Encoded using `LabelEncoder` for columns like `neighbourhood_cleansed`, `property_type`, and `room_type`.
- **Derived Features**:
  - `bathrooms`: Extracted as a float from `bathrooms_text`.
  - `bathroom_type`: Derived from `bathrooms_text` (0 = not specified, 1 = shared, 2 = private).
  - **Amenities**: Binary columns for each amenity (e.g., `Wifi`, `Kitchen`, `Air conditioning`) based on presence in the `amenities` column.

## Installation
1. Clone the repository:
   ```bash
   git clone https://github.com/your-username/airbnb-price-prediction.git
   cd airbnb-price-prediction
   ```

2. Install the required dependencies:
   ```bash
   pip install -r requirements.txt
   ```

## Usage
1. Ensure the dataset (`data.csv`) is in the correct path.
2. Run the script:
   ```bash
   python airbnb_price_prediction.py
   ```
3. The script will preprocess the data, train the TabNet model, and output the following metrics:
   - **R² Score**: Measures the proportion of variance explained by the model.
   - **Mean Squared Error (MSE)**: Average squared difference between predicted and actual prices.
   - **Root Mean Squared Error (RMSE)**: Square root of MSE, providing error in the same units as price.
   - **Mean Absolute Error (MAE)**: Average absolute difference between predicted and actual prices.

## Preprocessing Steps
1. **Data Loading**: Reads the CSV file into a pandas DataFrame.
2. **Column Removal**: Drops irrelevant columns (e.g., `id`, `listing_url`, `host_name`).
3. **Missing Value Handling**: Fills missing numerical values with column means.
4. **Price Cleaning**: Removes `$` and converts `price` to float.
5. **Feature Extraction**:
   - Extracts `bathrooms` as a float from `bathrooms_text`.
   - Creates a `bathroom_type` column based on keywords (`shared`, `private`).
   - Converts `amenities` into binary columns for a predefined list of amenities.
6. **Categorical Encoding**: Uses `LabelEncoder` to encode categorical columns.
7. **Data Splitting**: Splits data into 80% training and 20% testing sets.

## Model Training
- **Model**: TabNetRegressor from the `pytorch_tabnet` library.
- **Training Parameters**:
  - `max_epochs=100`: Trains for up to 100 epochs.
  - `patience=10`: Stops training if no improvement is seen for 10 epochs.
- **Input**: Features (X) and target (`price`) in NumPy format.
- **Output**: Predicted prices for the test set.

## Evaluation Metrics
The model is evaluated using:
- **R² Score**: Indicates how well the model explains the variance in the target variable.
- **MSE**: Measures average squared error.
- **RMSE**: Provides error in the same units as the target variable.
- **MAE**: Measures average absolute error.

Example output:
```
📊 Modelin nəticələri:
✔️ R²: 0.XXXX
✔️ MSE: XXXX.XXXX
✔️ RMSE: XXX.XXXX
✔️ MAE: XXX.XXXX
```

## Dependencies
- Python 3.x
- pandas
- numpy
- scikit-learn
- pytorch-tabnet
- torch

Install dependencies using:
```bash
pip install pandas numpy scikit-learn pytorch-tabnet torch
```

## Contributing
Contributions are welcome! Please follow these steps:
1. Fork the repository.
2. Create a new branch (`git checkout -b feature-branch`).
3. Make your changes and commit (`git commit -m 'Add feature'`).
4. Push to the branch (`git push origin feature-branch`).
5. Open a Pull Request.

## License
This project is licensed under the MIT License. See the [LICENSE](LICENSE) file for details.

---

This README provides a clear overview of the project, its functionality, and instructions for running the code. Let me know if you need any adjustments or additional sections!