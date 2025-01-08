# FarmAgent-Soil-Analysis-and-Crop-Recommendation

Project Overview
This project aims to assist farmers in selecting the most appropriate crop for their land based on the soil's chemical and environmental properties. By training a machine learning model on historical soil data, the system can predict the most suitable crop type, improving productivity and reducing the risk of crop failure.

The model is built using the Random Forest Classifier algorithm, which is an ensemble learning method that performs well with tabular data.

Technologies Used
Python (Programming Language)
Pandas (Data Processing)
Scikit-learn (Machine Learning)
Matplotlib (For Data Visualization – optional)
Jupyter Notebook (Optional for testing and prototyping)
Dataset
This project uses a CSV file (soil_data.csv) containing soil data and the corresponding crop types. The dataset consists of the following features:

N: Nitrogen content
P: Phosphorus content
K: Potassium content
temperature: Average temperature (°C)
humidity: Average humidity (%)
ph: Soil pH
rainfall: Average rainfall (mm)
crop_type: The target variable, indicating the recommended crop based on the soil conditions.
