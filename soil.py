# Importing libraries
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score

# Load dataset
data = pd.read_csv('soil_data.csv')  # Replace with your dataset path

# Features and target
X = data[['N', 'P', 'K', 'temperature', 'humidity', 'ph', 'rainfall']]  # Input features
y = data['crop_type']  # Target variable

# Split data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Initialize RandomForest Classifier
model = RandomForestClassifier(n_estimators=100, random_state=42)

# Train the model
model.fit(X_train, y_train)

# Make predictions
y_pred = model.predict(X_test)

# Evaluate the model
accuracy = accuracy_score(y_test, y_pred)
print(f'Accuracy: {(accuracy * 100):.2f}%')

# Function to predict crop based on new soil data
def recommend_crop(soil_data):
    return model.predict([soil_data])

# Get user input for soil data
print("Enter the following soil parameters:")
N = float(input("Nitrogen (N): "))
P = float(input("Phosphorus (P): "))
K = float(input("Potassium (K): "))
temperature = float(input("Temperature (°C): "))
humidity = float(input("Humidity (%): "))
ph = float(input("pH: "))
rainfall = float(input("Rainfall (mm): "))

# Create soil data array
new_soil = [N, P, K, temperature, humidity, ph, rainfall]

# Predict and display the recommended crop
recommended_crop = recommend_crop(new_soil)
print(f'Recommended Crop: {recommended_crop[0]}')
