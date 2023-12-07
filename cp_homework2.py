import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix

np.random.seed(42)
num_samples = 1000

earthquake_intensity = np.random.uniform(0, 10, num_samples)
damage_occurred = np.random.choice([0, 1], size=num_samples, p=[0.8, 0.2])

data = pd.DataFrame({'Earthquake_Intensity': earthquake_intensity, 'Damage_Occurred': damage_occurred})

X = data[['Earthquake_Intensity']]
y = data['Damage_Occurred']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

model = RandomForestClassifier(n_estimators=100, random_state=42)
model.fit(X_train, y_train)

y_pred = model.predict(X_test)
accuracy = accuracy_score(y_test, y_pred)
classification_rep = classification_report(y_test, y_pred)
conf_matrix = confusion_matrix(y_test, y_pred)

print(f"Accuracy: {accuracy}")
print("Classification Report:\n", classification_rep)
print("Confusion Matrix:\n", conf_matrix)
