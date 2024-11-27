import pandas as pd
import matplotlib.pyplot as plt
data = pd.read_csv('data/test.csv', delimiter=";", decimal=",")
print(data.head())

import matplotlib.pyplot as plt
fig, ax = plt.subplots(figsize=(12, 6))
data.plot('Datum', 'kWh', ax=ax)
ax.set(title="St Wolfgang 2024")
plt.show()

from scikit.svm import LinearSVC
 
# Construct data for the model
X = data[['petal length (cm)','petal width (cm)']]
y = data[['target']]
 
# Fit the model
model = LinearSVC()
model.fit(X, y)


# Create input array
X_predict = targets[['petal length (cm)', 'petal width (cm)']]

# Predict with the model
predictions = model.predict(X_predict)
print(predictions)

# Visualize predictions and actual values
plt.scatter(X_predict['petal length (cm)'], X_predict['petal width (cm)'],
            c=predictions, cmap=plt.cm.coolwarm)
plt.title("Predicted class values")
plt.show()
