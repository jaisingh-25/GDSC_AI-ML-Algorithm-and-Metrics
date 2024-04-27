import pandas as pd
import matplotlib.pyplot as plt
raw_data = pd.read_csv(r'...\ParisHousing.csv')
x = raw_data[['squareMeters', 'numberOfRooms', 'hasYard', 'hasPool', 'floors', 'cityCode', 'cityPartRange',
              'numPrevOwners', 'made', 'isNewBuilt', 'hasStormProtector', 'basement', 'attic', 'garage',
              'hasStorageRoom', 'hasGuestRoom']]
y = raw_data['price']
# Splitting into test and train data
from sklearn.model_selection import train_test_split
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size = 100)

# Applying linear regression
from sklearn.linear_model import LinearRegression
model = LinearRegression()
model.fit(x_train, y_train)
predictions = model.predict(x_test)
plt.scatter(y_test, predictions)

# Calculating mean squared error
from sklearn import metrics
mse=metrics.mean_squared_error(y_test, predictions)
print(mse)
