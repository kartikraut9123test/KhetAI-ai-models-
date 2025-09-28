
# importing necessary libraries 
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# loading the dataset
crop_data=pd.read_csv("predictioncrop.csv")
crop_data
crop_data.shape

#rows X columns
# dataset columns
crop_data.columns

# Statistical summary of data frame.
crop_data.describe()

# Checking missing values of the dataset in each column
crop_data.isnull().sum()

# Replacing missing values with mean of the production coloumn
crop_data['Production'] = crop_data['Production'].fillna(crop_data['Production'].mean())
crop_data

#checking
crop_data.isnull().values.any()

# Displaying State Names present in the dataset
print(crop_data.State_Name.unique())
print('Total count of states and Union Territories:', len(crop_data.State_Name.unique()))

# Adding a new column Yield which indicates Production per unit Area. 
crop_data['Yield'] = (crop_data['Production'] / crop_data['Area'])
crop_data.head(10) 

# Dropping unnecessary columns
data = crop_data.drop(['State_Name'], axis = 1)

# This filters only numeric columns (int, float)
numeric_data = data.select_dtypes(include=['number'])

# Now calculate correlation safely
corr_matrix = numeric_data.corr()

# Print or visualize
print(corr_matrix)
sns.heatmap(numeric_data.corr(), annot=True, cmap='coolwarm')
plt.title("Correlation Heatmap")
plt.show()
sns.heatmap(numeric_data.corr(), annot =True, fmt='.4f')
plt.title('Correlation Matrix')
plt.show()
dummy = pd.get_dummies(data)
dummy
from sklearn.model_selection import train_test_split
x = dummy.drop(["Production","Yield"], axis=1)
y = dummy["Production"]

# Splitting data set - 25% test dataset and 75% 
x_train,x_test,y_train,y_test = train_test_split(x,y,test_size=0.25, random_state=42)
print("x_train :",x_train.shape)
print("x_test :",x_test.shape)
print("y_train :",y_train.shape)
print("y_test :",y_test.shape)
print(x_train)
print(y_train)
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np

# creating the dataset
year = [2012,2013,2014]
values = []
data1 = pd.DataFrame(crop_data)
for x in range(3):
  curyearsum=0
  count=0
  for y in range(1000):
    if(data1.iloc[y]["Crop_Year"] == year[x]):
      count+=1;
      curyearsum += data1.iloc[x]["Humidity"]
  values.append(curyearsum/count)
fig = plt.figure(figsize = (7, 5))

# creating the bar plot
x = np.array(["2012","2013","2014"])
y = np.array(values)
plt.xlabel("year")
plt.ylabel("mean value of humidity")
plt.title("mean value of humidity of three years")
plt.bar(x,y,color ='maroon', width = 0.4)
plt.show()
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np

# creating the dataset
year = [2012,2013,2014]
values = []
data1 = pd.DataFrame(crop_data)
for x in range(3):
  curyearsum=0
  count=0
  for y in range(1000):
    if(data1.iloc[y]["Crop_Year"] == year[x]):
      count+=1;
      curyearsum += data1.iloc[x]["Temperature"]
  values.append(curyearsum/count)
fig = plt.figure(figsize = (7, 5))
 
# creating the bar plot
x = np.array(["2012","2013","2014"])
y = np.array(values)
plt.xlabel("year")
plt.ylabel("mean value of Temperature")
plt.title("mean value of Temperature of three years")
plt.bar(x,y,color ='maroon', width = 0.4)
plt.show()
from sklearn.ensemble import RandomForestRegressor
model = RandomForestRegressor(n_estimators = 11)
model.fit(x_train,y_train)
rf_predict = model.predict(x_test)
rf_predict
model.score(x_test,y_test)

# Calculating R2 score
from sklearn.metrics import r2_score
r1 = r2_score(y_test,rf_predict)
print("R2 score : ",r1)

#Calculating Adj. R2 score: 
Adjr2_1 = 1 - (1-r1)*(len(y_test)-1)/(len(y_test)-x_test.shape[1]-1)
print("Adj. R-Squared : {}".format(Adjr2_1))

# KDE Plot
ax = sns.kdeplot(y_test, color="r", label="Actual Value")
sns.kdeplot(rf_predict, color="b", label="Predicted Values", ax=ax)
plt.title("Random Forest Regression")
plt.xlabel("Actual")
plt.ylabel("Predicted")
plt.legend()
plt.show()
plt.scatter(y_test,rf_predict)
plt.title('Random Forest')
