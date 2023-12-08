import pandas as pd
from sklearn.preprocessing import MinMaxScaler
from sklearn.ensemble import RandomForestRegressor
from sklearn.preprocessing import PolynomialFeatures

x_train = pd.read_csv('dataset_4e732c/housing_x_train_4e732c.csv', sep=',', encoding='utf-8').values
y_train = pd.read_csv('dataset_4e732c/housing_y_train_4e732c.csv', sep=',', encoding='utf-8').values

scaler_x = MinMaxScaler()
scaler_y = MinMaxScaler()

scaler_x.fit(x_train)
normalized_x_train = scaler_x.transform(x_train)

scaler_y.fit(y_train)
normalized_y_train = scaler_y.transform(y_train)

x_test = pd.read_csv('dataset_4e732c/housing_x_test_4e732c.csv', sep=',', encoding='utf-8').values

normalized_x_test = scaler_x.transform(x_test)


poly = PolynomialFeatures(degree=2)
normalized_x_train_poly = poly.fit_transform(normalized_x_train)
normalized_x_test_poly = poly.transform(normalized_x_test)

##using rf

random_forest = RandomForestRegressor(n_estimators=50, random_state=42, n_jobs=-1)

random_forest.fit(normalized_x_train_poly, normalized_y_train.ravel())

predictions_rf = random_forest.predict(normalized_x_test_poly)

reverted_rf_predictions = scaler_y.inverse_transform(predictions_rf.reshape(-1, 1))



predictions_df = pd.DataFrame(reverted_rf_predictions)
predictions_df.to_csv('housing_y_test.csv', sep=',', encoding='utf-8', index=False)
#np.savetxt('housing_y_test.csv', reverted_ridge_predictions, delimiter=",", fmt="%g")

