#import mediapipe as mp
from sklearn.metrics import accuracy_score
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
import pandas as pd
import numpy as np
from sklearn.metrics import mean_squared_error
from sklearn.metrics import mean_absolute_error
from sklearn.metrics import median_absolute_error
#from sklearn.metrics import r2_score
# from sklearn.metrics import roc_auc_score
from sklearn.metrics import explained_variance_score
from numpy import trapz
from pathlib import Path
import os
directory = os.getcwd()
FolderName = "\V2_50"
Folder = directory + FolderName
Path(Folder).mkdir(parents=True, exist_ok=True)
Results = []
ResultsColumn =[]









# *******************************************
# Linear Regression 2 Seperate Without Limit
# *******************************************

ResultsColumn.append("Linear Single Without Limit")
Results.append("-----")
ResultsColumn.extend(["MEA_X", "MEA_Y", "RMSE_X", "RMSE_Y", "Variance_X", "Variance_Y", "Median_X", "Median_Y", "AUC_X", "AUC_Y"])

Testing = pd.read_csv(Folder + '\Testing.csv')
Training = pd.read_csv(Folder  + '\Training.csv')
TrainingLeftXInput = Training['LeftXInput'].values.reshape(-1, 1)
TrainingRightXInput = Training['RightXInput'].values.reshape(-1, 1)
TrainingX = pd.DataFrame(list(zip(TrainingLeftXInput, TrainingRightXInput)))
TrainingLeftYInput = Training['LeftYInput'].values.reshape(-1, 1)
TrainingRightYInput = Training['RightYInput'].values.reshape(-1, 1)
TrainingY = pd.DataFrame(list(zip(TrainingLeftYInput, TrainingRightYInput)))
TrainingXVal = Training['XVal'].values.reshape(-1, 1)
TrainingYVal = Training['YVal'].values.reshape(-1, 1)
LinearRegressionX = LinearRegression()
LinearRegressionY = LinearRegression()
LinearRegressionX.fit(TrainingX, TrainingXVal)
LinearRegressionY.fit(TrainingY, TrainingYVal)
TestingLeftXInput = Testing['LeftXInput'].values.reshape(-1, 1)
TestingRightXInput = Testing['RightXInput'].values.reshape(-1, 1)
TestingX = pd.DataFrame(list(zip(TestingLeftXInput, TestingRightXInput)))
TestingLeftYInput = Testing['LeftYInput'].values.reshape(-1, 1)
TestingRightYInput = Testing['RightYInput'].values.reshape(-1, 1)
TestingY = pd.DataFrame(list(zip(TestingLeftYInput, TestingRightYInput)))
TestingXVal = Testing['XVal'].values.reshape(-1, 1)
TestingYVal = Testing['YVal'].values.reshape(-1, 1)
PredictX = LinearRegressionX.predict(TestingX)
PredictY = LinearRegressionY.predict(TestingY)
Results.append(mean_absolute_error(TestingXVal, PredictX)) # MAE X
Results.append(mean_absolute_error(TestingYVal, PredictY)) # MAE Y
Results.append(mean_squared_error(TestingXVal, PredictX, squared=True)) # RMSE X
Results.append(mean_squared_error(TestingYVal, PredictY, squared=True)) # RMSE Y
Results.append(explained_variance_score(TestingXVal, PredictX)) # Variance X
Results.append(explained_variance_score(TestingYVal, PredictY)) # Variance Y
Results.append(median_absolute_error(TestingXVal, PredictX)) # Media X
Results.append(median_absolute_error(TestingYVal, PredictY)) # Median Y
AreaInitial = trapz(PredictX.flatten(), dx=1)
AreaFinal = trapz(TestingXVal.flatten(), dx=1)
Results.append(AreaInitial/AreaFinal) # AOC X
AreaInitial = trapz(PredictY.flatten(), dx=1)
AreaFinal = trapz(TestingYVal.flatten(), dx=1)
Results.append(AreaInitial/AreaFinal)  # AOC Y


# *******************************************
# Linear Regression 2 Seperate WITH Limit
# *******************************************

ResultsColumn.append("Linear Single WITH Limit")
Results.append("-----")
ResultsColumn.extend(["MEA_X", "MEA_Y", "RMSE_X", "RMSE_Y", "Variance_X", "Variance_Y", "Median_X", "Median_Y", "AUC_X", "AUC_Y"])

Testing = pd.read_csv(Folder + '\Testing.csv')
Training = pd.read_csv(Folder  + '\Training.csv')
TrainingLeftXInput = Training['LeftXInput'].values.reshape(-1, 1)
TrainingRightXInput = Training['RightXInput'].values.reshape(-1, 1)
TrainingX = pd.DataFrame(list(zip(TrainingLeftXInput, TrainingRightXInput)))
TrainingLeftYInput = Training['LeftYInput'].values.reshape(-1, 1)
TrainingRightYInput = Training['RightYInput'].values.reshape(-1, 1)
TrainingY = pd.DataFrame(list(zip(TrainingLeftYInput, TrainingRightYInput)))
TrainingXVal = Training['XVal'].values.reshape(-1, 1)
TrainingYVal = Training['YVal'].values.reshape(-1, 1)
LinearRegressionX = LinearRegression()
LinearRegressionY = LinearRegression()
LinearRegressionX.fit(TrainingX, TrainingXVal)
LinearRegressionY.fit(TrainingY, TrainingYVal)
TestingLeftXInput = Testing['LeftXInput'].values.reshape(-1, 1)
TestingRightXInput = Testing['RightXInput'].values.reshape(-1, 1)
TestingX = pd.DataFrame(list(zip(TestingLeftXInput, TestingRightXInput)))
TestingLeftYInput = Testing['LeftYInput'].values.reshape(-1, 1)
TestingRightYInput = Testing['RightYInput'].values.reshape(-1, 1)
TestingY = pd.DataFrame(list(zip(TestingLeftYInput, TestingRightYInput)))
TestingXVal = Testing['XVal'].values.reshape(-1, 1)
TestingYVal = Testing['YVal'].values.reshape(-1, 1)
PredictX = LinearRegressionX.predict(TestingX)
PredictX = [0 if i < 0 else int(i) for i in PredictX]
PredictX = [2560 if i > 2560 else int(i) for i in PredictX]
PredictX = np.array(PredictX)
PredictY = LinearRegressionY.predict(TestingY)
PredictY = [0 if i < 0 else int(i) for i in PredictY]
PredictY = [1600 if i > 1600 else int(i) for i in PredictY]
PredictY = np.array(PredictY)
Results.append(mean_absolute_error(TestingXVal, PredictX)) # MAE X
Results.append(mean_absolute_error(TestingYVal, PredictY)) # MAE Y
Results.append(mean_squared_error(TestingXVal, PredictX, squared=True)) # RMSE X
Results.append(mean_squared_error(TestingYVal, PredictY, squared=True)) # RMSE Y
Results.append(explained_variance_score(TestingXVal, PredictX)) # Variance X
Results.append(explained_variance_score(TestingYVal, PredictY)) # Variance Y
Results.append(median_absolute_error(TestingXVal, PredictX)) # Media X
Results.append(median_absolute_error(TestingYVal, PredictY)) # Median Y
AreaInitial = trapz(PredictX.flatten(), dx=1)
AreaFinal = trapz(TestingXVal.flatten(), dx=1)
Results.append(AreaInitial/AreaFinal) # AOC X
AreaInitial = trapz(PredictY.flatten(), dx=1)
AreaFinal = trapz(TestingYVal.flatten(), dx=1)
Results.append(AreaInitial/AreaFinal)  # AOC Y



# *******************************************
# Linear Regression Combined Without Limit
# *******************************************

ResultsColumn.append("Linear Combined Without Limit")
Results.append("-----")
ResultsColumn.extend(["MEA_X", "MEA_Y", "RMSE_X", "RMSE_Y", "Variance_X", "Variance_Y", "Median_X", "Median_Y", "AUC_X", "AUC_Y"])


Testing = pd.read_csv(Folder + '\Testing.csv')
Training = pd.read_csv(Folder + '\Training.csv')
TrainingLeftXInput = Training['LeftXInput'].values.reshape(-1, 1)
TrainingRightXInput = Training['RightXInput'].values.reshape(-1, 1)
TrainingLeftYInput = Training['LeftYInput'].values.reshape(-1, 1)
TrainingYRightInput = Training['RightYInput'].values.reshape(-1, 1)
TrainingXVal = Training['XVal'].values.reshape(-1, 1)
TrainingYVal = Training['YVal'].values.reshape(-1, 1)
LinearRegressionX = LinearRegression()
LinearRegressionY = LinearRegression()
TrainingDF = pd.DataFrame(list(zip(TrainingLeftXInput, TrainingRightXInput, TrainingLeftYInput, TrainingYRightInput)))
LinearRegressionX.fit(TrainingDF, TrainingXVal)
LinearRegressionY.fit(TrainingDF, TrainingYVal)
TestingLeftXInput = Testing['LeftXInput'].values.reshape(-1, 1)
TestingLeftYInput = Testing['LeftYInput'].values.reshape(-1, 1)
TestingRightXInput = Testing['RightXInput'].values.reshape(-1, 1)
TestingYRightInput = Testing['RightYInput'].values.reshape(-1, 1)
TestingDF = pd.DataFrame(list(zip(TestingLeftXInput, TestingRightXInput, TestingLeftYInput, TestingYRightInput)))
TestingXVal = Testing['XVal'].values.reshape(-1, 1)
TestingYVal = Testing['YVal'].values.reshape(-1, 1)
PredictX = LinearRegressionX.predict(TestingDF)
PredictY = LinearRegressionY.predict(TestingDF)
Results.append(mean_absolute_error(TestingXVal, PredictX)) # MAE X
Results.append(mean_absolute_error(TestingYVal, PredictY)) # MAE Y
Results.append(mean_squared_error(TestingXVal, PredictX, squared=True)) # RMSE X
Results.append(mean_squared_error(TestingYVal, PredictY, squared=True)) # RMSE Y
Results.append(explained_variance_score(TestingXVal, PredictX)) # Variance X
Results.append(explained_variance_score(TestingYVal, PredictY)) # Variance Y
Results.append(median_absolute_error(TestingXVal, PredictX)) # Media X
Results.append(median_absolute_error(TestingYVal, PredictY)) # Median Y
AreaInitial = trapz(PredictX.flatten(), dx=1)
AreaFinal = trapz(TestingXVal.flatten(), dx=1)
Results.append(AreaInitial/AreaFinal) # AOC X
AreaInitial = trapz(PredictY.flatten(), dx=1)
AreaFinal = trapz(TestingYVal.flatten(), dx=1)
Results.append(AreaInitial/AreaFinal)  # AOC Y
PredictX = PredictX.flatten()
PredictY = PredictY.flatten()


# *******************************************
# Linear Regression Combined WITH Limit
# *******************************************

ResultsColumn.append("Linear Combined WITH Limit")
Results.append("-----")
ResultsColumn.extend(["MEA_X", "MEA_Y", "RMSE_X", "RMSE_Y", "Variance_X", "Variance_Y", "Median_X", "Median_Y", "AUC_X", "AUC_Y"])


Testing = pd.read_csv(Folder + '\Testing.csv')
Training = pd.read_csv(Folder + '\Training.csv')
TrainingLeftXInput = Training['LeftXInput'].values.reshape(-1, 1)
TrainingRightXInput = Training['RightXInput'].values.reshape(-1, 1)
TrainingLeftYInput = Training['LeftYInput'].values.reshape(-1, 1)
TrainingYRightInput = Training['RightYInput'].values.reshape(-1, 1)
TrainingXVal = Training['XVal'].values.reshape(-1, 1)
TrainingYVal = Training['YVal'].values.reshape(-1, 1)
LinearRegressionX = LinearRegression()
LinearRegressionY = LinearRegression()
TrainingDF = pd.DataFrame(list(zip(TrainingLeftXInput, TrainingRightXInput, TrainingLeftYInput, TrainingYRightInput)))
LinearRegressionX.fit(TrainingDF, TrainingXVal)
LinearRegressionY.fit(TrainingDF, TrainingYVal)
TestingLeftXInput = Testing['LeftXInput'].values.reshape(-1, 1)
TestingLeftYInput = Testing['LeftYInput'].values.reshape(-1, 1)
TestingRightXInput = Testing['RightXInput'].values.reshape(-1, 1)
TestingYRightInput = Testing['RightYInput'].values.reshape(-1, 1)
TestingDF = pd.DataFrame(list(zip(TestingLeftXInput, TestingRightXInput, TestingLeftYInput, TestingYRightInput)))
TestingXVal = Testing['XVal'].values.reshape(-1, 1)
TestingYVal = Testing['YVal'].values.reshape(-1, 1)
PredictX = LinearRegressionX.predict(TestingDF)
PredictX = [0 if i < 0 else int(i) for i in PredictX]
PredictX = [2560 if i > 2560 else int(i) for i in PredictX]
PredictX = np.array(PredictX)
PredictY = LinearRegressionY.predict(TestingDF)
PredictY = [0 if i < 0 else int(i) for i in PredictY]
PredictY = [1600 if i > 1600 else int(i) for i in PredictY]
PredictY = np.array(PredictY)
Results.append(mean_absolute_error(TestingXVal, PredictX)) # MAE X
Results.append(mean_absolute_error(TestingYVal, PredictY)) # MAE Y
Results.append(mean_squared_error(TestingXVal, PredictX, squared=True)) # RMSE X
Results.append(mean_squared_error(TestingYVal, PredictY, squared=True)) # RMSE Y
Results.append(explained_variance_score(TestingXVal, PredictX)) # Variance X
Results.append(explained_variance_score(TestingYVal, PredictY)) # Variance Y
Results.append(median_absolute_error(TestingXVal, PredictX)) # Media X
Results.append(median_absolute_error(TestingYVal, PredictY)) # Median Y
AreaInitial = trapz(PredictX.flatten(), dx=1)
AreaFinal = trapz(TestingXVal.flatten(), dx=1)
Results.append(AreaInitial/AreaFinal) # AOC X
AreaInitial = trapz(PredictY.flatten(), dx=1)
AreaFinal = trapz(TestingYVal.flatten(), dx=1)
Results.append(AreaInitial/AreaFinal)  # AOC Y
PredictX = PredictX.flatten()
PredictY = PredictY.flatten()




# *******************************************
# Random Forest Without Limit
# *******************************************

ResultsColumn.append("RF Without Limit")
Results.append("-----")
ResultsColumn.extend(["MEA_X", "MEA_Y", "RMSE_X", "RMSE_Y", "Variance_X", "Variance_Y", "Median_X", "Median_Y", "AUC_X", "AUC_Y"])

from sklearn.ensemble import RandomForestRegressor
Testing = pd.read_csv(Folder + '\Testing.csv')
Training = pd.read_csv(Folder + '\Training.csv')
TrainingLeftXInput = Training['LeftXInput'].values.reshape(-1, 1)
TrainingRightXInput = Training['RightXInput'].values.reshape(-1, 1)
TrainingX = pd.DataFrame(list(zip(TrainingLeftXInput, TrainingRightXInput)))
TrainingLeftYInput = Training['LeftYInput'].values.reshape(-1, 1)
TrainingRightYInput = Training['RightYInput'].values.reshape(-1, 1)
TrainingY = pd.DataFrame(list(zip(TrainingLeftYInput, TrainingRightYInput)))
TrainingXVal = Training['XVal'].values.reshape(-1, 1)
TrainingYVal = Training['YVal'].values.reshape(-1, 1)
LinearRegressionX = RandomForestRegressor()
LinearRegressionY = RandomForestRegressor()
LinearRegressionX.fit(TrainingX, TrainingXVal)
LinearRegressionY.fit(TrainingY, TrainingYVal)
TestingLeftXInput = Testing['LeftXInput'].values.reshape(-1, 1)
TestingRightXInput = Testing['RightXInput'].values.reshape(-1, 1)
TestingX = pd.DataFrame(list(zip(TestingLeftXInput, TestingRightXInput)))
TestingLeftYInput = Testing['LeftYInput'].values.reshape(-1, 1)
TestingRightYInput = Testing['RightYInput'].values.reshape(-1, 1)
TestingY = pd.DataFrame(list(zip(TestingLeftYInput, TestingRightYInput)))
TestingXVal = Testing['XVal'].values.reshape(-1, 1)
TestingYVal = Testing['YVal'].values.reshape(-1, 1)
PredictX = LinearRegressionX.predict(TestingX)
PredictY = LinearRegressionY.predict(TestingY)
Results.append(mean_absolute_error(TestingXVal, PredictX)) # MAE X
Results.append(mean_absolute_error(TestingYVal, PredictY)) # MAE Y
Results.append(mean_squared_error(TestingXVal, PredictX, squared=True)) # RMSE X
Results.append(mean_squared_error(TestingYVal, PredictY, squared=True)) # RMSE Y
Results.append(explained_variance_score(TestingXVal, PredictX)) # Variance X
Results.append(explained_variance_score(TestingYVal, PredictY)) # Variance Y
Results.append(median_absolute_error(TestingXVal, PredictX)) # Media X
Results.append(median_absolute_error(TestingYVal, PredictY)) # Median Y
AreaInitial = trapz(PredictX.flatten(), dx=1)
AreaFinal = trapz(TestingXVal.flatten(), dx=1)
Results.append(AreaInitial/AreaFinal) # AOC X
AreaInitial = trapz(PredictY.flatten(), dx=1)
AreaFinal = trapz(TestingYVal.flatten(), dx=1)
Results.append(AreaInitial/AreaFinal)  # AOC Y





# *******************************************
# Random Forest With Limit
# *******************************************

ResultsColumn.append("RF With Limit")
Results.append("-----")
ResultsColumn.extend(["MEA_X", "MEA_Y", "RMSE_X", "RMSE_Y", "Variance_X", "Variance_Y", "Median_X", "Median_Y", "AUC_X", "AUC_Y"])

from sklearn.ensemble import RandomForestRegressor
Testing = pd.read_csv(Folder + '\Testing.csv')
Training = pd.read_csv(Folder + '\Training.csv')
TrainingLeftXInput = Training['LeftXInput'].values.reshape(-1, 1)
TrainingRightXInput = Training['RightXInput'].values.reshape(-1, 1)
TrainingX = pd.DataFrame(list(zip(TrainingLeftXInput, TrainingRightXInput)))
TrainingLeftYInput = Training['LeftYInput'].values.reshape(-1, 1)
TrainingRightYInput = Training['RightYInput'].values.reshape(-1, 1)
TrainingY = pd.DataFrame(list(zip(TrainingLeftYInput, TrainingRightYInput)))
TrainingXVal = Training['XVal'].values.reshape(-1, 1)
TrainingYVal = Training['YVal'].values.reshape(-1, 1)
LinearRegressionX = RandomForestRegressor()
LinearRegressionY = RandomForestRegressor()
LinearRegressionX.fit(TrainingX, TrainingXVal)
LinearRegressionY.fit(TrainingY, TrainingYVal)
TestingLeftXInput = Testing['LeftXInput'].values.reshape(-1, 1)
TestingRightXInput = Testing['RightXInput'].values.reshape(-1, 1)
TestingX = pd.DataFrame(list(zip(TestingLeftXInput, TestingRightXInput)))
TestingLeftYInput = Testing['LeftYInput'].values.reshape(-1, 1)
TestingRightYInput = Testing['RightYInput'].values.reshape(-1, 1)
TestingY = pd.DataFrame(list(zip(TestingLeftYInput, TestingRightYInput)))
TestingXVal = Testing['XVal'].values.reshape(-1, 1)
TestingYVal = Testing['YVal'].values.reshape(-1, 1)
PredictX = LinearRegressionX.predict(TestingX)
PredictX = [0 if i < 0 else int(i) for i in PredictX]
PredictX = [2560 if i > 2560 else int(i) for i in PredictX]
PredictX = np.array(PredictX)
PredictY = LinearRegressionY.predict(TestingY)
PredictY = [0 if i < 0 else int(i) for i in PredictY]
PredictY = [1600 if i > 1600 else int(i) for i in PredictY]
PredictY = np.array(PredictY)
Results.append(mean_absolute_error(TestingXVal, PredictX)) # MAE X
Results.append(mean_absolute_error(TestingYVal, PredictY)) # MAE Y
Results.append(mean_squared_error(TestingXVal, PredictX, squared=True)) # RMSE X
Results.append(mean_squared_error(TestingYVal, PredictY, squared=True)) # RMSE Y
Results.append(explained_variance_score(TestingXVal, PredictX)) # Variance X
Results.append(explained_variance_score(TestingYVal, PredictY)) # Variance Y
Results.append(median_absolute_error(TestingXVal, PredictX)) # Media X
Results.append(median_absolute_error(TestingYVal, PredictY)) # Median Y
AreaInitial = trapz(PredictX.flatten(), dx=1)
AreaFinal = trapz(TestingXVal.flatten(), dx=1)
Results.append(AreaInitial/AreaFinal) # AOC X
AreaInitial = trapz(PredictY.flatten(), dx=1)
AreaFinal = trapz(TestingYVal.flatten(), dx=1)
Results.append(AreaInitial/AreaFinal)  # AOC Y






# *******************************************
# GPR Without Limit
# *******************************************

ResultsColumn.append("GPR Without Limit")
Results.append("-----")
ResultsColumn.extend(["MEA_X", "MEA_Y", "RMSE_X", "RMSE_Y", "Variance_X", "Variance_Y", "Median_X", "Median_Y", "AUC_X", "AUC_Y"])

from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.preprocessing import MinMaxScaler
from sklearn.gaussian_process.kernels import RBF, DotProduct, WhiteKernel
scalerX = MinMaxScaler()
scalerY = MinMaxScaler()
kernel =  RBF() + WhiteKernel() + DotProduct() 
Testing = pd.read_csv(Folder + '\Testing.csv')
Training = pd.read_csv(Folder + '\Training.csv')
TrainingLeftXInput = Training['LeftXInput'].values.reshape(-1, 1)
TrainingRightXInput = Training['RightXInput'].values.reshape(-1, 1)
TrainingX = pd.DataFrame(list(zip(TrainingLeftXInput, TrainingRightXInput)))
TrainingLeftYInput = Training['LeftYInput'].values.reshape(-1, 1)
TrainingRightYInput = Training['RightYInput'].values.reshape(-1, 1)
TrainingY = pd.DataFrame(list(zip(TrainingLeftYInput, TrainingRightYInput)))
TrainingXInput = scalerX.fit_transform(TrainingX)
TrainingYInput = scalerY.fit_transform(TrainingY)
TrainingXVal = Training['XVal'].values.reshape(-1, 1)
TrainingYVal = Training['YVal'].values.reshape(-1, 1)
LinearRegressionX = GaussianProcessRegressor(kernel=kernel)
LinearRegressionY = GaussianProcessRegressor(kernel=kernel)
LinearRegressionX.fit(TrainingXInput, TrainingXVal)
LinearRegressionY.fit(TrainingYInput, TrainingYVal)
TestingLeftXInput = Testing['LeftXInput'].values.reshape(-1, 1)
TestingRightXInput = Testing['RightXInput'].values.reshape(-1, 1)
TestingX = pd.DataFrame(list(zip(TestingLeftXInput, TestingRightXInput)))
TestingXInput = scalerX.transform(TestingX)
TestingLeftYInput = Testing['LeftYInput'].values.reshape(-1, 1)
TestingRightYInput = Testing['RightYInput'].values.reshape(-1, 1)
TestingY = pd.DataFrame(list(zip(TestingLeftYInput, TestingRightYInput)))
TestingYInput = scalerY.transform(TestingY)
TestingXVal = Testing['XVal'].values.reshape(-1, 1)
TestingYVal = Testing['YVal'].values.reshape(-1, 1)
PredictX = LinearRegressionX.predict(TestingXInput)
PredictY = LinearRegressionY.predict(TestingYInput)
Results.append(mean_absolute_error(TestingXVal, PredictX)) # MAE X
Results.append(mean_absolute_error(TestingYVal, PredictY)) # MAE Y
Results.append(mean_squared_error(TestingXVal, PredictX, squared=True)) # RMSE X
Results.append(mean_squared_error(TestingYVal, PredictY, squared=True)) # RMSE Y
Results.append(explained_variance_score(TestingXVal, PredictX)) # Variance X
Results.append(explained_variance_score(TestingYVal, PredictY)) # Variance Y
Results.append(median_absolute_error(TestingXVal, PredictX)) # Media X
Results.append(median_absolute_error(TestingYVal, PredictY)) # Median Y
AreaInitial = trapz(PredictX.flatten(), dx=1)
AreaFinal = trapz(TestingXVal.flatten(), dx=1)
Results.append(AreaInitial/AreaFinal) # AOC X
AreaInitial = trapz(PredictY.flatten(), dx=1)
AreaFinal = trapz(TestingYVal.flatten(), dx=1)
Results.append(AreaInitial/AreaFinal)  # AOC Y
PredictX = PredictX.flatten()
PredictY = PredictY.flatten()



# *******************************************
# GPR With Limit
# *******************************************

ResultsColumn.append("GPR WITH Limit")
Results.append("-----")
ResultsColumn.extend(["MEA_X", "MEA_Y", "RMSE_X", "RMSE_Y", "Variance_X", "Variance_Y", "Median_X", "Median_Y", "AUC_X", "AUC_Y"])

from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.preprocessing import MinMaxScaler
from sklearn.gaussian_process.kernels import RBF, DotProduct, WhiteKernel
scalerX = MinMaxScaler()
scalerY = MinMaxScaler()
kernel =  RBF() + WhiteKernel() + DotProduct() 
Testing = pd.read_csv(Folder + '\Testing.csv')
Training = pd.read_csv(Folder + '\Training.csv')
TrainingLeftXInput = Training['LeftXInput'].values.reshape(-1, 1)
TrainingRightXInput = Training['RightXInput'].values.reshape(-1, 1)
TrainingX = pd.DataFrame(list(zip(TrainingLeftXInput, TrainingRightXInput)))
TrainingLeftYInput = Training['LeftYInput'].values.reshape(-1, 1)
TrainingRightYInput = Training['RightYInput'].values.reshape(-1, 1)
TrainingY = pd.DataFrame(list(zip(TrainingLeftYInput, TrainingRightYInput)))
TrainingXInput = scalerX.fit_transform(TrainingX)
TrainingYInput = scalerY.fit_transform(TrainingY)
TrainingXVal = Training['XVal'].values.reshape(-1, 1)
TrainingYVal = Training['YVal'].values.reshape(-1, 1)
LinearRegressionX = GaussianProcessRegressor(kernel=kernel)
LinearRegressionY = GaussianProcessRegressor(kernel=kernel)
LinearRegressionX.fit(TrainingXInput, TrainingXVal)
LinearRegressionY.fit(TrainingYInput, TrainingYVal)
TestingLeftXInput = Testing['LeftXInput'].values.reshape(-1, 1)
TestingRightXInput = Testing['RightXInput'].values.reshape(-1, 1)
TestingX = pd.DataFrame(list(zip(TestingLeftXInput, TestingRightXInput)))
TestingXInput = scalerX.transform(TestingX)
TestingLeftYInput = Testing['LeftYInput'].values.reshape(-1, 1)
TestingRightYInput = Testing['RightYInput'].values.reshape(-1, 1)
TestingY = pd.DataFrame(list(zip(TestingLeftYInput, TestingRightYInput)))
TestingYInput = scalerY.transform(TestingY)
TestingXVal = Testing['XVal'].values.reshape(-1, 1)
TestingYVal = Testing['YVal'].values.reshape(-1, 1)
PredictX = LinearRegressionX.predict(TestingXInput)
PredictX = [0 if i < 0 else int(i) for i in PredictX]
PredictX = [2560 if i > 2560 else int(i) for i in PredictX]
PredictX = np.array(PredictX)
PredictY = LinearRegressionY.predict(TestingYInput)
PredictY = [0 if i < 0 else int(i) for i in PredictY]
PredictY = [1600 if i > 1600 else int(i) for i in PredictY]
PredictY = np.array(PredictY)
Results.append(mean_absolute_error(TestingXVal, PredictX)) # MAE X
Results.append(mean_absolute_error(TestingYVal, PredictY)) # MAE Y
Results.append(mean_squared_error(TestingXVal, PredictX, squared=True)) # RMSE X
Results.append(mean_squared_error(TestingYVal, PredictY, squared=True)) # RMSE Y
Results.append(explained_variance_score(TestingXVal, PredictX)) # Variance X
Results.append(explained_variance_score(TestingYVal, PredictY)) # Variance Y
Results.append(median_absolute_error(TestingXVal, PredictX)) # Media X
Results.append(median_absolute_error(TestingYVal, PredictY)) # Median Y
AreaInitial = trapz(PredictX.flatten(), dx=1)
AreaFinal = trapz(TestingXVal.flatten(), dx=1)
Results.append(AreaInitial/AreaFinal) # AOC X
AreaInitial = trapz(PredictY.flatten(), dx=1)
AreaFinal = trapz(TestingYVal.flatten(), dx=1)
Results.append(AreaInitial/AreaFinal)  # AOC Y
PredictX = PredictX.flatten()
PredictY = PredictY.flatten()



# *******************************************
# Logistic Regression Without Limit
# *******************************************


ResultsColumn.append("Logistic Without Limit")
Results.append("-----")
ResultsColumn.extend(["MEA_X", "MEA_Y", "RMSE_X", "RMSE_Y", "Variance_X", "Variance_Y", "Median_X", "Median_Y", "AUC_X", "AUC_Y"])

from sklearn import linear_model
Testing = pd.read_csv(Folder + '\Testing.csv')
Training = pd.read_csv(Folder  + '\Training.csv')
TrainingLeftXInput = Training['LeftXInput'].values.reshape(-1, 1)
TrainingRightXInput = Training['RightXInput'].values.reshape(-1, 1)
TrainingX = pd.DataFrame(list(zip(TrainingLeftXInput, TrainingRightXInput)))
TrainingLeftYInput = Training['LeftYInput'].values.reshape(-1, 1)
TrainingRightYInput = Training['RightYInput'].values.reshape(-1, 1)
TrainingY = pd.DataFrame(list(zip(TrainingLeftYInput, TrainingRightYInput)))
TrainingXVal = Training['XVal'].values.reshape(-1, 1)
TrainingYVal = Training['YVal'].values.reshape(-1, 1)
LinearRegressionX = linear_model.LogisticRegression()
LinearRegressionY = linear_model.LogisticRegression()
LinearRegressionX.fit(TrainingX, TrainingXVal)
LinearRegressionY.fit(TrainingY, TrainingYVal)
TestingLeftXInput = Testing['LeftXInput'].values.reshape(-1, 1)
TestingRightXInput = Testing['RightXInput'].values.reshape(-1, 1)
TestingX = pd.DataFrame(list(zip(TestingLeftXInput, TestingRightXInput)))
TestingLeftYInput = Testing['LeftYInput'].values.reshape(-1, 1)
TestingRightYInput = Testing['RightYInput'].values.reshape(-1, 1)
TestingY = pd.DataFrame(list(zip(TestingLeftYInput, TestingRightYInput)))
TestingXVal = Testing['XVal'].values.reshape(-1, 1)
TestingYVal = Testing['YVal'].values.reshape(-1, 1)
PredictX = LinearRegressionX.predict(TestingX)
PredictY = LinearRegressionY.predict(TestingY)
Results.append(mean_absolute_error(TestingXVal, PredictX)) # MAE X
Results.append(mean_absolute_error(TestingYVal, PredictY)) # MAE Y
Results.append(mean_squared_error(TestingXVal, PredictX, squared=True)) # RMSE X
Results.append(mean_squared_error(TestingYVal, PredictY, squared=True)) # RMSE Y
Results.append(explained_variance_score(TestingXVal, PredictX)) # Variance X
Results.append(explained_variance_score(TestingYVal, PredictY)) # Variance Y
Results.append(median_absolute_error(TestingXVal, PredictX)) # Media X
Results.append(median_absolute_error(TestingYVal, PredictY)) # Median Y
AreaInitial = trapz(PredictX.flatten(), dx=1)
AreaFinal = trapz(TestingXVal.flatten(), dx=1)
Results.append(AreaInitial/AreaFinal) # AOC X
AreaInitial = trapz(PredictY.flatten(), dx=1)
AreaFinal = trapz(TestingYVal.flatten(), dx=1)
Results.append(AreaInitial/AreaFinal)  # AOC Y



# *******************************************
# Logistic Regression With Limit
# *******************************************


ResultsColumn.append("Logistic With Limit")
Results.append("-----")
ResultsColumn.extend(["MEA_X", "MEA_Y", "RMSE_X", "RMSE_Y", "Variance_X", "Variance_Y", "Median_X", "Median_Y", "AUC_X", "AUC_Y"])

from sklearn import linear_model
Testing = pd.read_csv(Folder + '\Testing.csv')
Training = pd.read_csv(Folder  + '\Training.csv')
TrainingLeftXInput = Training['LeftXInput'].values.reshape(-1, 1)
TrainingRightXInput = Training['RightXInput'].values.reshape(-1, 1)
TrainingX = pd.DataFrame(list(zip(TrainingLeftXInput, TrainingRightXInput)))
TrainingLeftYInput = Training['LeftYInput'].values.reshape(-1, 1)
TrainingRightYInput = Training['RightYInput'].values.reshape(-1, 1)
TrainingY = pd.DataFrame(list(zip(TrainingLeftYInput, TrainingRightYInput)))
TrainingXVal = Training['XVal'].values.reshape(-1, 1)
TrainingYVal = Training['YVal'].values.reshape(-1, 1)
LinearRegressionX = linear_model.LogisticRegression()
LinearRegressionY = linear_model.LogisticRegression()
LinearRegressionX.fit(TrainingX, TrainingXVal)
LinearRegressionY.fit(TrainingY, TrainingYVal)
TestingLeftXInput = Testing['LeftXInput'].values.reshape(-1, 1)
TestingRightXInput = Testing['RightXInput'].values.reshape(-1, 1)
TestingX = pd.DataFrame(list(zip(TestingLeftXInput, TestingRightXInput)))
TestingLeftYInput = Testing['LeftYInput'].values.reshape(-1, 1)
TestingRightYInput = Testing['RightYInput'].values.reshape(-1, 1)
TestingY = pd.DataFrame(list(zip(TestingLeftYInput, TestingRightYInput)))
TestingXVal = Testing['XVal'].values.reshape(-1, 1)
TestingYVal = Testing['YVal'].values.reshape(-1, 1)
PredictX = LinearRegressionX.predict(TestingX)
PredictX = [0 if i < 0 else int(i) for i in PredictX]
PredictX = [2560 if i > 2560 else int(i) for i in PredictX]
PredictX = np.array(PredictX)
PredictY = LinearRegressionY.predict(TestingY)
PredictY = [0 if i < 0 else int(i) for i in PredictY]
PredictY = [1600 if i > 1600 else int(i) for i in PredictY]
PredictY = np.array(PredictY)
Results.append(mean_absolute_error(TestingXVal, PredictX)) # MAE X
Results.append(mean_absolute_error(TestingYVal, PredictY)) # MAE Y
Results.append(mean_squared_error(TestingXVal, PredictX, squared=True)) # RMSE X
Results.append(mean_squared_error(TestingYVal, PredictY, squared=True)) # RMSE Y
Results.append(explained_variance_score(TestingXVal, PredictX)) # Variance X
Results.append(explained_variance_score(TestingYVal, PredictY)) # Variance Y
Results.append(median_absolute_error(TestingXVal, PredictX)) # Media X
Results.append(median_absolute_error(TestingYVal, PredictY)) # Median Y
AreaInitial = trapz(PredictX.flatten(), dx=1)
AreaFinal = trapz(TestingXVal.flatten(), dx=1)
Results.append(AreaInitial/AreaFinal) # AOC X
AreaInitial = trapz(PredictY.flatten(), dx=1)
AreaFinal = trapz(TestingYVal.flatten(), dx=1)
Results.append(AreaInitial/AreaFinal)  # AOC Y




# *******************************************
# Parametric Without Limit
# *******************************************

ResultsColumn.append("Parametric Without Limit")
Results.append("-----")
ResultsColumn.extend(["MEA_X", "MEA_Y", "RMSE_X", "RMSE_Y", "Variance_X", "Variance_Y", "Median_X", "Median_Y", "AUC_X", "AUC_Y"])


Testing = pd.read_csv(Folder + '\Testing.csv')
Training = pd.read_csv(Folder + '\Training.csv')
TrainingLeftXInput = Training['LeftXInput'].values.reshape(-1, 1)
TrainingRightXInput = Training['RightXInput'].values.reshape(-1, 1)
TrainingLeftDia = Training['LeftDiaInput'].values.reshape(-1,1)
TrainingRightDia = Training['RightDiaInput'].values.reshape(-1,1)
TrainingFaceXInput = Training['FaceXinput'].values.reshape(-1,1)
TrainingX = pd.DataFrame(list(zip(TrainingLeftXInput, TrainingRightXInput, TrainingLeftDia, TrainingRightDia, TrainingFaceXInput)))
TrainingLeftYInput = Training['LeftYInput'].values.reshape(-1, 1)
TrainingRightYInput = Training['RightYInput'].values.reshape(-1, 1)
TrainingLeftDia = Training['LeftDiaInput'].values.reshape(-1,1)
TrainingRightDia = Training['RightDiaInput'].values.reshape(-1,1)
TrainingFaceYInput = Training['FaceYinput'].values.reshape(-1,1)
TrainingY = pd.DataFrame(list(zip(TrainingLeftYInput, TrainingRightYInput, TrainingLeftDia, TrainingRightDia, TrainingFaceYInput)))
TrainingXVal = Training['XVal'].values.reshape(-1, 1)
TrainingYVal = Training['YVal'].values.reshape(-1, 1)
LinearRegressionX = LinearRegression()
LinearRegressionY = LinearRegression()
LinearRegressionX.fit(TrainingX, TrainingXVal)
LinearRegressionY.fit(TrainingY, TrainingYVal)
TestingLeftXInput = Testing['LeftXInput'].values.reshape(-1, 1)
TestingRightXInput = Testing['RightXInput'].values.reshape(-1, 1)
TestingLeftDia = Testing['LeftDiaInput'].values.reshape(-1,1)
TestingRightDia = Testing['RightDiaInput'].values.reshape(-1,1)
TestingFaceXInput = Testing['FaceXinput'].values.reshape(-1,1)
TestingX = pd.DataFrame(list(zip(TestingLeftXInput, TestingRightXInput, TestingLeftDia, TestingRightDia, TestingFaceXInput)))
TestingLeftYInput = Testing['LeftYInput'].values.reshape(-1, 1)
TestingRightYInput = Testing['RightYInput'].values.reshape(-1, 1)
TestingLeftDia = Testing['LeftDiaInput'].values.reshape(-1,1)
TestingRightDia = Testing['RightDiaInput'].values.reshape(-1,1)
TestingFaceYInput = Testing['FaceYinput'].values.reshape(-1,1)
TestingY = pd.DataFrame(list(zip(TestingLeftYInput, TestingRightYInput, TestingLeftDia, TestingRightDia, TestingFaceYInput)))
TestingXVal = Testing['XVal'].values.reshape(-1, 1)
TestingYVal = Testing['YVal'].values.reshape(-1, 1)
PredictX = LinearRegressionX.predict(TestingX)
PredictY = LinearRegressionY.predict(TestingY)
Results.append(mean_absolute_error(TestingXVal, PredictX)) # MAE X
Results.append(mean_absolute_error(TestingYVal, PredictY)) # MAE Y
Results.append(mean_squared_error(TestingXVal, PredictX, squared=True)) # RMSE X
Results.append(mean_squared_error(TestingYVal, PredictY, squared=True)) # RMSE Y
Results.append(explained_variance_score(TestingXVal, PredictX)) # Variance X
Results.append(explained_variance_score(TestingYVal, PredictY)) # Variance Y
Results.append(median_absolute_error(TestingXVal, PredictX)) # Media X
Results.append(median_absolute_error(TestingYVal, PredictY)) # Median Y
AreaInitial = trapz(PredictX.flatten(), dx=1)
AreaFinal = trapz(TestingXVal.flatten(), dx=1)
Results.append(AreaInitial/AreaFinal) # AOC X
AreaInitial = trapz(PredictY.flatten(), dx=1)
AreaFinal = trapz(TestingYVal.flatten(), dx=1)
Results.append(AreaInitial/AreaFinal)  # AOC Y
PredictX = PredictX.flatten()
PredictY = PredictY.flatten()




# *******************************************
# Parametric WITH Limit
# *******************************************

ResultsColumn.append("Parametric WITH Limit")
Results.append("-----")
ResultsColumn.extend(["MEA_X", "MEA_Y", "RMSE_X", "RMSE_Y", "Variance_X", "Variance_Y", "Median_X", "Median_Y", "AUC_X", "AUC_Y"])


Testing = pd.read_csv(Folder + '\Testing.csv')
Training = pd.read_csv(Folder + '\Training.csv')
TrainingLeftXInput = Training['LeftXInput'].values.reshape(-1, 1)
TrainingRightXInput = Training['RightXInput'].values.reshape(-1, 1)
TrainingLeftDia = Training['LeftDiaInput'].values.reshape(-1,1)
TrainingRightDia = Training['RightDiaInput'].values.reshape(-1,1)
TrainingFaceXInput = Training['FaceXinput'].values.reshape(-1,1)
TrainingX = pd.DataFrame(list(zip(TrainingLeftXInput, TrainingRightXInput, TrainingLeftDia, TrainingRightDia, TrainingFaceXInput)))
TrainingLeftYInput = Training['LeftYInput'].values.reshape(-1, 1)
TrainingRightYInput = Training['RightYInput'].values.reshape(-1, 1)
TrainingLeftDia = Training['LeftDiaInput'].values.reshape(-1,1)
TrainingRightDia = Training['RightDiaInput'].values.reshape(-1,1)
TrainingFaceYInput = Training['FaceYinput'].values.reshape(-1,1)
TrainingY = pd.DataFrame(list(zip(TrainingLeftYInput, TrainingRightYInput, TrainingLeftDia, TrainingRightDia, TrainingFaceYInput)))
TrainingXVal = Training['XVal'].values.reshape(-1, 1)
TrainingYVal = Training['YVal'].values.reshape(-1, 1)
LinearRegressionX = LinearRegression()
LinearRegressionY = LinearRegression()
LinearRegressionX.fit(TrainingX, TrainingXVal)
LinearRegressionY.fit(TrainingY, TrainingYVal)
TestingLeftXInput = Testing['LeftXInput'].values.reshape(-1, 1)
TestingRightXInput = Testing['RightXInput'].values.reshape(-1, 1)
TestingLeftDia = Testing['LeftDiaInput'].values.reshape(-1,1)
TestingRightDia = Testing['RightDiaInput'].values.reshape(-1,1)
TestingFaceXInput = Testing['FaceXinput'].values.reshape(-1,1)
TestingX = pd.DataFrame(list(zip(TestingLeftXInput, TestingRightXInput, TestingLeftDia, TestingRightDia, TestingFaceXInput)))
TestingLeftYInput = Testing['LeftYInput'].values.reshape(-1, 1)
TestingRightYInput = Testing['RightYInput'].values.reshape(-1, 1)
TestingLeftDia = Testing['LeftDiaInput'].values.reshape(-1,1)
TestingRightDia = Testing['RightDiaInput'].values.reshape(-1,1)
TestingFaceYInput = Testing['FaceYinput'].values.reshape(-1,1)
TestingY = pd.DataFrame(list(zip(TestingLeftYInput, TestingRightYInput, TestingLeftDia, TestingRightDia, TestingFaceYInput)))
TestingXVal = Testing['XVal'].values.reshape(-1, 1)
TestingYVal = Testing['YVal'].values.reshape(-1, 1)
PredictX = LinearRegressionX.predict(TestingX)
PredictX = [0 if i < 0 else int(i) for i in PredictX]
PredictX = [2560 if i > 2560 else int(i) for i in PredictX]
PredictX = np.array(PredictX)
PredictY = LinearRegressionY.predict(TestingY)
PredictY = [0 if i < 0 else int(i) for i in PredictY]
PredictY = [1600 if i > 1600 else int(i) for i in PredictY]
PredictY = np.array(PredictY)
Results.append(mean_absolute_error(TestingXVal, PredictX)) # MAE X
Results.append(mean_absolute_error(TestingYVal, PredictY)) # MAE Y
Results.append(mean_squared_error(TestingXVal, PredictX, squared=True)) # RMSE X
Results.append(mean_squared_error(TestingYVal, PredictY, squared=True)) # RMSE Y
Results.append(explained_variance_score(TestingXVal, PredictX)) # Variance X
Results.append(explained_variance_score(TestingYVal, PredictY)) # Variance Y
Results.append(median_absolute_error(TestingXVal, PredictX)) # Media X
Results.append(median_absolute_error(TestingYVal, PredictY)) # Median Y
AreaInitial = trapz(PredictX.flatten(), dx=1)
AreaFinal = trapz(TestingXVal.flatten(), dx=1)
Results.append(AreaInitial/AreaFinal) # AOC X
AreaInitial = trapz(PredictY.flatten(), dx=1)
AreaFinal = trapz(TestingYVal.flatten(), dx=1)
Results.append(AreaInitial/AreaFinal)  # AOC Y
PredictX = PredictX.flatten()
PredictY = PredictY.flatten()

df = pd.DataFrame(list(zip(Testing['LeftXInput'].values, Testing['RightXInput'].values, Testing['LeftYInput'].values, Testing['RightYInput'].values, Testing['XVal'].values, PredictX, Testing['YVal'].values, PredictY)), columns= ['TestingLeftXInput', 'TestingRightXInput', 'TestingLeftYInput', 'TestingRightYInput', 'TestingXVal', 'PredictX',  'TestingYVal', 'PredictY'])
df.to_csv(Folder + '\PLR_withLimit_Factor100.csv')


Results = pd.DataFrame(list(zip(ResultsColumn, Results)))
Results.to_csv(Folder + '\Results_V2.csv')

