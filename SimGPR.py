# -*- coding:utf-8 -*-

"""
Spatially-implicit Gaussian Process Regression for Building Height Estimation
@Chen Yang, College of Urban and Environmental Sciences, Peking University
2021/06/11 13:16
Contact: cyangcues@stu.pku.edu.cn
"""
import numpy as np
import sklearn as skl
# import matplotlib as mpl
# import matplotlib.pyplot as plt
import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import RBF, WhiteKernel
from sklearn.gaussian_process.kernels import RBF, ConstantKernel as C
from sklearn.gaussian_process.kernels import WhiteKernel, ExpSineSquared
from sklearn.gaussian_process.kernels import RationalQuadratic
from sklearn.gaussian_process.kernels import ExpSineSquared
from sklearn.gaussian_process.kernels import ConstantKernel, DotProduct
from sklearn.gaussian_process.kernels import Matern
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error
from sklearn.metrics import r2_score

rng = np.random.RandomState(0)

#read
data = pd.read_excel('trial_data1.xlsx')  # replace your reference data here
dt = np.array(data) # please note that you should extract your own reference data as your table
dtx = dt[:,2:11]
dty = dt[:,1].reshape(-1,1)
scaler = StandardScaler()
dtx_std = scaler.fit_transform(dtx)
dty_std = scaler.fit_transform(dty)

x_train, x_test, y_train, y_test = train_test_split(dtx_std,dty_std,test_size=0.3,random_state=0)

# construct RBF kernel
# kernel = C(0.1, (0.001,0.1))*RBF(0.5,(1e-4,10))
# kernel = C(1.0, (1e-3, 1e3)) * RBF(10, (0.5, 2))
# kernel = ExpSineSquared(1.0, 5.0, periodicity_bounds=(1e-2, 1e1)) + WhiteKernel(1e-1)
# kernel = 1.0 * RBF(length_scale=1.0, length_scale_bounds=(1e-2, 1e3)) + WhiteKernel(
#    noise_level=1e-5, noise_level_bounds=(1e-10, 1e1)
#)
# kernel = RBF(length_scale=9.0, length_scale_bounds=(1e-2, 1e3))
# kernel = RationalQuadratic(length_scale=1.0, alpha=1.5)
# kernel = ExpSineSquared(length_scale=1, periodicity=1)
# kernel = ConstantKernel(0.1, (0.01, 10.0)) * (
#    DotProduct(sigma_0=1.0, sigma_0_bounds=(0.1, 10.0)) ** 2
#)
kernel = 1.0 * Matern(length_scale=1.0, length_scale_bounds=(1e-1, 10.0), nu=1.5)
reg = GaussianProcessRegressor(kernel=kernel,n_restarts_optimizer=10,alpha=0.1)
reg.fit(x_train,y_train)
reg.score(x_train,y_train)
y_pred, y_std = reg.predict(x_test, return_std=True)
y_pred1 = scaler.inverse_transform(y_pred)
y_test1 = scaler.inverse_transform(y_test)

Rsq = r2_score(y_test1,y_pred1)
mse = mean_squared_error(y_test1,y_pred1)

# predict BH values
BH_pred, BH_std = reg.predict(dtx_std, return_std=True)
BH_pred1 = scaler.inverse_transform(BH_pred)

# write the prediction results into Excel table
output = pd.DataFrame(np.hstack([np.array(data.OID).reshape(-1,1),np.array(data.XCO).reshape(-1,1),np.array(data.YCO).reshape(-1,1),np.array(BH_pred1).reshape(-1,1)]))
writer = pd.ExcelWriter('prediction_BH.xlsx')
output.to_excel(writer,'page_1',float_format='%.5f')
writer.save()