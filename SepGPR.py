# -*- coding:utf-8 -*-

"""
Spatially-implicit Gaussian Process Regression for Building Height Estimation
The operation of SP regression is demonstrated in this source code, where the parameters can be reinitialized according to the user's dataset.
@Chen Yang, College of Urban and Environmental Sciences, Peking University
2021/06/13 15:16
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
import cmath

rng = np.random.RandomState(0)

#read
data = pd.read_excel('trial_data1.xlsx')  # replace your reference data here
dt = np.array(data) # please note that you should extract your own reference data as your table
dtx = dt[:,2:9]
dty = dt[:,1].reshape(-1,1)
city = np.unique(np.array(data.CID).reshape(-1,1))
city = city.reshape(-1,1)
xco = dt[:,9].reshape(-1,1)
yco = dt[:,10].reshape(-1,1)
oid = dt[:,0].reshape(-1,1)

BH_pred = np.zeros((dtx.shape[0],1), dtype = float)
BH_pred1 = np.hstack([oid, np.array(BH_pred).reshape(-1,1)])

for i in city:
    city_dtx = dtx[dt[:, 11] == i, :]
    city_dty = dty[dt[:, 11] == i, :]
    city_xco = xco[dt[:, 11] == i, :]
    city_yco = yco[dt[:, 11] == i, :]
    city_oid = oid[dt[:, 11] == i, :]
    bw_no = np.arange(5000,7000,1000).reshape(-1,1) # you can modify the bandwidth value for your own dataset
    bw_mse = np.zeros((bw_no.shape[0],2), dtype = float)
    bw_idx = 0
    city_ref_num = np.arange(0, city_dtx.shape[0], 1).reshape(-1, 1)
    """
    cross-validation for nominating optimal bandwidth
    """
    for tmp_bw in bw_no:
        tmp_mse = 0;
        for j in city_ref_num:
            bf_dist = np.zeros((city_dtx.shape[0],2), dtype = float)
            for j1 in city_ref_num:
                bf_dist[j1, 0] = city_oid[j1]
                bf_dist[j1, 1] = ((city_xco[j1] - city_xco[j])**2 + (city_yco[j1] - city_yco[j])**2)**0.5

            bf_dtx = city_dtx[bf_dist[:, 1] <= tmp_bw, :]
            bf_dty = city_dty[bf_dist[:, 1] <= tmp_bw, :]
            scaler = StandardScaler()
            bf_dtx_std = scaler.fit_transform(bf_dtx)
            bf_dty_std = scaler.fit_transform(bf_dty)
            bf_oid = city_oid[bf_dist[:, 1] <= tmp_bw, :]

            kernel = 1.0 * Matern(length_scale=1.0, length_scale_bounds=(1e-1, 10.0), nu=1.5)
            reg = GaussianProcessRegressor(kernel=kernel, n_restarts_optimizer=10, alpha=0.1)
            reg.fit(bf_dtx_std, bf_dty_std)
            reg.score(bf_dtx_std, bf_dty_std)
            bf_pred, bf_std = reg.predict(bf_dtx_std, return_std=True)
            bf_pred1 = scaler.inverse_transform(bf_pred)
            bf_mse = mean_squared_error(bf_dty_std, bf_pred)
            tmp_mse = tmp_mse + bf_mse

            for i1 in range(len(bf_oid)):
                for j1 in range(len(BH_pred1)):
                    if BH_pred1[j1][0] == bf_oid[i1][0]:
                        BH_pred1[j1][1] = bf_pred1[i1][0]
                        break

        bw_mse[bw_idx][0] = tmp_bw
        bw_mse[bw_idx][1] = tmp_mse
        bw_idx = bw_idx + 1

    """
    nominating optimal bandwidth
    """
    bw_mse[np.argsort(bw_mse[:,1])]
    city_bw = bw_mse[0][0]

    """
    predict BH using nominated bandwidth
    note that: 
    The bf_dtx should be replaced in the actual regression with a point that has no building height references 
    and is renamed. Here, the authors chose to keep bf_dtx for the success of the compilation. 
    """
    for j in city_ref_num: # pixels with building height references
        bf_dist = np.zeros((city_dtx.shape[0], 2), dtype=float)
        for j1 in city_ref_num: # pixels without building height references
            bf_dist[j1, 0] = city_oid[j1]
            bf_dist[j1, 1] = ((city_xco[j1] - city_xco[j]) ** 2 + (city_yco[j1] - city_yco[j]) ** 2) ** 0.5

        bf_dtx = city_dtx[bf_dist[:, 1] <= city_bw, :]
        bf_dty = city_dty[bf_dist[:, 1] <= city_bw, :]
        scaler = StandardScaler()
        bf_dtx_std = scaler.fit_transform(bf_dtx)
        bf_dty_std = scaler.fit_transform(bf_dty)
        bf_oid = city_oid[bf_dist[:, 1] <= city_bw, :]

        kernel = 1.0 * Matern(length_scale=1.0, length_scale_bounds=(1e-1, 10.0), nu=1.5)
        reg = GaussianProcessRegressor(kernel=kernel, n_restarts_optimizer=10, alpha=0.1)
        reg.fit(bf_dtx_std, bf_dty_std)
        reg.score(bf_dtx_std, bf_dty_std)
        bf_pred, bf_std = reg.predict(bf_dtx_std, return_std=True)
        bf_pred1 = scaler.inverse_transform(bf_pred)

        for i1 in range(len(bf_oid)):
            for j1 in range(len(BH_pred1)):
                if BH_pred1[j1][0] == bf_oid[i1][0]:
                    if BH_pred1[j1][1] == 0:
                        BH_pred1[j1][1] = bf_pred1[i1][0]
                        break
    # a = 1

# write the prediction results into Excel table
output = pd.DataFrame(np.hstack([BH_pred1[:, 0].reshape(-1, 1), np.array(data.XCO).reshape(-1, 1), np.array(data.YCO).reshape(-1, 1), BH_pred1[:, 1].reshape(-1, 1)]))
writer = pd.ExcelWriter('prediction_BH.xlsx')
output.to_excel(writer,'page_1',float_format='%.5f')
writer.save()