#!/usr/bin/env python3
# -*- coding: UTF-8 -*-
"""
@ Project : ConvTransformerTS
@ FileName: plot_rmse_mae.py
@ IDE     : PyCharm
@ Author  : Jimeng Shi
@ Time    : 1/21/22 22:45
"""
import matplotlib.pyplot as plt
import numpy as np

# Heavy_RMSE = [0.4466, 0.3071, 0.4618, 0.5118, 0.4895]
# Heavy_MAE = [0.3531, 0.2567, 0.3813, 0.3686, 0.3812]
# Medium_RMSE = [0.3429, 0.2953, 0.3317, 0.4203, 0.4702]
# Medium_MAE = [0.227, 0.213, 0.2225, 0.2863, 0.3992]
# Small_RMSE = [0.2006, 0.2302, 0.2109, 0.2153, 0.5396]
# Small_MAE = [0.148, 0.1592, 0.1355, 0.1535, 0.4573]

# (Heavy Storm) Test RMSE for WS_S1, WS_S4, TWS_S25A, TWS_S25B, TWS_S26: [0.4466, 0.3071, 0.4618, 0.5118, 0.4895]
# (Heavy Storm) Test MAE for WS_S1, WS_S4, TWS_S25A, TWS_S25B, TWS_S26: [0.3531, 0.2567, 0.3813, 0.3686, 0.3812]
# (Medium Storm) Test RMSE for WS_S1, WS_S4, TWS_S25A, TWS_S25B, TWS_S26: [0.3429, 0.2953, 0.3317, 0.4203, 0.4702]
# (Medium Storm) Test MAE for WS_S1, WS_S4, TWS_S25A, TWS_S25B, TWS_S26: [0.227, 0.213, 0.2225, 0.2863, 0.3992]
# (Small Storm) Test RMSE for WS_S1, WS_S4, TWS_S25A, TWS_S25B, TWS_S26: [0.2006, 0.2302, 0.2109, 0.2153, 0.5396]
# (Small Storm) Test MAE for WS_S1, WS_S4, TWS_S25A, TWS_S25B, TWS_S26: [0.148, 0.1592, 0.1355, 0.1535, 0.4573]
#
# plt.rcParams["figure.figsize"] = (8, 6)
# plt.title('RMSE Comparison for Water Stage Prediction', fontsize=18)
# plt.plot(Heavy_RMSE, label='Heavy Storm', marker='D')
# plt.plot(Medium_RMSE, label='Medium Storm', marker='*')
# plt.plot(Small_RMSE, label='Small Storm', marker='o')
# plt.xlabel('Locations', fontsize=18)
# plt.ylabel('RMSE', fontsize=18)
#
# plt.xticks(np.arange(5), ['S1', 'S4', 'S25A', 'S25B', 'S26'], fontsize=14)
# plt.yticks(fontsize=14)
# # plt.axhline(y=0.15, color='red', linestyle='-', linewidth=2)
# # plt.axhline(y=0.25, color='orange', linestyle='-', linewidth=2)
# # plt.text(0, 0.26, 'R=0.25', fontsize=14)
# # plt.text(0, 0.16, 'R=0.15', fontsize=14)
# plt.legend(fontsize=14)
# plt.show()
# plt.close()
#
#
# plt.title('MAE Comparison for Water Stage Prediction', fontsize=18)
# plt.plot(Heavy_MAE, label='Heavy Storm', marker='D')
# plt.plot(Medium_MAE, label='Medium Storm', marker='*')
# plt.plot(Small_MAE, label='Small Storm', marker='o')
# plt.xlabel('Locations', fontsize=18)
# plt.ylabel('MAE', fontsize=18)
#
# plt.xticks(np.arange(5), ['S1', 'S4', 'S25A', 'S25B', 'S26'], fontsize=14)
# plt.yticks(fontsize=14)
# # plt.axhline(y=0.15, color='red', linestyle='-', linewidth=2)
# # plt.axhline(y=0.25, color='orange', linestyle='-', linewidth=2)
# # plt.text(0, 0.26, 'R=0.25', fontsize=14)
# # plt.text(0, 0.16, 'R=0.15', fontsize=14)
# plt.legend(fontsize=14)
# plt.show()
# plt.close()



# Test_RMSE = [0.2435, 0.2186, 0.2717, 0.3294, 0.522]
# Test_MAE = [0.174, 0.1533, 0.2264, 0.2159, 0.4239]
#
# # plt.rcParams["figure.figsize"] = (8, 6)
# plt.title('Comparison RMSE & MAE at Different Locations', fontsize=18)
# plt.plot(Test_RMSE, label='RMSE', marker='D')
# plt.plot(Test_MAE, label='MAE', marker='o')
# plt.xlabel('Locations', fontsize=18)
# plt.ylabel('Error', fontsize=18)
#
# plt.xticks(np.arange(5), ['S1', 'S4', 'S25A', 'S25B', 'S26'], fontsize=14)
# plt.yticks(fontsize=14)
# # plt.axhline(y=0.15, color='red', linestyle='-', linewidth=2)
# # plt.axhline(y=0.25, color='orange', linestyle='-', linewidth=2)
# # plt.text(0, 0.26, 'R=0.25', fontsize=14)
# # plt.text(0, 0.16, 'R=0.15', fontsize=14)
# plt.legend(fontsize=14)
# plt.show()


## 10 min
MAE_Flow = [0.170, 0.129, 0.184, 0.130, 0.428]
MAE_Pump = [0.133, 0.158, 0.143, 0.138, 0.367]
MAE_FlowPump = [0.182, 0.155, 0.185, 0.170, 0.474]

plt.rcParams["figure.figsize"] = (8, 6)
plt.title('Comparison of MAE at Different Locations (10 mins)', fontsize=18)
plt.plot(MAE_Flow, label='Flow', marker='D')
plt.plot(MAE_Pump, label='Pump', marker='o')
plt.plot(MAE_FlowPump, label='Flow+Pump', marker='*')
plt.xlabel('Locations', fontsize=18)
plt.ylabel('Error of MAE', fontsize=18)
plt.xticks(np.arange(5), ['S1', 'S4', 'S25A', 'S25B', 'S26'], fontsize=14)
plt.yticks(fontsize=14)
# plt.axhline(y=0.15, color='red', linestyle='-', linewidth=2)
# plt.axhline(y=0.25, color='orange', linestyle='-', linewidth=2)
# plt.text(0, 0.26, 'R=0.25', fontsize=14)
# plt.text(0, 0.16, 'R=0.15', fontsize=14)
plt.legend(fontsize=14)
plt.show()
plt.close()


## 1h
MAE_Flow = [0.27, 0.25, 0.20, 0.24, 0.39]
MAE_Pump = [0.22, 0.22, 0.21, 0.26, 0.42]
MAE_FlowPump = [0.26, 0.22, 0.28, 0.20, 0.39]

plt.rcParams["figure.figsize"] = (8, 6)
plt.title('Comparison of MAE at Different Locations (60 mins)', fontsize=18)
plt.plot(MAE_Flow, label='Flow', marker='D')
plt.plot(MAE_Pump, label='Pump', marker='o')
plt.plot(MAE_FlowPump, label='Flow+Pump', marker='*')
plt.xlabel('Locations', fontsize=18)
plt.ylabel('Error of MAE', fontsize=18)
plt.xticks(np.arange(5), ['S1', 'S4', 'S25A', 'S25B', 'S26'], fontsize=14)
plt.yticks(fontsize=14)
# plt.axhline(y=0.15, color='red', linestyle='-', linewidth=2)
# plt.axhline(y=0.25, color='orange', linestyle='-', linewidth=2)
# plt.text(0, 0.26, 'R=0.25', fontsize=14)
# plt.text(0, 0.16, 'R=0.15', fontsize=14)
plt.legend(fontsize=14)
plt.show()
plt.close()