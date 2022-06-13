
import numpy as np
import torch
from torch import Tensor
import torch.nn.functional as F
import matplotlib.pyplot as plt
import streamlit as st
from sklearn.preprocessing import StandardScaler
import pandas as pd
from MLP import MLP
import math

st.title('Lane Distribution Factor')

aadt = st.slider('Directional AADT', value=10000, min_value=2000, max_value=100000)
area = st.selectbox('Select area type', ["Urban", "Rural"])
facility = st.selectbox('Select facility type', ["Interstate, Other Freeways or Expressways", "Others"])
num_lanes = st.selectbox('Select the number of lanes (directional)', ["2", "3+"])

#preprocess the input
df = pd.read_csv('std_params.csv')

b0 = df["const"][0]
b1 = df["AADT"][0]
b2 = df["Urban"][0]
b3 = df["Interstate"][0]
b4 = df["3+ln"][0]

c0 = df["const"][1]
c1 = df["AADT"][1]
c2 = df["Urban"][1]
c3 = df["Interstate"][1]

if area == "Urban":
    Urban = 1.0
else:
    Urban = 0.0

if facility == "Interstate, Other Freeways or Expressways":
    IS = 1.0
else:
    IS = 0.0

if num_lanes =="2":
    LN = 0.0
else:
    LN = 1.0

bx = b0 + b1*math.log(aadt) + b2*Urban + b3*IS + b4*LN
ldf_outer = 1/(1+ math.exp(-bx))

# only for 3+ directional lanes
if LN == 0.0:
    ldf_inner = 1.0 - ldf_outer
else:
    cx = c0 + c1*math.log(aadt) + c2*Urban + c3*IS
    ldf_center = (1/(1.0 + math.exp(-cx)))*(1.0-ldf_outer)
    ldf_inner = 1.0-ldf_center-ldf_outer

# Plot LDF
if LN == 0.0:
    data_dict = {'inner_lane':ldf_inner, 'outer_lane':ldf_outer}
    lanes = list(data_dict.keys())
    values = list(data_dict.values())
else:
    data_dict = {'inner_lane':ldf_inner, 'center_lane': ldf_center, 'outer_lane':ldf_outer}
    lanes = list(data_dict.keys())
    values = list(data_dict.values())

fig = plt.figure(figsize = (10, 5))
#  Bar plot
plt.bar(lanes, values, color ='blue', width = 0.5)

#plt.xlabel("Lanes")
plt.ylabel("percent of trucks")

if LN == 0.0:
    plt.title(f"Lane Distribution Factor: inner lane={data_dict['inner_lane']:.2f}, outer lane={data_dict['outer_lane']:.2f}")
else:
    plt.title(f"Lane Distribution Factor: inner lane={data_dict['inner_lane']:.2f}, center lane={data_dict['center_lane']:.2f}, outer lane={data_dict['outer_lane']:.2f}")
plt.show()

st.pyplot(fig)
