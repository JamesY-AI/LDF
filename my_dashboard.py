
import numpy as np
import torch
from torch import Tensor
import torch.nn.functional as F
import matplotlib.pyplot as plt
import streamlit as st
from sklearn.preprocessing import StandardScaler
from MLP import MLP
import math

st.title('Lane Distribution Factor')

model = MLP(7, 100, 100, 2)
model.load_state_dict(torch.load("model_4ln"))
model.eval()

#@st.cache()

aadt = st.slider('Directional AADT', value=10000, min_value=1000, max_value=80000)
trk_pct = st.slider('Truck Percent', value=20.0, min_value=0.0, max_value=50.0)
area = st.selectbox('Select area type', ["Rural", "Urban"])
facility = st.selectbox('Select facilty type', ["Interstate", "Principal Arterial-Other Freeways/Expressways", "Principal Arterial-Other"])  

#preprocess the input
f = open('std_params', 'r') 
reader = csv.reader(f, delimiter=',')
params = []
for row in reader:
    params.append(row)
f.close()

aadt_mean = float(params[0][0])
aadt_var = float(params[0][1])
tp_mean = float(params[1][0])
tp_var = float(params[1][1])
aadt_std = (aadt-aadt_mean)/math.sqrt(aadt_var)
tp_std = (trk_pct-tp_mean)/math.sqrt(tp_var)


if area == "Rural":
    area_r = 1.0
    area_u = 0.0
else:
    area_r = 0.0
    area_u = 1.0

if facility == "Interstate":
    IS = 1.0
    PA_O =0.0
    PA_OFE = 0.0
elif facility == "Principal Arterial-Other":
    IS = 0.0
    PA_O =1.0
    PA_OFE = 0.0
else:
    IS = 0.0
    PA_O = 0.0
    PA_OFE = 1.0

X = torch.tensor([[aadt_std[0][0], tp_std[0][0], IS, PA_O, PA_OFE, area_r, area_u]])

y_pred = model(X.float(), "inference")
ldf = y_pred.detach().numpy()

# Plot LDF
data_dict = {'outer lane':ldf[0][0], 'inner_lane':ldf[0][1]}
lanes = list(data_dict.keys())
values = list(data_dict.values())
fig = plt.figure(figsize = (10, 5))
#  Bar plot
plt.bar(lanes, values, color ='blue', width = 0.5)
#plt.xlabel("Lanes")
plt.ylabel("percent of trucks")
plt.title(f"Lane Distribution Factor: outer_lane={ldf[0][0]:.2f},  inner_lane={ldf[0][1]:.2f}")
plt.show()

st.pyplot(fig)
