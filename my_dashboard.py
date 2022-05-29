
import numpy as np
import torch
from torch import Tensor
import torch.nn.functional as F
import matplotlib.pyplot as plt
import pickle
import streamlit as st
from sklearn.preprocessing import StandardScaler
from MLP import MLP

st.title('Lane Distribution Factor')

pickle_in = open("preprocess_4ln.pkl", "rb") 
preprocess = pickle.load(pickle_in)
pickle_in.close()


# class MLP(torch.nn.Module):
#     def __init__(self, in_size, h1_size, h2_size, out_size):
#         super(MLP, self).__init__()
#         self.input_size = in_size
#         self.hidden1_size = h1_size
#         self.hidden2_size = h2_size
#         self.output_size = out_size
#         self.fc1 = torch.nn.Linear(self.input_size, self.hidden1_size)
#         self.bn1 = torch.nn.BatchNorm1d(self.hidden1_size) #batchnorm layer, test doing before ReLU
#         self.relu = torch.nn.ReLU()
#         self.drop1 = torch.nn.Dropout(0.5) #drop out layer, to try improving accuracy
#         self.fc2 = torch.nn.Linear(self.hidden1_size, self.hidden2_size)
#         self.bn2 = torch.nn.BatchNorm1d(self.hidden2_size) #batchnorm layer, test doing before ReLU
#         self.drop2 = torch.nn.Dropout(0.5) #drop out layer, to try improving accuracy
#         self.fc3 = torch.nn.Linear(self.hidden2_size, self.output_size)
        
#     def forward(self, X, mode):
#         h1 = self.fc1(X)
#         bn1 = self.bn1(h1)
#         #drop1 = self.drop1(bn1)
#         relu1 = self.relu(bn1)
#         h2 = self.fc2(relu1)
#         bn2 = self.bn2(h2)
#         #drop2 = self.drop2(bn2) 
#         relu2 = self.relu(bn2)
        
#         if mode == "inference":
#             out = self.fc3(relu2)
#             out = F.softmax(out, dim=1)
#         else:
#             out = self.fc3(relu2)
        
#         return out
    

model = MLP(7, 100, 100, 2)
model.load_state_dict(torch.load("model_4ln"))
model.eval()

#@st.cache()

# mu_in = st.slider('Mean', value=5, min_value=-10, max_value=10)
# std_in = st.slider('Standard deviation', value=5.0, min_value=0.0, max_value=10.0)
# size = st.slider('Number of samples', value=100, max_value=500)

aadt = st.slider('Directional AADT', value=10000, min_value=1000, max_value=80000)
trk_pct = st.slider('Truck Percent', value=20.0, min_value=0.0, max_value=50.0)
area = st.selectbox('Select area type', ["Rural", "Urban"])
facility = st.selectbox('Select facilty type', ["Interstate", "Principal Arterial-Other Freeways/Expressways", "Principal Arterial-Other"])  

#preprocess the input
aadt_std = preprocess[0].transform([[aadt]])
tp_std = preprocess[1].transform([[trk_pct]])

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