# LSTM 기반의 온실내 온도 예측 - neoWizard

import tensorflow as tf
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Dropout

# 데이터 불러오기

raw_df = pd.read_csv('./climate_data.csv')  # 온실 환경 데이터
raw_df.head()

#plt.title('Air Temperature')
#plt.ylabel('degree')
#plt.xlabel('period')
#plt.grid()

#plt.plot(raw_df['inside_temp'], label='inside_temp')

#plt.show()


# 데이터 전처리 (Missing Data 처리, 정규화 등)


# 통계정보 확인

#raw_df.describe()

# 정규화 (Date 제외한 모든 수치부분 정규화)

from sklearn.preprocessing import MinMaxScaler
scaler = MinMaxScaler()
scale_cols = ['inside_temp', 'Outside_temp', 'outside_RH', 'Radiation_intensity', 'Rail_Heating',
              'AHU_Cooling', 'Fan_speed', 'inlet_vent_position']
scaled_df = scaler.fit_transform(raw_df[scale_cols])
scaled_df = pd.DataFrame(scaled_df, columns=scale_cols)

#print(scaled_df)


#  - 시계열 데이터를 위한 window_size = 30 선정

# 입력 파라미터 feature, label => numpy type
def make_sequene_dataset(feature, label, window_size):

    feature_list = []      # 생성될 feature list
    label_list = []        # 생성될 label list

    for i in range(len(feature)-window_size):
        feature_list.append(feature[i:i+window_size])
        label_list.append(label[i+window_size])

    return np.array(feature_list), np.array(label_list)

# feature_df, label_df 생성
feature_cols = ['Outside_temp', 'outside_RH', 'Radiation_intensity', 'Rail_Heating',
              'AHU_Cooling', 'Fan_speed', 'inlet_vent_position']
label_cols = [ 'inside_temp' ]

feature_df = pd.DataFrame(scaled_df, columns=feature_cols)
label_df = pd.DataFrame(scaled_df, columns=label_cols)

# DataFrame => Numpy 변환
feature_np = feature_df.to_numpy()
label_np = label_df.to_numpy()

#print(feature_np.shape, label_np.shape)

# 시계열 데이터 생성 (make_sequence_dataset)

window_size = 40
X, Y = make_sequene_dataset(feature_np, label_np, window_size)

#print(X.shape, Y.shape)

# 학습데이터, 테스트데이터 생성

# train, test 분리

#split = int(len(X)*0.80)
split = -432

x_train = X[0:split]
y_train = Y[0:split]

x_test = X[split:]
y_test = Y[split:]

#print(x_train.shape, y_train.shape)
#print(x_test.shape, y_test.shape)

# 모델 구축 및 컴파일

# model 생성

model = Sequential()
model.add(LSTM(128, activation='tanh', input_shape=x_train[0].shape))
model.add(Dense(1, activation='linear'))
model.compile(loss='mse', optimizer='adam', metrics=['mae'])

model.summary()


# 모델 학습 (EarlyStopping 적용)
from tensorflow.keras.callbacks import EarlyStopping

early_stop = EarlyStopping(monitor='val_loss', patience=5)
model.fit(x_train, y_train, 
          validation_data=(x_test, y_test),
          epochs=100, batch_size=16,
          callbacks=[early_stop])

# 예측을 통한 정답과의 비교 
# (오차계산 MAPE 사용, 평균절대값백분율오차)

pred = model.predict(x_test)

plt.figure(figsize=(12, 6))
plt.title('Temperature Prediction')
plt.ylabel('Temperature')
plt.xlabel('period')
plt.plot(y_test, label='measured')
plt.plot(pred, label='predicted')
plt.grid()
plt.legend(loc='best')

plt.show()

# 평균절대값백분율오차계산 (MAPE)

print( np.sum(abs(y_test-pred)/y_test) / len(x_test) )


