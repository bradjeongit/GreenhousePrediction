# %% [markdown]
# LSTM 기반의 삼성전자 주가 예측 예제
#  - yahoo finance 에서 데이터 다운로드 후 3일(3MA), 5일(5MA) 가격이평선 추가

# %%
import tensorflow as tf
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Dropout

# %% [markdown]
# 데이터 불러오기

# %%
raw_df = pd.read_csv('./005930.KS_3MA_5MA.csv')  # yahoo finance 로부터 데이터 다운로드

raw_df.head()

# %%
plt.title('SAMSUNG ELECTRONIC STCOK PRICE')
plt.ylabel('price')
plt.xlabel('period')
plt.grid()

plt.plot(raw_df['Adj Close'], label='Adj Close')

plt.show()

# %% [markdown]
# 데이터 전처리 (Missing Data 처리, 정규화 등)

# %%
# 통계정보 확인

raw_df.describe()

# %%
# Missing Data 확인

raw_df.isnull().sum()

# %%
# 최소값이 0 인 column 체크

for col in raw_df.columns:

    if raw_df[col].min() == 0:
        col_name = col
        print(col_name, type(col_name))

# %%
raw_df.loc[raw_df['Volume']==0]

# %%
# 각 column에 0 몇개인지 확인

for col in raw_df.columns:

    missing_rows = raw_df.loc[raw_df[col]==0].shape[0]
    print(col + ': ' + str(missing_rows))

# %%
# 먼저 0 을 NaN 으로 바꾼후, Missing Data 처리

raw_df['Volume'] = raw_df['Volume'].replace(0, np.nan)

# 각 column에 0 몇개인지 확인

for col in raw_df.columns:

    missing_rows = raw_df.loc[raw_df[col]==0].shape[0]
    print(col + ': ' + str(missing_rows))

# %%
# missing data 확인

raw_df.isnull().sum()

# %%
raw_df.isnull().any()

# %%
raw_df.loc[raw_df['Open'].isna()]

# %%
# missing data 처리

raw_df = raw_df.dropna()

raw_df.isnull().sum()

# %%
# 정규화 (Date 제외한 모든 수치부분 정규화)

from sklearn.preprocessing import MinMaxScaler

scaler = MinMaxScaler()

scale_cols = ['Open', 'High', 'Low', 'Close', 'Adj Close',
              '3MA', '5MA', 'Volume']

scaled_df = scaler.fit_transform(raw_df[scale_cols])

scaled_df = pd.DataFrame(scaled_df, columns=scale_cols)

print(scaled_df)

# %% [markdown]
# 주가예측을 위해 3MA, 5MA, Adj Close 항목을 feature 선정
#  - 정답은 Adj Close 선정
#  - 시계열 데이터를 위한 window_size = 40 선정

# %%
# 입력 파라미터 feature, label => numpy type

def make_sequene_dataset(feature, label, window_size):

    feature_list = []      # 생성될 feature list
    label_list = []        # 생성될 label list

    for i in range(len(feature)-window_size):

        feature_list.append(feature[i:i+window_size])
        label_list.append(label[i+window_size])

    return np.array(feature_list), np.array(label_list)

# %%
# feature_df, label_df 생성

feature_cols = [ '3MA', '5MA', 'Adj Close' ]
label_cols = [ 'Adj Close' ]

feature_df = pd.DataFrame(scaled_df, columns=feature_cols)
label_df = pd.DataFrame(scaled_df, columns=label_cols)

# %%
# DataFrame => Numpy 변환

feature_np = feature_df.to_numpy()
label_np = label_df.to_numpy()

print(feature_np.shape, label_np.shape)

# %% [markdown]
# 시계열 데이터 생성 (make_sequence_dataset)

# %%
window_size = 40

X, Y = make_sequene_dataset(feature_np, label_np, window_size)

print(X.shape, Y.shape)

# %% [markdown]
# 학습데이터, 테스트데이터 생성

# %%
# train, test 분리

#split = int(len(X)*0.95)
split = -200

x_train = X[0:split]
y_train = Y[0:split]

x_test = X[split:]
y_test = Y[split:]

print(x_train.shape, y_train.shape)
print(x_test.shape, y_test.shape)

# %% [markdown]
# 모델 구축 및 컴파일

# %%
# model 생성

model = Sequential()

model.add(LSTM(128, activation='tanh', input_shape=x_train[0].shape))

model.add(Dense(1, activation='linear'))

# %%
model.compile(loss='mse', optimizer='adam', metrics=['mae'])

model.summary()

# %% [markdown]
# 모델 학습 (EarlyStopping 적용)

# %%
from tensorflow.keras.callbacks import EarlyStopping

early_stop = EarlyStopping(monitor='val_loss', patience=5)

model.fit(x_train, y_train, 
          validation_data=(x_test, y_test),
          epochs=100, batch_size=16,
          callbacks=[early_stop])

# %% [markdown]
# 예측을 통한 정답과의 비교 
# (오차계산 MAPE 사용, 평균절대값백분율오차)

# %%
pred = model.predict(x_test)

plt.figure(figsize=(12, 6))
plt.title('3MA + 5MA + Adj Close, window_size=40')
plt.ylabel('adj close')
plt.xlabel('period')
plt.plot(y_test, label='actual')
plt.plot(pred, label='prediction')
plt.grid()
plt.legend(loc='best')

plt.show()

# %%
# 평균절대값백분율오차계산 (MAPE)

print( np.sum(abs(y_test-pred)/y_test) / len(x_test) )


