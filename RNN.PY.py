import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import GRU, Dense

DisplayPlots = True
theta = 0.95
T = 100
nlags = 10
T_oos = T - nlags - 20

# autoregressive process

np.random.seed(42)
epsilon = np.random.normal(0,1,T)
x_ts = np.zeros((T,1))
for t in range(1,T):
    x_ts[t] = theta * x_ts[t-1] + epsilon[t]

X = np.zeros((T-nlags,nlags))
y = np.zeros(T-nlags)
for s in range(nlags,T):
    X[s-nlags,:] = x_ts[s-nlags:s].reshape(1,nlags)
    y[s-nlags] = x_ts[s]

model = Sequential()
model.add(GRU(1, activation='tanh', input_shape=(nlags, 1)))
model.add(Dense(1))
model.compile(optimizer='adam', loss='mse')
model.fit(X[:T_oos], y[:T_oos], epochs=1000, batch_size=32, validation_split=0.2)

forecast_y = np.zeros(T-nlags)
for t in range(T-nlags):
    tmp = model.predict(X[t,:].reshape(1,nlags,1))
    forecast_y[t] = tmp[0][0]

if DisplayPlots:
    Max_y = np.max(y)

    plt.figure()
    plt.plot(y)
    plt.plot(forecast_y)
    plt.legend(['y','forecast'],loc='lower left')
    plt.axvline(x=T_oos, color='red', linestyle='--')
    plt.text(T_oos + 1, Max_y, "Out of sample", fontsize=12, verticalalignment='center', bbox=dict(facecolor='red', alpha=0.5))
    plt.title('y vs forecast')

# Hawkes process
T = 300
nlags = 10
T_oos = T - nlags - int(T/5)
# parameters ------------------
lambda0 = 1.2
alpha = 0.6
beta = 0.8
mu = lambda0
t = 0
ts = [0]
lambdas = [lambda0]
eps = 1e-6
while t < T:
    M = mu + (lambdas[-1] - mu) * np.exp(-beta * (t - ts[-1] + eps))
    u = np.random.uniform(0,1)
    t = t - np.log(u) / M
    v = np.random.uniform(0,1)
    if ( v <= (mu + (lambdas[-1] - mu) * np.exp(-beta * (t - ts[-1]))) / M ):
        ts.append(t)
        lambdas.append((lambdas[-1] - mu) * np.exp(-beta * (ts[-1] - ts[-2])) + mu + alpha)
ts = ts[1:-1]
lambdas = lambdas[1:-1]
n_events = len(ts)
print("n events=",n_events)
#print("ts=",ts)
dt = 0.01
nt = int(T/dt)
lambda_t = lambda0 * np.ones(nt)
events_t = np.zeros(nt)
time = (T/nt) * np.array(range(nt))
ind = 0
indices = []
for k in range(n_events):
    new_ind = int((ts[k] / T) * (nt - 1))
    if new_ind > ind:
        ind = new_ind
        indices.append(ind)
        events_t[ind] = 1
        lambda_t[ind:] += alpha * np.exp(-beta * (time[ind:] - ts[k]))

x_ts = lambda_t

X = np.zeros((T-nlags,nlags))
y = np.zeros(T-nlags)
for s in range(nlags,T):
    X[s-nlags,:] = x_ts[s-nlags:s].reshape(1,nlags)
    y[s-nlags] = x_ts[s]

model = Sequential()
model.add(GRU(1, activation='tanh', input_shape=(nlags, 1)))
model.add(Dense(1))
model.compile(optimizer='adam', loss='mse')
model.fit(X[:T_oos], y[:T_oos], epochs=1000, batch_size=32, validation_split=0.2)

forecast_y = np.zeros(T-nlags)
for t in range(T-nlags):
    tmp = model.predict(X[t,:].reshape(1,nlags,1))
    forecast_y[t] = tmp[0][0]

if DisplayPlots:
    Max_y = np.max(y)

    plt.figure()





    
    plt.plot(y)
    plt.plot(forecast_y)
    plt.legend(['y','forecast'],loc='lower left')
    plt.axvline(x=T_oos, color='red', linestyle='--')
    plt.text(T_oos + 1, Max_y, "Out of sample", fontsize=12, verticalalignment='center', bbox=dict(facecolor='red', alpha=0.5))
    plt.title('y vs forecast')

    plt.show()