"""
Coupled AR model of temperature and demand consumption.

2023/10/17
Junha Yoo, modified by James Richard Forbes

Data from:
https://climate.weather.gc.ca/historical_data/search_historic_data_e.html
https://www.hydroquebec.com/documents-data/open-data/history-electricity-demand-quebec/

Sample code for students.

2023/12/16
Modified by Mohamed-Anis Mansour
for side project
#We will attempt to couple the OPEN (O) and CLOSE (C) prices of of a stock for each day. 
# This will help us predict the OPEN and CLOSE prices 1 day later

"""
#%%
# Import libraries
import numpy as np
import scipy.linalg as linalg
import matplotlib.pyplot as plt
import yfinance as yf
# %%
# Plotting parameters
# plt.rc('text', usetex=True)
plt.rc("font", family="serif", size=16)
plt.rc("lines", linewidth=2)
plt.rc("axes", grid=True)
plt.rc("grid", linestyle="--")

# %%

# Download historical data using yfinance
symbol = 'AAPL'
start_date = '2020-10-01'
end_date = '2023-11-13'
Data = yf.download(symbol, start=start_date, end=end_date)
print(Data.columns)

# load csv file
data_o = Data['Open']
'''np.loadtxt(
    "en_climate_hourly_QC_7025251_08-2022_P1H.csv",
    delimiter=",",
    skiprows=1,
    usecols=(9,),
    dtype=str,
)
'''
data_c = Data['Close']
'''np.loadtxt(
    "2022-demande-electricite-quebec.csv",
    delimiter=",",
    skiprows=5088,
    usecols=(1,),
    max_rows=744,
    dtype=str,
)
'''

'''
Make sure that the data transformation from pandas to np has no issues

Find the size of the data and split in half
Use the first half to trian the data 
and the second half to test it

I added breakpoints for each needed variable needing modification
'''

label_data = np.array(["Open Prices", "Close Prices"])  # data label
data = np.block([[data_o], [data_c]]).T  # stack the data

data_size = data.shape  # data size
#data = np.zeros(data_size, dtype=float)  # preallocation for the numerical data
unit = ["$", "$"]  # unit of each data
num_var = data_size[1]  # number of variables

'''
for i in range(data_size[1]):
    unit.append(label_data[i].split(" ")[-1])
# Convert strings to float
for i in range(data_size[1]):
    data[:, i] = np.array([float(temp.strip('"')) for temp in data_str[:, i]])
    label_data[i] = label_data[i].strip('"')
'''
# do not change above this line

Y = data.copy()  # data to fit the AR model
N = data.shape[0]  # number of data points
t = np.linspace(0, N, N, endpoint=False)  # time

print(f'N: {N},\nData:\n{data}')


# plot the raw data
fig, ax = plt.subplots(2, 1, figsize=(18.5, 10.5))
ax[0].set_title(r"Daily Open Prices of Stock")
ax[1].set_title(r"Daily Close Prices of Stock")
for i in range(2):
    ax[i].plot(t, data[:, i], label=label_data[i])
    ax[i].set_xlabel(r"t (hours)")
    ax[i].set_ylabel(label_data[i])
    fig.tight_layout()
    # ax.legend(loc="center left", bbox_to_anchor=(1, 0.5))
    # fig.savefig("figs/MVAR_data.pdf")


# %%
# AR model functions


def form_A_B(Y, ell):
    # Write a function to compute A and b.
    # These A and B matrices are ``dummy" matrices. Delete them.
    # N = time steps, num_var = number of variables
    N = Y.shape[0]
    num_var = Y.shape[1]

    # ------------------Modify below------------------
    # A contains data from time 0 to time (N - ell) (day 16 - ell)
    A = np.zeros((N - ell, ell * num_var))    # A is a (N - ell) by (ell * num_var)matrix
    for i in range(ell*num_var): # iterate through columns of A
        A[:,[i]]=Y[(i-i//ell*ell):N-ell+(i-i//ell*ell),[i//ell]]
    
    B =Y[ell:N,:]
    '''
    TD = data[:N, 0] #all temp data
    ED = data[:N, 1] #all electricity data
    print("N = ", N) #383 (to represent the first 16 days)
    print("num_var = ", num_var)
    print(f'data shape: {data.shape[0]},{data.shape[1]}')
    print(f'TD shape: {TD.shape[0]}, ED shape: {ED.shape[0]}')
    A = np.random.random((N-ell, ell * num_var))  # a matrix full of random numbers.
    B = np.ones((N-ell, num_var))  # a matrix full of ones.
    #A = np.random.random((N - ell, ell * num_var))  # a matrix full of random numbers.
    #B = np.ones((N - ell, num_var))  # a matrix full of ones.
    
    print(f'A shape: {A.shape[0]},{A.shape[1]}\nB shape: {B.shape[0]},{B.shape[1]}')
    
    A[:, 0] = TD[1:-1]
    A[:, 1] = TD[:-2]
    A[:, 2] = ED[1:-1]
    A[:, 3] = ED[:-2]
    
    B[:, 0] = TD[2:]
    B[:, 1] = ED[2:]
    '''
    # ------------------Modify above------------------

    return A, B


def fit(A, B):
    # Write a function that solves AX=B where X are the AR model parameters

    # ------------------Modify below------------------
    X = linalg.solve(A.T @ A,A.T @ B) 
    
    '''
    X = np.ones(
        (A.shape[1], B.shape[1]), dtype=float
    )  # a dummy variable, delete and modify
    
    X = np.linalg.solve((A.T)@A,(A.T)@B)'''
    
    #R = B - (A@X)
    #Rnorm = np.norm(R)
    #print(f'Rnorm = {Rnorm}')
    # ------------------Modify above------------------

    return X


def predict(y, x, train=False):
    # Write a function that predicts the outputs given the AR model
    # parameters and data at each time step.
    N = y.shape[0]
    num_var = y.shape[1]
    ell = x.shape[0] // num_var
    # Write your prediction code here. Your prediction should go from ell to
    # N. The first ell - 1 data points will be set to an average, as done
    # below.

    # ------------------Modify below------------------
    y_pred = np.zeros((N,num_var))
    a_kt = np.zeros((1,ell*num_var))
    for k in range(ell,N):
        for i in range(ell*num_var):
            a_kt[:,i] = y[k-ell+(i-i//ell*ell),i//ell]
            y_pred[[k],:] = a_kt @ x
    
    '''
    print("_____________________________________")
    print("N = ", N) #383 (to represent the first 16 days)
    print("num_var = ", num_var)
    print("ell = ", ell)
    print(f'data shape: {data.shape[0]},{data.shape[1]}')
    
    C = np.zeros((N-ell, 4))
    C[:, 0] = y[ell-1:1-ell, 0]
    C[:, 1] = y[:N-ell, 0]
    C[:, 2] = y[ell-1:1-ell, 1]
    C[:, 3] = y[:N-ell, 1]
    
    y_pred = np.zeros((N, ell))
    print(f'y_pred shape: {y_pred.shape[0]},{y_pred.shape[1]}')

    for i in range(N-ell):
        y_pred[ell + i, :] = C[i, :] @ x
    
    return y_pred''' 
    # ------------------Modify above------------------
    y_pred_mean = np.zeros((num_var),dtype=float)
    for i in range(num_var):
        y_pred_mean[i] = np.mean(y[:ell,i])
        y_pred[:ell,i] = np.ones(ell) * y_pred_mean[i]
    return y_pred



# %%
# Fit (train) the AR model
N_start = 0  # start day, don't change
N_end = int(N/2) # end day, don't change 383
print(f'Train\nStart: {N_start}, End: {N_end}')
ell = 2  # AR model memory, don't change
t = np.arange(N_start, N_end)  # time, don't change
y_scale = np.zeros(num_var, dtype=float)  # scale factor preallocation
y_scale = np.max(np.linalg.norm(Y[N_start:N_end], 2))  # scale factor
Y = Y / y_scale  # non-dimensionalize the data, don't change
A, B = form_A_B(Y[N_start:N_end], ell)  # form A, B matrices
print(np.linalg.matrix_rank(A))  # shoudl be full rank
X = fit(A, B)  # find the AR parameters
Y = Y * y_scale  # dimensionalize the data again
y_pred = predict(Y[N_start:N_end], X, train=True)  # predictions
e = np.abs(Y[N_start:N_end] - y_pred)  # absolute error

# Plotting
fig, ax = plt.subplots(2, 1)
fig.set_size_inches(18.5, 10.5, forward=True)
ax[0].set_title(r"AR Model Train")
# Format axes
for a in np.ravel(ax):
    a.set_xlabel(r"$t$ (Days)")
ax[0].set_ylabel(r"$O_k$ %s" % unit[0])
ax[1].set_ylabel(r"$e_k %s" % unit[0])
ax[0].plot(t[ell:], y_pred[ell:, 0], label=r"$O_{k, pred, \ell=%s}$" % ell)
ax[0].plot(t[ell:], Y[N_start + ell : N_end, 0], "--", label=r"$O_{k, true}$")
ax[1].plot(t[ell:], e[ell:, 0], label=r"$e_{k, \ell=%s}$" % ell)
ax[0].legend(loc="center left", bbox_to_anchor=(1, 0.5))
ax[1].legend(loc="center left", bbox_to_anchor=(1, 0.5))
fig.tight_layout()
# fig.savefig("figs/MVAR_response_train_temp.pdf")
plt.show()

fig, ax = plt.subplots(2, 1)
fig.set_size_inches(18.5, 10.5, forward=True)
ax[0].set_title(r"AR Model Train")
for a in np.ravel(ax):
    a.set_xlabel(r"$t$ (hours)")
ax[0].set_ylabel(r"$C_k$ %s" % unit[1])
ax[1].set_ylabel(r"$e_k$ %s" % unit[1])
ax[0].plot(t[ell:], y_pred[ell:, 1], label=r"$C_{k, pred, \ell=%s}$" % ell)
ax[0].plot(t[ell:], Y[N_start + ell : N_end, 1], "--", label=r"$C_{k, true}$")
ax[1].plot(t[ell:], e[ell:, 1], label=r"$e_{k, \ell=%s}$" % ell)
ax[0].legend(loc="center left", bbox_to_anchor=(1, 0.5))
ax[1].legend(loc="center left", bbox_to_anchor=(1, 0.5))
fig.tight_layout()
# fig.savefig("figs/MVAR_response_train_demand.pdf")
plt.show()


# %%
# Test
Y = data.copy()
N_start = int(N/2) + 1 # start day, don't change 384
N_end = N  # end day, don't change 743
print(f'Test\nStart: {N_start}, End: {N_end}')
t = np.arange(N_start, N_end)  # time, don't change

y_pred = predict(Y[N_start:N_end], X)  # predictions

# Compute various metrics associated with predictinon error

# ------------------Modify below------------------
e = np.abs(Y[N_start:N_end] - y_pred)
e_rel = e / np.abs(Y[N_start:N_end])*100
mu = np.mean(e,axis=0)
sigma = np.std(e,axis=0)
mu_e_rel = np.mean(e_rel,axis=0)
sigma_e_rel = np.std(e_rel,axis=0)

'''
e = np.zeros(y_pred.shape)  # delete zeros
for i in range((-N_start+N_end)):
    e[i, :] = abs(Y[N_start+i, :] - y_pred[i, :])

e_rel = np.zeros(y_pred.shape)  # delete zeros
for i in range((-N_start+N_end)):
    e_rel[i, :] = (abs(Y[N_start+i, :] - y_pred[i, :])/abs(Y[N_start+i, :]))*100

mu = np.zeros(2)  # delete zeros
mu[0] = np.mean(e[:,0])
mu[1] = np.mean(e[:,1])
sigma = np.zeros(2)  # delete zeros
sigma[0] = np.std(e[:,0])
sigma[1] = np.std(e[:,1])
mu_e_rel = np.zeros(2)  # delete zeros
mu_e_rel[0] = np.mean(e_rel[:,0])
mu_e_rel[1] = np.mean(e_rel[:,1])
sigma_e_rel = np.zeros(2)  # delete zeros
sigma_e_rel[0] = np.std(e_rel[:,0])
sigma_e_rel[1] = np.std(e_rel[:,1])
'''
# ------------------Modify above----------------

# %%
# Plotting

fig, ax = plt.subplots(3, 1)
fig.set_size_inches(18.5, 10.5, forward=True)
ax[0].set_title(r"AR Model Test")
for a in np.ravel(ax):
    a.set_xlabel(r"$t$ (hours)")
ax[0].set_ylabel(r"$O_k$ %s" % unit[0])
ax[1].set_ylabel(r"$e_{O,k}$ %s" % unit[0])
ax[2].set_ylabel(r"$e_{O,k, rel}$ (%)")
ax[0].plot(t[ell:], y_pred[ell:, 0], label=r"$O_{k, pred, \ell=%s}$" % ell)
ax[1].plot(t[ell:], e[ell:, 0], label=r"$e_{O,k, \ell=%s}$" % ell)
ax[0].plot(t[ell:], Y[N_start + ell : N_end, 0], "--", label=r"$T_{k, true}$")
ax[2].plot(t[ell:], e_rel[ell:, 0], label=r"$e_{O,k, rel, \ell=%s}$" % ell)
ax[0].legend(loc="center left", bbox_to_anchor=(1, 0.5))
ax[1].legend(loc="center left", bbox_to_anchor=(1, 0.5))
ax[2].legend(loc="center left", bbox_to_anchor=(1, 0.5))
fig.tight_layout()
# fig.savefig("figs/MVAR_test_temp_ell_%s.pdf" % ell)
plt.show()

fig, ax = plt.subplots(3, 1)
fig.set_size_inches(18.5, 10.5, forward=True)
ax[0].set_title(r"AR Model Test")
for a in np.ravel(ax):
    a.set_xlabel(r"$t$ (hours)")
ax[0].set_ylabel(r"$C_k$ %s" % unit[1])
ax[1].set_ylabel(r"$e_{C,k}$ %s" % unit[1])
ax[2].set_ylabel(r"$e_{C,k, rel}$ (%)")
ax[0].plot(t[ell:], y_pred[ell:, 1], label=r"$C_{k, pred, \ell=%s}$" % ell)
ax[1].plot(t[ell:], e[ell:, 1], label=r"$e_{C,k, \ell=%s}$" % ell)
ax[0].plot(t[ell:], Y[N_start + ell : N_end, 1], "--", label=r"$C_{k, true}$")
ax[2].plot(t[ell:], e_rel[ell:, 1], label=r"$e_{C,k, rel, \ell=%s}$" % ell)
ax[0].legend(loc="center left", bbox_to_anchor=(1, 0.5))
ax[1].legend(loc="center left", bbox_to_anchor=(1, 0.5))
ax[2].legend(loc="center left", bbox_to_anchor=(1, 0.5))
fig.tight_layout()
# fig.savefig("figs/MVAR_test_demand_ell_%s.pdf" % ell)
plt.show()

print("Mean absolute error ($, $) is ", mu, "\n")
print("Absolute error standard deviation ($, $) is", sigma, "\n")
print("Mean relative error (percent) is ", mu_e_rel, "\n")
print("Relative error standard deviation (percent) is", sigma_e_rel, "\n")

# %%
