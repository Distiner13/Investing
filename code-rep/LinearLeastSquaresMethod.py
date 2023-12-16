"""
Coupled AR model of temperature and demand consumption.

2023/10/17
Junha Yoo, modified by James Richard Forbes

Data from:
https://climate.weather.gc.ca/historical_data/search_historic_data_e.html
https://www.hydroquebec.com/documents-data/open-data/history-electricity-demand-quebec/

Sample code for students.
"""
#%%
# Import libraries
import numpy as np
import scipy.linalg as linalg
import matplotlib.pyplot as plt

# %%
# Plotting parameters
# plt.rc('text', usetex=True)
plt.rc("font", family="serif", size=16)
plt.rc("lines", linewidth=2)
plt.rc("axes", grid=True)
plt.rc("grid", linestyle="--")


# %%
# load csv file

data_t = np.loadtxt(
    "en_climate_hourly_QC_7025251_08-2022_P1H.csv",
    delimiter=",",
    skiprows=1,
    usecols=(9,),
    dtype=str,
)
data_p = np.loadtxt(
    "2022-demande-electricite-quebec.csv",
    delimiter=",",
    skiprows=5088,
    usecols=(1,),
    max_rows=744,
    dtype=str,
)
label_data = np.array(["Temp (°C)", "Demand (MW)"])  # data label
data_str = np.block([[data_t], [data_p]]).T  # stack the data

data_size = data_str.shape  # data size
data = np.zeros(data_size, dtype=float)  # preallocation for the numerical data
unit = []  # unit of each data
num_var = data_size[1]  # number of variables
for i in range(data_size[1]):
    unit.append(label_data[i].split(" ")[-1])
# Convert strings to float
for i in range(data_size[1]):
    data[:, i] = np.array([float(temp.strip('"')) for temp in data_str[:, i]])
    label_data[i] = label_data[i].strip('"')

# do not change above this line

Y = data.copy()  # data to fit the AR model
N = data.shape[0]  # number of data points
t = np.linspace(0, N, N, endpoint=False)  # time


# plot the raw data
fig, ax = plt.subplots(2, 1, figsize=(18.5, 10.5))
ax[0].set_title(r"Hourly Temperature at YUL (August 2022)")
ax[1].set_title(r"Hourly Electricity Demand in Quebec (August 2022)")
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
    
    # ------------------Modify above------------------

    return A, B


def fit(A, B):
    # Write a function that solves AX=B where X are the AR model parameters

    # ------------------Modify below------------------
    X = np.ones(
        (A.shape[1], B.shape[1]), dtype=float
    )  # a dummy variable, delete and modify
    
    X = np.linalg.solve((A.T)@A,(A.T)@B)
    
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
    
    return y_pred 
    # ------------------Modify above------------------




# %%
# Fit (train) the AR model
N_start = 0  # start day, don't change
N_end = 383  # end day, don't change
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
    a.set_xlabel(r"$t$ (hours)")
ax[0].set_ylabel(r"$T_k$ %s" % unit[0])
ax[1].set_ylabel(r"$e_k %s" % unit[0])
ax[0].plot(t[ell:], y_pred[ell:, 0], label=r"$T_{k, pred, \ell=%s}$" % ell)
ax[0].plot(t[ell:], Y[N_start + ell : N_end, 0], "--", label=r"$T_{k, true}$")
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
ax[0].set_ylabel(r"$P_k$ %s" % unit[1])
ax[1].set_ylabel(r"$e_k$ %s" % unit[1])
ax[0].plot(t[ell:], y_pred[ell:, 1], label=r"$P_{k, pred, \ell=%s}$" % ell)
ax[0].plot(t[ell:], Y[N_start + ell : N_end, 1], "--", label=r"$P_{k, true}$")
ax[1].plot(t[ell:], e[ell:, 1], label=r"$e_{k, \ell=%s}$" % ell)
ax[0].legend(loc="center left", bbox_to_anchor=(1, 0.5))
ax[1].legend(loc="center left", bbox_to_anchor=(1, 0.5))
fig.tight_layout()
# fig.savefig("figs/MVAR_response_train_demand.pdf")
plt.show()


# %%
# Test
Y = data.copy()
N_start = 384  # start day, don't change
N_end = 743  # end day, don't change
t = np.arange(N_start, N_end)  # time, don't change

y_pred = predict(Y[N_start:N_end], X)  # predictions

# Compute various metrics associated with predictinon error

# ------------------Modify below------------------

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

# ------------------Modify above----------------

# %%
# Plotting

fig, ax = plt.subplots(3, 1)
fig.set_size_inches(18.5, 10.5, forward=True)
ax[0].set_title(r"AR Model Test")
for a in np.ravel(ax):
    a.set_xlabel(r"$t$ (hours)")
ax[0].set_ylabel(r"$T_k$ %s" % unit[0])
ax[1].set_ylabel(r"$e_{T,k}$ %s" % unit[0])
ax[2].set_ylabel(r"$e_{T,k, rel}$ (%)")
ax[0].plot(t[ell:], y_pred[ell:, 0], label=r"$T_{k, pred, \ell=%s}$" % ell)
ax[1].plot(t[ell:], e[ell:, 0], label=r"$e_{T,k, \ell=%s}$" % ell)
ax[0].plot(t[ell:], Y[N_start + ell : N_end, 0], "--", label=r"$T_{k, true}$")
ax[2].plot(t[ell:], e_rel[ell:, 0], label=r"$e_{T,k, rel, \ell=%s}$" % ell)
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
ax[0].set_ylabel(r"$P_k$ %s" % unit[1])
ax[1].set_ylabel(r"$e_{P,k}$ %s" % unit[1])
ax[2].set_ylabel(r"$e_{P,k, rel}$ (%)")
ax[0].plot(t[ell:], y_pred[ell:, 1], label=r"$P_{k, pred, \ell=%s}$" % ell)
ax[1].plot(t[ell:], e[ell:, 1], label=r"$e_{P,k, \ell=%s}$" % ell)
ax[0].plot(t[ell:], Y[N_start + ell : N_end, 1], "--", label=r"$P_{k, true}$")
ax[2].plot(t[ell:], e_rel[ell:, 1], label=r"$e_{P,k, rel, \ell=%s}$" % ell)
ax[0].legend(loc="center left", bbox_to_anchor=(1, 0.5))
ax[1].legend(loc="center left", bbox_to_anchor=(1, 0.5))
ax[2].legend(loc="center left", bbox_to_anchor=(1, 0.5))
fig.tight_layout()
# fig.savefig("figs/MVAR_test_demand_ell_%s.pdf" % ell)
plt.show()

print("Mean absolute error (°C, MW) is ", mu, "\n")
print("Absolute error standard deviation (°C, MW) is", sigma, "\n")
print("Mean relative error (percent) is ", mu_e_rel, "\n")
print("Relative error standard deviation (percent) is", sigma_e_rel, "\n")

# %%
