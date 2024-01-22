'''
non-functional multipletest for S&P500 companies using LLS to try and get average error 

(NOT Functional, for now)
'''

#%%
# Import libraries
import numpy as np
import scipy.linalg as linalg
import matplotlib.pyplot as plt
import yfinance as yf
import pandas as pd

stocktray = pd.read_html('https://en.wikipedia.org/wiki/List_of_S%26P_500_companies')[0]['Symbol'].tolist()

stocktray = stocktray[:10]

ROWS = len(stocktray)
print(f'length = {ROWS}\n{stocktray}')

# %%
def _Run_ErrorAnalysis(symbol):

    # Download historical data using yfinance
    #symbol = 'TSLA'
    start_date = '2020-10-01'
    end_date = '2023-11-13'
    Data = yf.download(symbol, start=start_date, end=end_date)
    #print(Data.columns)

    data_o = Data['Open']

    data_c = Data['Close']

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

    Y = data.copy()  # data to fit the AR model
    N = data.shape[0]  # number of data points
    t = np.linspace(0, N, N, endpoint=False)  # time

    #print(f'N: {N},\nData:\n{data}')

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
        # ------------------Modify above------------------

        return A, B


    def fit(A, B):
        # Write a function that solves AX=B where X are the AR model parameters

        # ------------------Modify below------------------
        X = linalg.solve(A.T @ A,A.T @ B) 
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
        # ------------------Modify above------------------
        y_pred_mean = np.zeros((num_var),dtype=float)
        for i in range(num_var):
            y_pred_mean[i] = np.mean(y[:ell,i])
            y_pred[:ell,i] = np.ones(ell) * y_pred_mean[i]
        return y_pred


    # Fit (train) the AR model
    N_start = 0  # start day, don't change
    N_end = int(N/2) # end day, don't change 383
    #print(f'Train\nStart: {N_start}, End: {N_end}')
    ell = 2  # AR model memory, don't change
    t = np.arange(N_start, N_end)  # time, don't change
    y_scale = np.zeros(num_var, dtype=float)  # scale factor preallocation
    y_scale = np.max(np.linalg.norm(Y[N_start:N_end], 2))  # scale factor
    Y = Y / y_scale  # non-dimensionalize the data, don't change
    A, B = form_A_B(Y[N_start:N_end], ell)  # form A, B matrices
    #print(np.linalg.matrix_rank(A))  # shoudl be full rank
    X = fit(A, B)  # find the AR parameters
    Y = Y * y_scale  # dimensionalize the data again
    y_pred = predict(Y[N_start:N_end], X, train=True)  # predictions
    e = np.abs(Y[N_start:N_end] - y_pred)  # absolute error

    # Test
    Y = data.copy()
    N_start = int(N/2) + 1 # start day, don't change 384
    N_end = N  # end day, don't change 743
    #print(f'Test\nStart: {N_start}, End: {N_end}')
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
    # ------------------Modify above----------------
    ErrorMatrix = [[],[],[],[]]
    
    ErrorMatrix[0] = mu
    ErrorMatrix[1] = sigma
    ErrorMatrix[2] = mu_e_rel
    ErrorMatrix[3] = sigma_e_rel

    #print("Mean absolute error ($, $) is ", mu, "\n")
    #print("Absolute error standard deviation ($, $) is", sigma, "\n")
    #print("Mean relative error (percent) is ", mu_e_rel, "\n")
    #print("Relative error standard deviation (percent) is", sigma_e_rel, "\n")
    return ErrorMatrix
# %%
# ERROR CALCULATION FOR MULTIPLE STOCKS

ErrorAcc = np.zeros((ROWS, 4))
count = 0
for i, tic in enumerate(stocktray):
    try:
        ErrorMatrix = _Run_ErrorAnalysis(tic)
        print(ErrorMatrix)
        print(ErrorAcc[i])
        ErrorAcc[i] = ErrorMatrix
        count += 1
        print(f'PROCESSED TICKER, {count}')
    except Exception as e:
        count += 1
        print(f'ERROR WITH TICKER, {count}, {e}')
        continue

meanError = 0
Std = 0

print(ErrorAcc)

#for i in range(ROWS):
    
#%%