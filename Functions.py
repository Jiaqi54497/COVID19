import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from scipy.integrate import odeint
from lmfit import minimize, Parameters, Parameter, report_fit
from scipy.integrate import odeint


"""Functions for SIR model"""

def f(y, t, paras):
    """
    System of differential equations
    """

    s = y[0]
    i = y[1]
    r = y[2]

    try:
        beta = paras['beta'].value
        gamma = paras['gamma'].value

    except KeyError:
        beta, gamma = paras
    # the model equations
    ds_dt = -beta*s*i
    di_dt = beta*s*i - gamma*i
    dr_dt = gamma*i
    return [ds_dt, di_dt, dr_dt]

def f2(y, t, paras):
    """
    System of differential equations
    """

    s = y[0]
    i = y[1]
    r = y[2]

    try:
        beta = paras[0]
        gamma = paras[1]

    except KeyError:
        beta, gamma = paras
    # the model equations
    ds_dt = -beta*s*i
    di_dt = beta*s*i - gamma*i
    dr_dt = gamma*i
    return [ds_dt, di_dt, dr_dt]

def g(t, x0, paras):
    """
    Solution to the ODE x'(t) = f(t,x,k) with initial condition x(0) = x0
    """
    x = odeint(f, x0, t, args=(paras,))
    return x

def g2(t, x0, paras):
    """
    Solution to the ODE x'(t) = f(t,x,k) with initial condition x(0) = x0
    """
    x = odeint(f2, x0, t, args=(paras,))
    return x

def residual(paras, t, i_data, r_data):

    """
    compute the residual between actual data and fitted data
    """

    x0 = paras['s0'].value, paras['i0'].value, paras['r0'].value
    model = g(t, x0, paras)

    # you only have data for one of your variables
    i_model = model[:, 1]

    return (i_model-i_data).ravel()

def run(i_measured, r_measured):
    """
    Run this function to train parameters. Initial values of parameters are written here.
    """

    t_measured = np.linspace(0, len(i_measured), len(i_measured))

# initial conditions
    i0 = i_measured[0]
    r0 = r_measured[0]
    s0 = 1-i0-r0
    y0 = [s0,i0,r0]

    params = Parameters()
    params.add('s0', value=s0, vary=False)
    params.add('i0', value=i0, vary=False)
    params.add('r0', value=r0, vary=False)
    params.add('beta', value=.5, min=0., max=1.)
    params.add('gamma', value=0.5, min=0., max=1.)

# fit model
    result = minimize(residual, params, args=(t_measured, i_measured, r_measured), method='leastsq')  # leastsq nelder
# check results of the fit
    data_fitted = g(t_measured, y0, result.params)

#     report_fit(result)
    return params, data_fitted, result

"""Search for N that produces the best fit."""

def try_N(N, i_measured, r_measured):
    i_rescale=i_measured/N
    r_rescale=r_measured/N
    return run(i_rescale, r_rescale)

def plot_fit(result,i_measured):
    i_fitted=result[1][:,1]
    x=range(len(i_measured))
    plt.plot(x,i_measured,label='true infected')
    plt.plot(x,i_fitted,label='fitted infected')
    plt.legend()
    plt.show()
    
def plot_more(result,i_measured,r_measured):
    i_fitted=result[1][:,1]
    r_fitted=result[1][:,2]
    x=range(len(i_measured))
    plt.plot(x,i_measured,label='true infected')
    plt.plot(x,i_fitted,label='fitted infected')
    plt.plot(x,r_measured,label='true removed')
    plt.plot(x,r_fitted,label='fitted removed')

    plt.legend()
    plt.show()

def N_range(arr,i_measured,r_measured):
    get=[]
    for N in arr:
        result=try_N(N,i_measured, r_measured)
        reletive_error=[]
        for name, param in result[2].params.items():
            reletive_error.append(param.stderr/param.value)
        get.append([N,result[2].chisqr,reletive_error])
    return get

def compare(get):
    chi=[]
    for p in get:
        chi.append(p[0]*p[0]*p[1])
    plt.scatter(range(len(get)),chi)
    plt.show()
    return chi.index(min(chi))

def best_fit(arr,i_measured,r_measured,index):
    N=arr[index]
    result=try_N(N,i_measured, r_measured)
#     plot_more(result,i_measured/N,r_measured/N)
    plot_fit(result,i_measured/N)
    report_fit(result[2])
    return result

def get_params(arr,i_measured,r_measured):
    get=N_range(arr,i_measured,r_measured)
    index=compare(get)
    result=best_fit(arr,i_measured,r_measured,index)
    return arr[index],result[2].params['beta'].value,result[2].params['gamma'].value

def in_one_peak(arr,i1,r1):
    divide=range(0,len(i1),20)
    divide=list(divide)
    divide=divide[1:]
    divide.append(len(i1))
    gogo=[]
    for t in divide:
        gogo.append([t,get_params(arr,i1[:t],r1[:t])])
    return gogo

"""Find data points for all peaks."""
def all_peaks(i_all,r_all,N_all,expand=3,num_points=50):
    gogo=[]
    for i in range(len(i_all)):
        arr=np.linspace(N_all[i],N_all[i]*expand,num_points)
        data=in_one_peak(arr,i_all[i],r_all[i])
        gogo.append(data)
        print('This is the',i+1,'th peak')
    return gogo

"""Calculate the avarage policy for each time period considered."""

def get_ave(name,policy_part):
    ave=[]
    for na in name[1:]:
        ave.append(policy_part[na].mean())
    return ave

def x_array(split,policy,points,name):
    x_array=[]
    # Get the x data used to train the neural network.
    for j in range(len(points)):
        for i in range(len(points[j])):
            policy_part = policy[split[j]:split[j]+points[j][i][0]]
            ave=get_ave(name,policy_part)
            x_array.append(ave)
    return x_array

def y_array(points):
    y_array=[]
    for j in range(len(points)):
        for i in range(len(points[j])):
            y_array.append(points[j][i][1])
    return y_array

"""Delete the poorly fitted data points manually."""
def delete_points(x_array,y_array,delete):
    for i in range(len(delete)):
        x_array.pop(delete[i]-i)
        y_array.pop(delete[i]-i)
    return x_array,y_array

"""Exam the predictions"""
def params_back(yy_0,predictions):
    y=np.array(yy_0)
    mm=y[:,0].max()
    p=np.array(predictions)
    p[:,0]=p[:,0]*mm
    return p.tolist()

def params_add(delete,predictions):
    nn=[0]*len(predictions[0])
    for i in range(len(delete)):
        predictions.insert(delete[i],nn)
    return predictions

def show_one_set(i_measured,r_measured,y_train,y_predict):
    N_t=y_train[0]
    N_p=y_predict[0]
    i_true=i_measured/N_t
    r_true=r_measured/N_t
    i_not_true=i_measured/N_p
    r_not_true=r_measured/N_p
    t_measured = np.linspace(0, len(i_measured), len(i_measured))

    i0 = i_true[0]
    r0 = r_true[0]
    s0 = 1-i0-r0
    y0 = [s0,i0,r0]

    data_fitted = g2(t_measured, y0, y_train)
    i_train=data_fitted[:,1]

    i1 = i_not_true[0]
    r1 = r_not_true[0]
    s1 = 1-i1-r1
    y1 = [s1,i1,r1]

    data_fitted_1 = g2(t_measured, y1, y_predict)
    i_predict=data_fitted_1[:,1]

    print('params_fitted:',y_train)
    print('initial_fitted:',y0)
    print('params_predicted:',y_predict)
    print('initial_predicted:',y1)
    plot_3(i_true,i_train,i_predict)

def plot_3(i_true,i_train,i_predict):
    x=range(len(i_true))
    plt.plot(x,i_true,label='infection measured')
    plt.plot(x,i_train,label='infection fitted')
    plt.plot(x,i_predict,label='infection predicted')
    plt.legend()
    plt.show()

def show_all(i_all,r_all,y_train,y_predict):
    number=0
    for i_peak in range(len(i_all)):
        divide=range(20,len(i_all[i_peak]),20)
        divide=list(divide)
        divide.append(len(i_all[i_peak]))
        for t in divide:
            show_one_set(i_all[i_peak][:t],r_all[i_peak][:t],y_train[number],y_predict[number])
            number=number+1
