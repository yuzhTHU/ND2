# -*- coding: UTF-8 -*-
"""Time series data mapping to a comprehensive library of elementary functions. Library can be customized for special issues.

Created on Wed July 7 15:34:20 2021
@authors: Tingting Gao, Gang Yan

"""

import pandas as pd
import numpy as np 
import itertools as it
import math

def elementary_functions_name(dimensionList,order):
    Combination_func = list(it.combinations_with_replacement(dimensionList,order))
    Num_of_func = len(Combination_func)
    Name_of_func = []
    for i in range(0,Num_of_func):
        tmp = "*".join(Combination_func[i])
        Name_of_func.append(tmp)
    return Num_of_func, Name_of_func

def sigmoidfun(x,alpha,beta):
    sigmoidOutput = 1/(1+np.exp(-alpha*(x-beta)))
    return sigmoidOutput

def tangentH(x):
    # Tanh = (np.exp(x)-np.exp(-x))/(np.exp(x)+np.exp(-x))
    Tanh = np.tanh(x)
    return Tanh

def Regulation_func(x,gamma):
    Regulation_result = (np.abs(x)**gamma)/(np.abs(x)**gamma+1)
    return Regulation_result
        


def Polynomial_functions(TimeSeries, var_names, Nnodes, PolyOrder):
    Timelength = np.size(TimeSeries, 0)
    dim_multi_Nnodes = np.size(TimeSeries, 1)
    dim = len(var_names)
    if PolyOrder >= 1:
        if dim == 1:
            lst = var_names # ['x1']
            Numfunc, Namefunc = elementary_functions_name(lst,1)
            PolyOne = np.zeros(shape=(Timelength*Nnodes,Numfunc))
            for j in range(0,Numfunc):
                for i in range(0,Nnodes):
                    PolyOne[i*Timelength:(i+1)*Timelength,j]  = TimeSeries[:,i]
            column_values = Namefunc
            PolyOne = pd.DataFrame(data = PolyOne, columns = column_values)

        if dim == 2:
            lst = var_names # ['x1','x2']
            Numfunc, Namefunc = elementary_functions_name(lst,1)
            PolyOne = np.zeros(shape=(Timelength*Nnodes,Numfunc))
            for j in range(0,Numfunc):
                for i in range(0,Nnodes):
                    PolyOne[i*Timelength:(i+1)*Timelength,j]  = TimeSeries[:,dim*i+j]
            column_values = Namefunc
            PolyOne = pd.DataFrame(data = PolyOne, columns = column_values) 
        
        if dim == 3:
            lst = var_names # ['x1','x2','x3']
            Numfunc, Namefunc = elementary_functions_name(lst,1)
            PolyOne = np.zeros(shape=(Timelength*Nnodes,Numfunc))
            for j in range(0,Numfunc):
                for i in range(0,Nnodes):
                    PolyOne[i*Timelength:(i+1)*Timelength,j]  = TimeSeries[:,dim*i+j]
            column_values = Namefunc
            PolyOne = pd.DataFrame(data = PolyOne, columns = column_values)

        if dim == 4:
            lst = var_names # ['x1','x2','x3','x4']
            Numfunc, Namefunc = elementary_functions_name(lst,1)
            PolyOne = np.zeros(shape=(Timelength*Nnodes,Numfunc))
            for j in range(0,Numfunc):
                for i in range(0,Nnodes):
                    PolyOne[i*Timelength:(i+1)*Timelength,j]  = TimeSeries[:,dim*i+j]
            column_values = Namefunc
            PolyOne = pd.DataFrame(data = PolyOne, columns = column_values)

        if dim >= 5:
            raise ValueError('dim must be less than 5.')

    if PolyOrder >= 2:
        if dim == 1:
            lst = var_names # ['x1']
            Numfunc, Namefunc = elementary_functions_name(lst,2)
            PolyTwo = np.zeros(shape=(Timelength*Nnodes,Numfunc))
            for i in range(0,Nnodes):
                j = 0
                PolyTwo[i*Timelength:(i+1)*Timelength,j] = TimeSeries[:,dim*i]**2
            column_values = Namefunc
            PolyTwo = pd.DataFrame(data = PolyTwo, columns = column_values)

        if dim == 2:
            lst = var_names # ['x1','x2']
            Numfunc, Namefunc = elementary_functions_name(lst,2)
            PolyTwo = np.zeros(shape=(Timelength*Nnodes,Numfunc))
            for i in range(0,Nnodes):
                j = 0
                for ii in range(0,dim):
                    for jj in range(ii,dim):
                        PolyTwo[i*Timelength:(i+1)*Timelength,j] = TimeSeries[:,dim*i+ii]*TimeSeries[:,dim*i+jj]
                        j = j+1
            column_values = Namefunc
            PolyTwo = pd.DataFrame(data = PolyTwo, columns = column_values)

        if dim == 3:
            lst = var_names # ['x1','x2','x3']
            Numfunc, Namefunc = elementary_functions_name(lst,2)
            PolyTwo = np.zeros(shape=(Timelength*Nnodes,Numfunc))
            for i in range(0,Nnodes):
                j = 0
                for ii in range(0,dim):
                    for jj in range(ii,dim):
                        PolyTwo[i*Timelength:(i+1)*Timelength,j] = TimeSeries[:,dim*i+ii]*TimeSeries[:,dim*i+jj]
                        j = j+1
            column_values = Namefunc
            PolyTwo = pd.DataFrame(data = PolyTwo, columns = column_values)

        if dim == 4:
            lst = var_names # ['x1','x2','x3','x4']
            Numfunc, Namefunc = elementary_functions_name(lst,2)
            PolyTwo = np.zeros(shape=(Timelength*Nnodes,Numfunc))
            for i in range(0,Nnodes):
                j = 0
                for ii in range(0,dim):
                    for jj in range(ii,dim):
                        PolyTwo[i*Timelength:(i+1)*Timelength,j] = TimeSeries[:,dim*i+ii]*TimeSeries[:,dim*i+jj]
                        j = j+1
            column_values = Namefunc
            PolyTwo = pd.DataFrame(data = PolyTwo, columns = column_values)

        
    if PolyOrder >= 3:
        if dim == 1:
            lst = var_names # ['x1']
            Numfunc, Namefunc = elementary_functions_name(lst,3)
            PolyThree = np.zeros(shape=(Timelength*Nnodes,Numfunc))
            for i in range(0,Nnodes):
                j = 0
                for ii in range(0,dim):
                    for jj in range(ii,dim):
                        for kk in range(jj,dim):
                            PolyThree[i*Timelength:(i+1)*Timelength,j] = TimeSeries[:,dim*i+ii]*TimeSeries[:,dim*i+jj]*TimeSeries[:,dim*i+kk]
                            j = j+1
            column_values = Namefunc
            PolyThree = pd.DataFrame(data = PolyThree, columns = column_values)

        if dim == 2:
            lst = var_names # ['x1','x2']
            Numfunc, Namefunc = elementary_functions_name(lst,3)
            PolyThree = np.zeros(shape=(Timelength*Nnodes,Numfunc))
            for i in range(0,Nnodes):
                j = 0
                for ii in range(0,dim):
                    for jj in range(ii,dim):
                        for kk in range(jj,dim):
                            PolyThree[i*Timelength:(i+1)*Timelength,j] = TimeSeries[:,dim*i+ii]*TimeSeries[:,dim*i+jj]*TimeSeries[:,dim*i+kk]
                            j = j+1
            column_values = Namefunc
            PolyThree = pd.DataFrame(data = PolyThree, columns = column_values)

        if dim == 3:
            lst = var_names # ['x1','x2','x3']
            Numfunc, Namefunc = elementary_functions_name(lst,3)
            PolyThree = np.zeros(shape=(Timelength*Nnodes,Numfunc))
            for i in range(0,Nnodes):
                j = 0
                for ii in range(0,dim):
                    for jj in range(ii,dim):
                        for kk in range(jj,dim):
                            PolyThree[i*Timelength:(i+1)*Timelength,j] = TimeSeries[:,dim*i+ii]*TimeSeries[:,dim*i+jj]*TimeSeries[:,dim*i+kk]
                            j = j+1
            column_values = Namefunc
            PolyThree = pd.DataFrame(data = PolyThree, columns = column_values)

        if dim == 4:
            lst = var_names # ['x1','x2','x3','x4']
            Numfunc, Namefunc = elementary_functions_name(lst,3)
            PolyThree = np.zeros(shape=(Timelength*Nnodes,Numfunc))
            for i in range(0,Nnodes):
                j = 0
                for ii in range(0,dim):
                    for jj in range(ii,dim):
                        for kk in range(jj,dim):
                            PolyThree[i*Timelength:(i+1)*Timelength,j] = TimeSeries[:,dim*i+ii]*TimeSeries[:,dim*i+jj]*TimeSeries[:,dim*i+kk]
                            j = j+1
            column_values = Namefunc
            PolyThree = pd.DataFrame(data = PolyThree, columns = column_values)

    if PolyOrder >= 4:
        if dim == 1:
            lst = var_names # ['x1']
            Numfunc, Namefunc = elementary_functions_name(lst,4)
            PolyFour = np.zeros(shape=(Timelength*Nnodes,Numfunc))
            for i in range(0,Nnodes):
                j = 0
                for ii in range(0,dim):
                    for jj in range(ii,dim):
                        for kk in range(jj,dim):
                            for ll in range(kk,dim):
                                PolyFour[i*Timelength:(i+1)*Timelength,j] = TimeSeries[:,dim*i+ii]*TimeSeries[:,dim*i+jj]*TimeSeries[:,dim*i+kk]*TimeSeries[:,dim*i+ll]
                                j = j+1
            column_values = Namefunc
            PolyFour = pd.DataFrame(data = PolyFour, columns = column_values)

        if dim == 2:
            lst = var_names # ['x1','x2']
            Numfunc, Namefunc = elementary_functions_name(lst,4)
            PolyFour = np.zeros(shape=(Timelength*Nnodes,Numfunc))
            for i in range(0,Nnodes):
                j = 0
                for ii in range(0,dim):
                    for jj in range(ii,dim):
                        for kk in range(jj,dim):
                            for ll in range(kk,dim):
                                PolyFour[i*Timelength:(i+1)*Timelength,j] = TimeSeries[:,dim*i+ii]*TimeSeries[:,dim*i+jj]*TimeSeries[:,dim*i+kk]*TimeSeries[:,dim*i+ll]
                                j = j+1
            column_values = Namefunc
            PolyFour = pd.DataFrame(data = PolyFour, columns = column_values)

        if dim == 3:
            lst = var_names # ['x1','x2','x3']
            Numfunc, Namefunc = elementary_functions_name(lst,4)
            PolyFour = np.zeros(shape=(Timelength*Nnodes,Numfunc))
            for i in range(0,Nnodes):
                j = 0
                for ii in range(0,dim):
                    for jj in range(ii,dim):
                        for kk in range(jj,dim):
                            for ll in range(kk,dim):
                                PolyFour[i*Timelength:(i+1)*Timelength,j] = TimeSeries[:,dim*i+ii]*TimeSeries[:,dim*i+jj]*TimeSeries[:,dim*i+kk]*TimeSeries[:,dim*i+ll]
                                j = j+1
            column_values = Namefunc
            PolyFour = pd.DataFrame(data = PolyFour, columns = column_values)

        if dim == 4:
            lst = var_names # ['x1','x2','x3','x4']
            Numfunc, Namefunc = elementary_functions_name(lst,4)
            PolyFour = np.zeros(shape=(Timelength*Nnodes,Numfunc))
            for i in range(0,Nnodes):
                j = 0
                for ii in range(0,dim):
                    for jj in range(ii,dim):
                        for kk in range(jj,dim):
                            for ll in range(kk,dim):
                                PolyFour[i*Timelength:(i+1)*Timelength,j] = TimeSeries[:,dim*i+ii]*TimeSeries[:,dim*i+jj]*TimeSeries[:,dim*i+kk]*TimeSeries[:,dim*i+ll]
                                j = j+1
            column_values = Namefunc
            PolyFour = pd.DataFrame(data = PolyFour, columns = column_values)
    if PolyOrder == 1:
        return PolyOne
    if PolyOrder == 2:
        return pd.concat([PolyOne, PolyTwo], axis=1)
    if PolyOrder == 3:
        return pd.concat([PolyOne, PolyTwo, PolyThree], axis=1)
    if PolyOrder == 4:
        return pd.concat([PolyOne, PolyTwo, PolyThree, PolyFour], axis=1)
    raise ValueError('PolyOrder must be 1, 2, 3, or 4.')

def Trigonometric(TimeSeries, var_names, Nnodes, Sin = True, Cos = True, Tan = True):
    Timelength = np.size(TimeSeries, 0)
    dim_multi_Nnodes = np.size(TimeSeries, 1)
    dim = len(var_names)
    if Sin == True:
        sine = np.zeros(shape=(Timelength*Nnodes,dim))
        for i in range(0,Nnodes):
            for j in range(0,dim):
                sine[i*Timelength:(i+1)*Timelength,j] = np.sin(TimeSeries[:,dim*i+j])
        if dim == 1:
            column_values = [f'sin({var})' for var in var_names] # ['sinx1']
        if dim == 2:
            column_values = [f'sin({var})' for var in var_names] # ['sinx1','sinx2']
        if dim == 3:
            column_values = [f'sin({var})' for var in var_names] # ['sinx1','sinx2','sinx3']
        if dim == 4:
            column_values = [f'sin({var})' for var in var_names] # ['sinx1','sinx2','sinx3','sinx4']
        if dim > 4:
            raise ValueError('dim must be less than 5.')
        sine = pd.DataFrame(data = sine, columns = column_values)

    if Cos == True:
        cosine = np.zeros(shape=(Timelength*Nnodes,dim))
        for i in range(0,Nnodes):
            for j in range(0,dim):
                cosine[i*Timelength:(i+1)*Timelength,j] = np.cos(TimeSeries[:,dim*i+j])
        if dim == 1:
            column_values = [f'cos({var})' for var in var_names] # ['cosx1']
        if dim == 2:
            column_values = [f'cos({var})' for var in var_names] # ['cosx1','cosx2']
        if dim == 3:
            column_values = [f'cos({var})' for var in var_names] # ['cosx1','cosx2','cosx3']
        if dim == 4:
            column_values = [f'cos({var})' for var in var_names] # ['cosx1','cosx2','cosx3','cosx4']
        if dim > 4:
            raise ValueError('dim must be less than 5.')
        cosine = pd.DataFrame(data = cosine, columns = column_values)

    if Tan == True:
        tangent = np.zeros(shape=(Timelength*Nnodes,dim))
        for i in range(0,Nnodes):
            for j in range(0,dim):
                tangent[i*Timelength:(i+1)*Timelength,j] = np.tan(TimeSeries[:,dim*i+j])
        if dim == 1:
            column_values = [f'tan({var})' for var in var_names] # ['tanx1']
        if dim == 2:
            column_values = [f'tan({var})' for var in var_names] # ['tanx1','tanx2']
        if dim == 3:
            column_values = [f'tan({var})' for var in var_names] # ['tanx1','tanx2','tanx3']
        if dim == 4:
            column_values = [f'tan({var})' for var in var_names] # ['tanx1','tanx2','tanx3','tanx4']
        if dim > 4:
            raise ValueError('dim must be less than 5.')
        tangent= pd.DataFrame(data = tangent, columns = column_values)
    return pd.concat([sine, cosine, tangent], axis=1)

def Exponential(TimeSeries, var_names, Nnodes, expomential = True):
    Timelength = np.size(TimeSeries, 0)
    dim_multi_Nnodes = np.size(TimeSeries, 1)
    dim = len(var_names)
    if expomential == True:
        Exp = np.zeros(shape=(Timelength*Nnodes,dim))
        for i in range(0,Nnodes):
            for j in range(0,dim):
                Exp[i*Timelength:(i+1)*Timelength,j] = np.exp(TimeSeries[:,dim*i+j])
        if dim == 1:
            column_values = [f'exp({var})' for var in var_names] # ['expx1']
        if dim == 2:
            column_values = [f'exp({var})' for var in var_names] # ['expx1','expx2']
        if dim == 3:
            column_values = [f'exp({var})' for var in var_names] # ['expx1','expx2','expx3']
        if dim == 4:
            column_values = [f'exp({var})' for var in var_names] # ['expx1','expx2','expx3','expx4']
        if dim > 4:
            raise ValueError('dim must be less than 5.')
        Exp = pd.DataFrame(data = Exp, columns = column_values)
    return Exp

def Fractional(TimeSeries, var_names, Nnodes, fractional = True):
    Timelength = np.size(TimeSeries, 0)
    dim_multi_Nnodes = np.size(TimeSeries, 1)
    dim = len(var_names)
    if fractional == True:
        frac = np.zeros(shape=(Timelength*Nnodes,dim))
        for i in range(0,Nnodes):
            for j in range(0,dim):
                frac[i*Timelength:(i+1)*Timelength,j] = 1/TimeSeries[:,dim*i+j]
        if dim == 1:
            column_values = [f'(1/{var})' for var in var_names] # ['fracx1']
        if dim == 2:
            column_values = [f'(1/{var})' for var in var_names] # ['fracx1','fracx2']
        if dim == 3:
            column_values = [f'(1/{var})' for var in var_names] # ['fracx1','fracx2','fracx3']
        if dim == 4:
            column_values = [f'(1/{var})' for var in var_names] # ['fracx1','fracx2','fracx3','fracx4']
        if dim > 4:
            raise ValueError('dim must be less than 5.')
        frac = pd.DataFrame(data = frac, columns = column_values)
    return frac

def Activation(TimeSeries, var_names, Nnodes, Sigmoid = True, Tanh = True, Regulation = True, alpha=[1, 5, 10], beta=[0, 1, 5, 10], gamma=[1,2,5,10]):
    Timelength = np.size(TimeSeries, 0)
    dim_multi_Nnodes = np.size(TimeSeries, 1)
    dim = len(var_names)
    if Sigmoid == True:
        #alpha = np.linspace(1,10,10)
        # alpha = [1, 5, 10]
        #beta = np.linspace(0,10,11)
        # beta = [0, 1, 5, 10]
        Numfunc = len(alpha)*len(beta)
        sigmoid = np.zeros(shape=(Timelength*Nnodes,dim*Numfunc))
        for i in range(0,Nnodes):
            kk = 0
            for j in range(0,dim):
                for ii in range(0,len(alpha)):
                    for jj in range(0,len(beta)):
                        sigmoid[i*Timelength:(i+1)*Timelength,kk] = 1./(1.+np.exp(-alpha[ii]*(TimeSeries[:,dim*i+j]-beta[jj])))
                        kk = kk+1
        kk = 0
        Sigmoid = pd.DataFrame()
        for j in range(0,dim):
            for ii in range(0,len(alpha)):
                for jj in range(0,len(beta)):
                    tmp = [f"sigmoid(({var_names[j]})*({alpha[ii]})+({beta[jj]}))"]
                    tmp_2 = pd.DataFrame(data = sigmoid[:,kk], columns = tmp)
                    kk = kk+1
                    Sigmoid = pd.concat([Sigmoid,tmp_2], axis=1)
    
    if Tanh == True:
        tanh = np.zeros(shape=(Timelength*Nnodes,dim))
        for i in range(0,Nnodes):
            for j in range(0,dim):
                tanh[i*Timelength:(i+1)*Timelength,j] = tangentH(TimeSeries[:,dim*i+j])
        if dim == 1:
            column_values = [f'tanh({var})' for var in var_names] # ['tanhx1']
        if dim == 2:
            column_values = [f'tanh({var})' for var in var_names] # ['tanhx1','tanhx2']
        if dim == 3:
            column_values = [f'tanh({var})' for var in var_names] # ['tanhx1','tanhx2','tanhx3']
        if dim == 4:
            column_values = [f'tanh({var})' for var in var_names] # ['tanhx1','tanhx2','tanhx3','tanhx4']
        if dim > 4:
            raise ValueError('dim must be less than 5.')
        tanh = pd.DataFrame(data = tanh, columns = column_values)

    if Regulation == True:
        #gamma = np.linspace(0,10,11)
        # gamma = [1,2,5,10]
        Numfunc = len(gamma)
        regulation = np.zeros(shape=(Timelength*Nnodes,dim*Numfunc))
        for i in range(0,Nnodes):
            kk = 0
            for j in range(0,dim):
                for ii in range(0,len(gamma)):
                        regulation[i*Timelength:(i+1)*Timelength,kk] = (TimeSeries[:,dim*i+j]**gamma[ii])/(TimeSeries[:,dim*i+j]**gamma[ii]+1)
                        kk = kk+1

        kk = 0
        Regulation = pd.DataFrame()
        for j in range(0,dim):
            for ii in range(0,len(gamma)):
                tmp = [f"regular({var_names[j]}, {gamma[ii]})"]
                tmp_2 = pd.DataFrame(data = regulation[:,kk], columns = tmp)
                kk = kk+1
                Regulation = pd.concat([Regulation,tmp_2], axis=1)
    return pd.concat([Sigmoid, tanh, Regulation], axis=1)

def rescaling(TimeSeries, var_names, Nnodes, A, Rescal = True):
    Timelength = np.size(TimeSeries, 0)
    dim_multi_Nnodes = np.size(TimeSeries, 1)
    dim = len(var_names)
    kin = np.sum(A>0, axis=1)
    Rescaling = np.zeros(shape=(Timelength*Nnodes,dim))
    if Rescal == True:
        for i in range(0,Nnodes):
            for j in range(0,dim):
                if kin[i] != 0:
                    Rescaling[i*Timelength:(i+1)*Timelength,j] = TimeSeries[:,dim*i+j]/kin[i]
                else:
                    Rescaling[i*Timelength:(i+1)*Timelength,j] = TimeSeries[:,dim*i+j]*0
        if dim == 1:
            column_values = [f'({var})/(aggr(1)+1e-6)' for var in var_names] # ['rescalx1']
        if dim == 2:
            column_values = [f'({var})/(aggr(1)+1e-6)' for var in var_names] # ['rescalx1','rescalx2']
        if dim == 3:
            column_values = [f'({var})/(aggr(1)+1e-6)' for var in var_names] # ['rescalx1','rescalx2','rescalx3']
        if dim == 4:
            column_values = [f'({var})/(aggr(1)+1e-6)' for var in var_names] # ['rescalx1','rescalx2','rescalx3','rescalx4']
        if dim > 4:
            raise ValueError('dim must be less than 5.')
        Rescaling = pd.DataFrame(data = Rescaling, columns = column_values)
    return Rescaling

def coupled_Polynomial_functions(TimeSeries, var_names, Nnodes, A, PolyOrder):
    Timelength = np.size(TimeSeries, 0)
    dim_multi_Nnodes = np.size(TimeSeries, 1)
    dim = len(var_names)
    if PolyOrder>=1:
        if dim == 1:
            column_values = [f'aggr(sour({var}))' for var in var_names] + [f'aggr(targ({var})*sour({var}))' for var in var_names] + [f'aggr((sour({var})-targ({var})))' for var in var_names] # ['x1j','x1ix1j','x1jMinusx1i']
            CoupledPolyOne = np.zeros(shape=(Timelength*Nnodes,len(column_values)))
            for j in range(len(column_values)):
                for i in range(0,Nnodes):
                    if j == 0:
                        tmp = np.zeros(shape=(Timelength,dim))
                        for jj in range(0,Nnodes):
                            tmp[:,0] = tmp[:,0]+A[i,jj]*TimeSeries[:,jj]
                        CoupledPolyOne[i*Timelength:(i+1)*Timelength,j] = tmp[:,0]
                    if j == 1:
                        tmp = np.zeros(shape=(Timelength,dim))
                        for jj in range(0,Nnodes):
                            tmp[:,0] = tmp[:,0]+A[i,jj]*TimeSeries[:,i]*TimeSeries[:,jj]
                        CoupledPolyOne[i*Timelength:(i+1)*Timelength,j] = tmp[:,0]
                    if j == 2:
                        tmp = np.zeros(shape=(Timelength,dim))
                        for jj in range(0,Nnodes):
                            tmp[:,0] = tmp[:,0]+A[i,jj]*(TimeSeries[:,jj]-TimeSeries[:,i])
                        CoupledPolyOne[i*Timelength:(i+1)*Timelength,j] = tmp[:,0]
            CoupledPolyOne = pd.DataFrame(data = CoupledPolyOne, columns = column_values)

        if dim == 2:
            column_values = [f'aggr(sour({var}))' for var in var_names] + [f'aggr(targ({var})*sour({var}))' for var in var_names] + [f'aggr((sour({var})-targ({var})))' for var in var_names] # ['x1j','x2j','x1ix1j','x2ix2j','x1jMinusx1i','x2jMinusx2i']
            CoupledPolyOne = np.zeros(shape=(Timelength*Nnodes,len(column_values)))
            for j in range(len(column_values)):
                for i in range(0,Nnodes):
                    if j == 0:
                        tmp = np.zeros(shape=(Timelength,dim))
                        for jj in range(0,Nnodes):
                            tmp[:,0] = tmp[:,0]+A[i,jj]*TimeSeries[:,jj*dim]
                            tmp[:,1] = tmp[:,1]+A[i,jj]*TimeSeries[:,jj*dim+1]
                        CoupledPolyOne[i*Timelength:(i+1)*Timelength,j] = tmp[:,0]
                        CoupledPolyOne[i*Timelength:(i+1)*Timelength,j+1] = tmp[:,1]

                    if j == 2:
                        tmp = np.zeros(shape=(Timelength,dim))
                        for jj in range(0,Nnodes):
                            tmp[:,0] = tmp[:,0]+A[i,jj]*TimeSeries[:,i*dim]*TimeSeries[:,jj*dim]
                            tmp[:,1] = tmp[:,1]+A[i,jj]*TimeSeries[:,i*dim+1]*TimeSeries[:,jj*dim+1]
                        CoupledPolyOne[i*Timelength:(i+1)*Timelength,j] = tmp[:,0]
                        CoupledPolyOne[i*Timelength:(i+1)*Timelength,j+1] = tmp[:,1]
                    if j == 4:
                        tmp = np.zeros(shape=(Timelength,dim))
                        for jj in range(0,Nnodes):
                            tmp[:,0] = tmp[:,0]+A[i,jj]*(TimeSeries[:,jj*dim]-TimeSeries[:,i*dim])
                            tmp[:,1] = tmp[:,1]+A[i,jj]*(TimeSeries[:,jj*dim+1]-TimeSeries[:,i*dim+1])
                        CoupledPolyOne[i*Timelength:(i+1)*Timelength,j] = tmp[:,0]
                        CoupledPolyOne[i*Timelength:(i+1)*Timelength,j+1] = tmp[:,1]
            CoupledPolyOne = pd.DataFrame(data = CoupledPolyOne, columns = column_values)

        if dim == 3:
            column_values = [f'aggr(sour({var}))' for var in var_names] + [f'aggr(targ({var})*sour({var}))' for var in var_names] + [f'aggr((sour({var})-targ({var})))' for var in var_names] # ['x1j','x2j','x3j','x1ix1j','x2ix2j','x3ix3j','x1jMinusx1i','x2jMinusx2i','x3jMinusx3i']
            CoupledPolyOne = np.zeros(shape=(Timelength*Nnodes,len(column_values)))
            for j in range(len(column_values)):
                for i in range(0,Nnodes):
                    if j == 0:
                        tmp = np.zeros(shape=(Timelength,dim))
                        for jj in range(0,Nnodes):
                            tmp[:,0] = tmp[:,0]+A[i,jj]*TimeSeries[:,jj*dim]
                            tmp[:,1] = tmp[:,1]+A[i,jj]*TimeSeries[:,jj*dim+1]
                            tmp[:,2] = tmp[:,2]+A[i,jj]*TimeSeries[:,jj*dim+2]
                        CoupledPolyOne[i*Timelength:(i+1)*Timelength,j] = tmp[:,0]
                        CoupledPolyOne[i*Timelength:(i+1)*Timelength,j+1] = tmp[:,1]
                        CoupledPolyOne[i*Timelength:(i+1)*Timelength,j+2] = tmp[:,2]

                    if j == 3:
                        tmp = np.zeros(shape=(Timelength,dim))
                        for jj in range(0,Nnodes):
                            tmp[:,0] = tmp[:,0]+A[i,jj]*TimeSeries[:,i*dim]*TimeSeries[:,jj*dim]
                            tmp[:,1] = tmp[:,1]+A[i,jj]*TimeSeries[:,i*dim+1]*TimeSeries[:,jj*dim+1]
                            tmp[:,2] = tmp[:,2]+A[i,jj]*TimeSeries[:,i*dim+2]*TimeSeries[:,jj*dim+2]
                        CoupledPolyOne[i*Timelength:(i+1)*Timelength,j] = tmp[:,0]
                        CoupledPolyOne[i*Timelength:(i+1)*Timelength,j+1] = tmp[:,1]
                        CoupledPolyOne[i*Timelength:(i+1)*Timelength,j+2] = tmp[:,2]
                    if j == 6:
                        tmp = np.zeros(shape=(Timelength,dim))
                        for jj in range(0,Nnodes):
                            tmp[:,0] = tmp[:,0]+A[i,jj]*(TimeSeries[:,jj*dim]-TimeSeries[:,i*dim])
                            tmp[:,1] = tmp[:,1]+A[i,jj]*(TimeSeries[:,jj*dim+1]-TimeSeries[:,i*dim+1])
                            tmp[:,2] = tmp[:,2]+A[i,jj]*(TimeSeries[:,jj*dim+2]-TimeSeries[:,i*dim+2])
                        CoupledPolyOne[i*Timelength:(i+1)*Timelength,j] = tmp[:,0]
                        CoupledPolyOne[i*Timelength:(i+1)*Timelength,j+1] = tmp[:,1]
                        CoupledPolyOne[i*Timelength:(i+1)*Timelength,j+2] = tmp[:,2]
            CoupledPolyOne = pd.DataFrame(data = CoupledPolyOne, columns = column_values)

        if dim == 4:
            column_values = [f'aggr(sour({var}))' for var in var_names] + [f'aggr(targ({var})*sour({var}))' for var in var_names] + [f'aggr((sour({var})-targ({var})))' for var in var_names] # ['x1j','x2j','x3j','x4j','x1ix1j','x2ix2j','x3ix3j','x4ix4j','x1jMinusx1i','x2jMinusx2i','x3jMinusx3i','x4jMinusx4i']
            CoupledPolyOne = np.zeros(shape=(Timelength*Nnodes,len(column_values)))
            for j in range(len(column_values)):
                for i in range(0,Nnodes):
                    if j == 0:
                        tmp = np.zeros(shape=(Timelength,dim))
                        for jj in range(0,Nnodes):
                            tmp[:,0] = tmp[:,0]+A[i,jj]*TimeSeries[:,jj*dim]
                            tmp[:,1] = tmp[:,1]+A[i,jj]*TimeSeries[:,jj*dim+1]
                            tmp[:,2] = tmp[:,2]+A[i,jj]*TimeSeries[:,jj*dim+2]
                            tmp[:,3] = tmp[:,3]+A[i,jj]*TimeSeries[:,jj*dim+3]
                        CoupledPolyOne[i*Timelength:(i+1)*Timelength,j] = tmp[:,0]
                        CoupledPolyOne[i*Timelength:(i+1)*Timelength,j+1] = tmp[:,1]
                        CoupledPolyOne[i*Timelength:(i+1)*Timelength,j+2] = tmp[:,2]
                        CoupledPolyOne[i*Timelength:(i+1)*Timelength,j+3] = tmp[:,3]
                    if j == 4:
                        tmp = np.zeros(shape=(Timelength,dim))
                        for jj in range(0,Nnodes):
                            tmp[:,0] = tmp[:,0]+A[i,jj]*TimeSeries[:,i*dim]*TimeSeries[:,jj*dim]
                            tmp[:,1] = tmp[:,1]+A[i,jj]*TimeSeries[:,i*dim+1]*TimeSeries[:,jj*dim+1]
                            tmp[:,2] = tmp[:,2]+A[i,jj]*TimeSeries[:,i*dim+2]*TimeSeries[:,jj*dim+2]
                            tmp[:,3] = tmp[:,3]+A[i,jj]*TimeSeries[:,i*dim+3]*TimeSeries[:,jj*dim+3]
                        CoupledPolyOne[i*Timelength:(i+1)*Timelength,j] = tmp[:,0]
                        CoupledPolyOne[i*Timelength:(i+1)*Timelength,j+1] = tmp[:,1]
                        CoupledPolyOne[i*Timelength:(i+1)*Timelength,j+2] = tmp[:,2]
                        CoupledPolyOne[i*Timelength:(i+1)*Timelength,j+3] = tmp[:,3]
                    if j == 8:
                        tmp = np.zeros(shape=(Timelength,dim))
                        for jj in range(0,Nnodes):
                            tmp[:,0] = tmp[:,0]+A[i,jj]*(TimeSeries[:,jj*dim]-TimeSeries[:,i*dim])
                            tmp[:,1] = tmp[:,1]+A[i,jj]*(TimeSeries[:,jj*dim+1]-TimeSeries[:,i*dim+1])
                            tmp[:,2] = tmp[:,2]+A[i,jj]*(TimeSeries[:,jj*dim+2]-TimeSeries[:,i*dim+2])
                            tmp[:,3] = tmp[:,3]+A[i,jj]*(TimeSeries[:,jj*dim+3]-TimeSeries[:,i*dim+3])
                        CoupledPolyOne[i*Timelength:(i+1)*Timelength,j] = tmp[:,0]
                        CoupledPolyOne[i*Timelength:(i+1)*Timelength,j+1] = tmp[:,1]
                        CoupledPolyOne[i*Timelength:(i+1)*Timelength,j+2] = tmp[:,2]
                        CoupledPolyOne[i*Timelength:(i+1)*Timelength,j+3] = tmp[:,3]
            CoupledPolyOne = pd.DataFrame(data = CoupledPolyOne, columns = column_values)
        if dim > 4:
            raise ValueError('dim must be less than 5.')
        
    if PolyOrder >= 2:
        if dim == 1:
            column_values = [f'aggr(sour({var})**2)' for var in var_names] + [f'aggr((targ({var})*sour({var}))**2)' for var in var_names] + [f'aggr((sour({var})-targ({var}))**2)' for var in var_names] # ['x1jpow2','x1ix1jpow2','x1jMinusx1ipow2']
            CoupledPolyTwo = np.zeros(shape=(Timelength*Nnodes,len(column_values)))
            for j in range(len(column_values)):
                for i in range(0,Nnodes):
                    if j == 0:
                        tmp = np.zeros(shape=(Timelength,dim))
                        for jj in range(0,Nnodes):
                            tmp[:,0] = tmp[:,0]+A[i,jj]*(TimeSeries[:,jj])**2
                        CoupledPolyTwo[i*Timelength:(i+1)*Timelength,j] = tmp[:,0]
                    if j == 1:
                        tmp = np.zeros(shape=(Timelength,dim))
                        for jj in range(0,Nnodes):
                            tmp[:,0] = tmp[:,0]+A[i,jj]*(TimeSeries[:,i]*TimeSeries[:,jj])**2
                        CoupledPolyTwo[i*Timelength:(i+1)*Timelength,j] = tmp[:,0]
                    if j == 2:
                        tmp = np.zeros(shape=(Timelength,dim))
                        for jj in range(0,Nnodes):
                            tmp[:,0] = tmp[:,0]+A[i,jj]*(TimeSeries[:,jj]-TimeSeries[:,i])**2
                        CoupledPolyTwo[i*Timelength:(i+1)*Timelength,j] = tmp[:,0]
            CoupledPolyTwo = pd.DataFrame(data = CoupledPolyTwo, columns = column_values)

        if dim == 2:
            column_values = [f'aggr(sour({var})**2)' for var in var_names] + [f'aggr((targ({var})*sour({var}))**2)' for var in var_names] + [f'aggr((sour({var})-targ({var}))**2)' for var in var_names] # ['x1jppow2','x2jpow2','x1ix1jpow2','x2ix2jpow2','x1jMinusx1ipow2','x2jMinusx2ipow2']
            CoupledPolyTwo = np.zeros(shape=(Timelength*Nnodes,len(column_values)))
            for j in range(len(column_values)):
                for i in range(0,Nnodes):
                    if j == 0:
                        tmp = np.zeros(shape=(Timelength,dim))
                        for jj in range(0,Nnodes):
                            tmp[:,0] = tmp[:,0]+A[i,jj]*(TimeSeries[:,jj*dim])**2
                            tmp[:,1] = tmp[:,1]+A[i,jj]*(TimeSeries[:,jj*dim+1])**2
                        CoupledPolyTwo[i*Timelength:(i+1)*Timelength,j] = tmp[:,0]
                        CoupledPolyTwo[i*Timelength:(i+1)*Timelength,j+1] = tmp[:,1]

                    if j == 2:
                        tmp = np.zeros(shape=(Timelength,dim))
                        for jj in range(0,Nnodes):
                            tmp[:,0] = tmp[:,0]+A[i,jj]*(TimeSeries[:,i*dim]*TimeSeries[:,jj*dim])**2
                            tmp[:,1] = tmp[:,1]+A[i,jj]*(TimeSeries[:,i*dim+1]*TimeSeries[:,jj*dim+1])**2
                        CoupledPolyTwo[i*Timelength:(i+1)*Timelength,j] = tmp[:,0]
                        CoupledPolyTwo[i*Timelength:(i+1)*Timelength,j+1] = tmp[:,1]
                    if j == 4:
                        tmp = np.zeros(shape=(Timelength,dim))
                        for jj in range(0,Nnodes):
                            tmp[:,0] = tmp[:,0]+A[i,jj]*(TimeSeries[:,jj*dim]-TimeSeries[:,i*dim])**2
                            tmp[:,1] = tmp[:,1]+A[i,jj]*(TimeSeries[:,jj*dim+1]-TimeSeries[:,i*dim+1])**2
                        CoupledPolyTwo[i*Timelength:(i+1)*Timelength,j] = tmp[:,0]
                        CoupledPolyTwo[i*Timelength:(i+1)*Timelength,j+1] = tmp[:,1]
            CoupledPolyTwo = pd.DataFrame(data = CoupledPolyTwo, columns = column_values)

        if dim == 3:
            column_values = [f'aggr(sour({var})**2)' for var in var_names] + [f'aggr((targ({var})*sour({var}))**2)' for var in var_names] + [f'aggr((sour({var})-targ({var}))**2)' for var in var_names] # ['x1jpow2','x2jpow2','x3jpow2','x1ix1jpow2','x2ix2jpow2','x3ix3jpow2','x1jMinusx1ipow2','x2jMinusx2ipow2','x3jMinusx3ipow2']
            CoupledPolyTwo = np.zeros(shape=(Timelength*Nnodes,len(column_values)))
            for j in range(len(column_values)):
                for i in range(0,Nnodes):
                    if j == 0:
                        tmp = np.zeros(shape=(Timelength,dim))
                        for jj in range(0,Nnodes):
                            tmp[:,0] = tmp[:,0]+A[i,jj]*(TimeSeries[:,jj*dim])**2
                            tmp[:,1] = tmp[:,1]+A[i,jj]*(TimeSeries[:,jj*dim+1])**2
                            tmp[:,2] = tmp[:,2]+A[i,jj]*(TimeSeries[:,jj*dim+2])**2
                        CoupledPolyTwo[i*Timelength:(i+1)*Timelength,j] = tmp[:,0]
                        CoupledPolyTwo[i*Timelength:(i+1)*Timelength,j+1] = tmp[:,1]
                        CoupledPolyTwo[i*Timelength:(i+1)*Timelength,j+2] = tmp[:,2]

                    if j == 3:
                        tmp = np.zeros(shape=(Timelength,dim))
                        for jj in range(0,Nnodes):
                            tmp[:,0] = tmp[:,0]+A[i,jj]*(TimeSeries[:,i*dim]*TimeSeries[:,jj*dim])**2
                            tmp[:,1] = tmp[:,1]+A[i,jj]*(TimeSeries[:,i*dim+1]*TimeSeries[:,jj*dim+1])**2
                            tmp[:,2] = tmp[:,2]+A[i,jj]*(TimeSeries[:,i*dim+2]*TimeSeries[:,jj*dim+2])**2
                        CoupledPolyTwo[i*Timelength:(i+1)*Timelength,j] = tmp[:,0]
                        CoupledPolyTwo[i*Timelength:(i+1)*Timelength,j+1] = tmp[:,1]
                        CoupledPolyTwo[i*Timelength:(i+1)*Timelength,j+2] = tmp[:,2]
                    if j == 6:
                        tmp = np.zeros(shape=(Timelength,dim))
                        for jj in range(0,Nnodes):
                            tmp[:,0] = tmp[:,0]+A[i,jj]*(TimeSeries[:,jj*dim]-TimeSeries[:,i*dim])**2
                            tmp[:,1] = tmp[:,1]+A[i,jj]*(TimeSeries[:,jj*dim+1]-TimeSeries[:,i*dim+1])**2
                            tmp[:,2] = tmp[:,2]+A[i,jj]*(TimeSeries[:,jj*dim+2]-TimeSeries[:,i*dim+2])**2
                        CoupledPolyTwo[i*Timelength:(i+1)*Timelength,j] = tmp[:,0]
                        CoupledPolyTwo[i*Timelength:(i+1)*Timelength,j+1] = tmp[:,1]
                        CoupledPolyTwo[i*Timelength:(i+1)*Timelength,j+2] = tmp[:,2]
            CoupledPolyTwo = pd.DataFrame(data = CoupledPolyTwo, columns = column_values)

        if dim == 4:
            column_values = [f'aggr(sour({var})**2)' for var in var_names] + [f'aggr((targ({var})*sour({var}))**2)' for var in var_names] + [f'aggr((sour({var})-targ({var}))**2)' for var in var_names] # ['x1jpow2','x2jpow2','x3jpow2','x4jpow2','x1ix1jpow2','x2ix2jpow2','x3ix3jpow2','x4ix4jpow2','x1jMinusx1ipow2','x2jMinusx2ipow2','x3jMinusx3ipow2','x4jMinusx4ipow2']
            CoupledPolyTwo = np.zeros(shape=(Timelength*Nnodes,len(column_values)))
            for j in range(len(column_values)):
                for i in range(0,Nnodes):
                    if j == 0:
                        tmp = np.zeros(shape=(Timelength,dim))
                        for jj in range(0,Nnodes):
                            tmp[:,0] = tmp[:,0]+A[i,jj]*(TimeSeries[:,jj*dim])**2
                            tmp[:,1] = tmp[:,1]+A[i,jj]*(TimeSeries[:,jj*dim+1])**2
                            tmp[:,2] = tmp[:,2]+A[i,jj]*(TimeSeries[:,jj*dim+2])**2
                            tmp[:,3] = tmp[:,3]+A[i,jj]*(TimeSeries[:,jj*dim+3])**2
                        CoupledPolyTwo[i*Timelength:(i+1)*Timelength,j] = tmp[:,0]
                        CoupledPolyTwo[i*Timelength:(i+1)*Timelength,j+1] = tmp[:,1]
                        CoupledPolyTwo[i*Timelength:(i+1)*Timelength,j+2] = tmp[:,2]
                        CoupledPolyTwo[i*Timelength:(i+1)*Timelength,j+3] = tmp[:,3]
                    if j == 4:
                        tmp = np.zeros(shape=(Timelength,dim))
                        for jj in range(0,Nnodes):
                            tmp[:,0] = tmp[:,0]+A[i,jj]*(TimeSeries[:,i*dim]*TimeSeries[:,jj*dim])**2
                            tmp[:,1] = tmp[:,1]+A[i,jj]*(TimeSeries[:,i*dim+1]*TimeSeries[:,jj*dim+1])**2
                            tmp[:,2] = tmp[:,2]+A[i,jj]*(TimeSeries[:,i*dim+2]*TimeSeries[:,jj*dim+2])**2
                            tmp[:,3] = tmp[:,3]+A[i,jj]*(TimeSeries[:,i*dim+3]*TimeSeries[:,jj*dim+3])**2
                        CoupledPolyTwo[i*Timelength:(i+1)*Timelength,j] = tmp[:,0]
                        CoupledPolyTwo[i*Timelength:(i+1)*Timelength,j+1] = tmp[:,1]
                        CoupledPolyTwo[i*Timelength:(i+1)*Timelength,j+2] = tmp[:,2]
                        CoupledPolyTwo[i*Timelength:(i+1)*Timelength,j+3] = tmp[:,3]
                    if j == 8:
                        tmp = np.zeros(shape=(Timelength,dim))
                        for jj in range(0,Nnodes):
                            tmp[:,0] = tmp[:,0]+A[i,jj]*(TimeSeries[:,jj*dim]-TimeSeries[:,i*dim])**2
                            tmp[:,1] = tmp[:,1]+A[i,jj]*(TimeSeries[:,jj*dim+1]-TimeSeries[:,i*dim+1])**2
                            tmp[:,2] = tmp[:,2]+A[i,jj]*(TimeSeries[:,jj*dim+2]-TimeSeries[:,i*dim+2])**2
                            tmp[:,3] = tmp[:,3]+A[i,jj]*(TimeSeries[:,jj*dim+3]-TimeSeries[:,i*dim+3])**2
                        CoupledPolyTwo[i*Timelength:(i+1)*Timelength,j] = tmp[:,0]
                        CoupledPolyTwo[i*Timelength:(i+1)*Timelength,j+1] = tmp[:,1]
                        CoupledPolyTwo[i*Timelength:(i+1)*Timelength,j+2] = tmp[:,2]
                        CoupledPolyTwo[i*Timelength:(i+1)*Timelength,j+3] = tmp[:,3]
            CoupledPolyTwo = pd.DataFrame(data = CoupledPolyTwo, columns = column_values)
        
        if dim > 4:
            raise ValueError('dim must be less than 5.')

    if PolyOrder == 1:
        return CoupledPolyOne
    if PolyOrder == 2:
        return pd.concat([CoupledPolyOne, CoupledPolyTwo], axis=1)
    if PolyOrder > 2:
        raise ValueError('PolyOrder must be less than 3.')

def Coupled_Trigonometric_functions(TimeSeries, var_names, Nnodes, A, Sine = True, Cos = False, Tan = False):
    Timelength = np.size(TimeSeries, 0)
    dim_multi_Nnodes = np.size(TimeSeries, 1)
    dim = len(var_names)
    if Sine == True:
        if dim == 1:
            column_values = [f'aggr(sin(sour({var})))' for var in var_names] + [f'aggr(sin(targ({var})*sour({var})))' for var in var_names] + [f'aggr(sin(sour({var})-targ({var})))' for var in var_names] + [f'aggr(targ({var})*sin(sour({var})))' for var in var_names] # ['sinx1j','sinx1ix1j','sinx1jMinusx1i','x1isinx1j']
            CoupledSine = np.zeros(shape=(Timelength*Nnodes,len(column_values)))
            for j in range(len(column_values)):
                for i in range(0,Nnodes):
                    if j == 0:
                        tmp = np.zeros(shape=(Timelength,dim))
                        for jj in range(0,Nnodes):
                            tmp[:,0] = tmp[:,0]+A[i,jj]*np.sin(TimeSeries[:,jj])
                        CoupledSine[i*Timelength:(i+1)*Timelength,j] = tmp[:,0]
                    if j == 1:
                        tmp = np.zeros(shape=(Timelength,dim))
                        for jj in range(0,Nnodes):
                            tmp[:,0] = tmp[:,0]+A[i,jj]*np.sin(TimeSeries[:,i]*TimeSeries[:,jj])
                        CoupledSine[i*Timelength:(i+1)*Timelength,j] = tmp[:,0]
                    if j == 2:
                        tmp = np.zeros(shape=(Timelength,dim))
                        for jj in range(0,Nnodes):
                            tmp[:,0] = tmp[:,0]+A[i,jj]*np.sin((TimeSeries[:,jj]-TimeSeries[:,i]))
                        CoupledSine[i*Timelength:(i+1)*Timelength,j] = tmp[:,0]
                    if j == 3:
                        tmp = np.zeros(shape=(Timelength,dim))
                        for jj in range(0,Nnodes):
                            tmp[:,0] = tmp[:,0]+A[i,jj]*np.sin(TimeSeries[:,jj])*TimeSeries[:,i]
                        CoupledSine[i*Timelength:(i+1)*Timelength,j] = tmp[:,0]
            CoupledSine = pd.DataFrame(data = CoupledSine, columns = column_values)

        if dim == 2:
            column_values = [f'aggr(sin(sour({var})))' for var in var_names] + [f'aggr(sin(targ({var})*sour({var})))' for var in var_names] + [f'aggr(sin(sour({var})-targ({var})))' for var in var_names] + [f'aggr(targ({var})*sin(sour({var})))' for var in var_names] # ['sinx1j','sinx2j','sinx1ix1j','sinx2ix2j','sinx1jMinusx1i','sinx2jMinusx2i','x1isinx1j','x2isinx2j']
            CoupledSine = np.zeros(shape=(Timelength*Nnodes,len(column_values)))
            for j in range(len(column_values)):
                for i in range(0,Nnodes):
                    if j == 0:
                        tmp = np.zeros(shape=(Timelength,dim))
                        for jj in range(0,Nnodes):
                            tmp[:,0] = tmp[:,0]+A[i,jj]*np.sin(TimeSeries[:,jj*dim])
                            tmp[:,1] = tmp[:,1]+A[i,jj]*np.sin(TimeSeries[:,jj*dim+1])
                        CoupledSine[i*Timelength:(i+1)*Timelength,j] = tmp[:,0]
                        CoupledSine[i*Timelength:(i+1)*Timelength,j+1] = tmp[:,1]
                    if j == 2:
                        tmp = np.zeros(shape=(Timelength,dim))
                        for jj in range(0,Nnodes):
                            tmp[:,0] = tmp[:,0]+A[i,jj]*np.sin(TimeSeries[:,i*dim]*TimeSeries[:,jj*dim])
                            tmp[:,1] = tmp[:,1]+A[i,jj]*np.sin(TimeSeries[:,i*dim+1]*TimeSeries[:,jj*dim+1])
                        CoupledSine[i*Timelength:(i+1)*Timelength,j] = tmp[:,0]
                        CoupledSine[i*Timelength:(i+1)*Timelength,j+1] = tmp[:,1]
                    if j == 4:
                        tmp = np.zeros(shape=(Timelength,dim))
                        for jj in range(0,Nnodes):
                            tmp[:,0] = tmp[:,0]+A[i,jj]*np.sin(TimeSeries[:,jj*dim]-TimeSeries[:,i*dim])
                            tmp[:,1] = tmp[:,1]+A[i,jj]*np.sin(TimeSeries[:,jj*dim+1]-TimeSeries[:,i*dim+1])
                        CoupledSine[i*Timelength:(i+1)*Timelength,j] = tmp[:,0]
                        CoupledSine[i*Timelength:(i+1)*Timelength,j+1] = tmp[:,1]
                    if j == 6:
                        tmp = np.zeros(shape=(Timelength,dim))
                        for jj in range(0,Nnodes):
                            tmp[:,0] = tmp[:,0]+A[i,jj]*np.sin(TimeSeries[:,jj*dim])*TimeSeries[:,i*dim]
                            tmp[:,1] = tmp[:,1]+A[i,jj]*np.sin(TimeSeries[:,jj*dim+1])*TimeSeries[:,i*dim+1]
                        CoupledSine[i*Timelength:(i+1)*Timelength,j] = tmp[:,0]
                        CoupledSine[i*Timelength:(i+1)*Timelength,j+1] = tmp[:,1]

            CoupledSine = pd.DataFrame(data = CoupledSine, columns = column_values)

        if dim == 3:
            column_values = [f'aggr(sin(sour({var})))' for var in var_names] + [f'aggr(sin(targ({var})*sour({var})))' for var in var_names] + [f'aggr(sin(sour({var})-targ({var})))' for var in var_names] + [f'aggr(targ({var})*sin(sour({var})))' for var in var_names] # ['sinx1j','sinx2j','sinx3j','sinx1ix1j','sinx2ix2j','sinx3ix3j','sinx1jMinusx1i','sinx2jMinusx2i','sinx3jMinusx3i','x1isinx1j','x2isinx2j','x3isinx3j']
            CoupledSine = np.zeros(shape=(Timelength*Nnodes,len(column_values)))
            for j in range(len(column_values)):
                for i in range(0,Nnodes):
                    if j == 0:
                        tmp = np.zeros(shape=(Timelength,dim))
                        for jj in range(0,Nnodes):
                            tmp[:,0] = tmp[:,0]+A[i,jj]*np.sin(TimeSeries[:,jj*dim])
                            tmp[:,1] = tmp[:,1]+A[i,jj]*np.sin(TimeSeries[:,jj*dim+1])
                            tmp[:,2] = tmp[:,2]+A[i,jj]*np.sin(TimeSeries[:,jj*dim+2])
                        CoupledSine[i*Timelength:(i+1)*Timelength,j] = tmp[:,0]
                        CoupledSine[i*Timelength:(i+1)*Timelength,j+1] = tmp[:,1]
                        CoupledSine[i*Timelength:(i+1)*Timelength,j+2] = tmp[:,2]

                    if j == 3:
                        tmp = np.zeros(shape=(Timelength,dim))
                        for jj in range(0,Nnodes):
                            tmp[:,0] = tmp[:,0]+A[i,jj]*np.sin(TimeSeries[:,i*dim]*TimeSeries[:,jj*dim])
                            tmp[:,1] = tmp[:,1]+A[i,jj]*np.sin(TimeSeries[:,i*dim+1]*TimeSeries[:,jj*dim+1])
                            tmp[:,2] = tmp[:,2]+A[i,jj]*np.sin(TimeSeries[:,i*dim+2]*TimeSeries[:,jj*dim+2])
                        CoupledSine[i*Timelength:(i+1)*Timelength,j] = tmp[:,0]
                        CoupledSine[i*Timelength:(i+1)*Timelength,j+1] = tmp[:,1]
                        CoupledSine[i*Timelength:(i+1)*Timelength,j+2] = tmp[:,2]
                    if j == 6:
                        tmp = np.zeros(shape=(Timelength,dim))
                        for jj in range(0,Nnodes):
                            tmp[:,0] = tmp[:,0]+A[i,jj]*np.sin(TimeSeries[:,jj*dim]-TimeSeries[:,i*dim])
                            tmp[:,1] = tmp[:,1]+A[i,jj]*np.sin(TimeSeries[:,jj*dim+1]-TimeSeries[:,i*dim+1])
                            tmp[:,2] = tmp[:,2]+A[i,jj]*np.sin(TimeSeries[:,jj*dim+2]-TimeSeries[:,i*dim+2])
                        CoupledSine[i*Timelength:(i+1)*Timelength,j] = tmp[:,0]
                        CoupledSine[i*Timelength:(i+1)*Timelength,j+1] = tmp[:,1]
                        CoupledSine[i*Timelength:(i+1)*Timelength,j+2] = tmp[:,2]
                    if j == 9:
                        tmp = np.zeros(shape=(Timelength,dim))
                        for jj in range(0,Nnodes):
                            tmp[:,0] = tmp[:,0]+A[i,jj]*np.sin(TimeSeries[:,jj*dim])*TimeSeries[:,i*dim]
                            tmp[:,1] = tmp[:,1]+A[i,jj]*np.sin(TimeSeries[:,jj*dim+1])*TimeSeries[:,i*dim+1]
                            tmp[:,2] = tmp[:,2]+A[i,jj]*np.sin(TimeSeries[:,jj*dim+2])*TimeSeries[:,i*dim+2]
                        CoupledSine[i*Timelength:(i+1)*Timelength,j] = tmp[:,0]
                        CoupledSine[i*Timelength:(i+1)*Timelength,j+1] = tmp[:,1]
                        CoupledSine[i*Timelength:(i+1)*Timelength,j+2] = tmp[:,2]

            CoupledSine = pd.DataFrame(data = CoupledSine, columns = column_values)

        if dim == 4:
            column_values = [f'aggr(sin(sour({var})))' for var in var_names] + [f'aggr(sin(targ({var})*sour({var})))' for var in var_names] + [f'aggr(sin(sour({var})-targ({var})))' for var in var_names] + [f'aggr(targ({var})*sin(sour({var})))' for var in var_names] # ['sinx1j','sinx2j','sinx3j','sinx4j','sinx1ix1j','sinx2ix2j','sinx3ix3j','sinx4ix4j','sinx1jMinusx1i','sinx2jMinusx2i','sinx3jMinusx3i','sinx4jMinusx4i','x1isinx1j','x2isinx2j','x3isinx3j','x4isinx4j']
            CoupledSine = np.zeros(shape=(Timelength*Nnodes,len(column_values)))
            for j in range(len(column_values)):
                for i in range(0,Nnodes):
                    if j == 0:
                        tmp = np.zeros(shape=(Timelength,dim))
                        for jj in range(0,Nnodes):
                            tmp[:,0] = tmp[:,0]+A[i,jj]*np.sin(TimeSeries[:,jj*dim])
                            tmp[:,1] = tmp[:,1]+A[i,jj]*np.sin(TimeSeries[:,jj*dim+1])
                            tmp[:,2] = tmp[:,2]+A[i,jj]*np.sin(TimeSeries[:,jj*dim+2])
                            tmp[:,3] = tmp[:,3]+A[i,jj]*np.sin(TimeSeries[:,jj*dim+3])
                        CoupledSine[i*Timelength:(i+1)*Timelength,j] = tmp[:,0]
                        CoupledSine[i*Timelength:(i+1)*Timelength,j+1] = tmp[:,1]
                        CoupledSine[i*Timelength:(i+1)*Timelength,j+2] = tmp[:,2]
                        CoupledSine[i*Timelength:(i+1)*Timelength,j+3] = tmp[:,3]
                    if j == 4:
                        tmp = np.zeros(shape=(Timelength,dim))
                        for jj in range(0,Nnodes):
                            tmp[:,0] = tmp[:,0]+A[i,jj]*np.sin(TimeSeries[:,i*dim]*TimeSeries[:,jj*dim])
                            tmp[:,1] = tmp[:,1]+A[i,jj]*np.sin(TimeSeries[:,i*dim+1]*TimeSeries[:,jj*dim+1])
                            tmp[:,2] = tmp[:,2]+A[i,jj]*np.sin(TimeSeries[:,i*dim+2]*TimeSeries[:,jj*dim+2])
                            tmp[:,3] = tmp[:,3]+A[i,jj]*np.sin(TimeSeries[:,i*dim+3]*TimeSeries[:,jj*dim+3])
                        CoupledSine[i*Timelength:(i+1)*Timelength,j] = tmp[:,0]
                        CoupledSine[i*Timelength:(i+1)*Timelength,j+1] = tmp[:,1]
                        CoupledSine[i*Timelength:(i+1)*Timelength,j+2] = tmp[:,2]
                        CoupledSine[i*Timelength:(i+1)*Timelength,j+3] = tmp[:,3]
                    if j == 8:
                        tmp = np.zeros(shape=(Timelength,dim))
                        for jj in range(0,Nnodes):
                            tmp[:,0] = tmp[:,0]+A[i,jj]*np.sin(TimeSeries[:,jj*dim]-TimeSeries[:,i*dim])
                            tmp[:,1] = tmp[:,1]+A[i,jj]*np.sin(TimeSeries[:,jj*dim+1]-TimeSeries[:,i*dim+1])
                            tmp[:,2] = tmp[:,2]+A[i,jj]*np.sin(TimeSeries[:,jj*dim+2]-TimeSeries[:,i*dim+2])
                            tmp[:,3] = tmp[:,3]+A[i,jj]*np.sin(TimeSeries[:,jj*dim+3]-TimeSeries[:,i*dim+3])
                        CoupledSine[i*Timelength:(i+1)*Timelength,j] = tmp[:,0]
                        CoupledSine[i*Timelength:(i+1)*Timelength,j+1] = tmp[:,1]
                        CoupledSine[i*Timelength:(i+1)*Timelength,j+2] = tmp[:,2]
                        CoupledSine[i*Timelength:(i+1)*Timelength,j+3] = tmp[:,3]
                    if j == 12:
                        tmp = np.zeros(shape=(Timelength,dim))
                        for jj in range(0,Nnodes):
                            tmp[:,0] = tmp[:,0]+A[i,jj]*np.sin(TimeSeries[:,jj*dim])*TimeSeries[:,i*dim]
                            tmp[:,1] = tmp[:,1]+A[i,jj]*np.sin(TimeSeries[:,jj*dim+1])*TimeSeries[:,i*dim+1]
                            tmp[:,2] = tmp[:,2]+A[i,jj]*np.sin(TimeSeries[:,jj*dim+2])*TimeSeries[:,i*dim+2]
                            tmp[:,3] = tmp[:,3]+A[i,jj]*np.sin(TimeSeries[:,jj*dim+3])*TimeSeries[:,i*dim+3]
                        CoupledSine[i*Timelength:(i+1)*Timelength,j] = tmp[:,0]
                        CoupledSine[i*Timelength:(i+1)*Timelength,j+1] = tmp[:,1]
                        CoupledSine[i*Timelength:(i+1)*Timelength,j+2] = tmp[:,2]
                        CoupledSine[i*Timelength:(i+1)*Timelength,j+3] = tmp[:,3]
                    
            CoupledSine = pd.DataFrame(data = CoupledSine, columns = column_values)

        if dim > 4:
            raise ValueError('dim must be less than 5.')
        
    if Cos == True:
        if dim == 1:
            column_values = [f'aggr(cos(sour({var})))' for var in var_names] + [f'aggr(cos(targ({var})*sour({var})))' for var in var_names] + [f'aggr(cos(sour({var})-targ({var})))' for var in var_names] + [f'aggr(targ({var})*cos(sour({var})))' for var in var_names] # ['cosx1j','cosx1ix1j','cosx1jMinusx1i','x1icosx1j']
            CoupledCosine = np.zeros(shape=(Timelength*Nnodes,len(column_values)))
            for j in range(len(column_values)):
                for i in range(0,Nnodes):
                    if j == 0:
                        tmp = np.zeros(shape=(Timelength,dim))
                        for jj in range(0,Nnodes):
                            tmp[:,0] = tmp[:,0]+A[i,jj]*np.cos(TimeSeries[:,jj])
                        CoupledCosine[i*Timelength:(i+1)*Timelength,j] = tmp[:,0]
                    if j == 1:
                        tmp = np.zeros(shape=(Timelength,dim))
                        for jj in range(0,Nnodes):
                            tmp[:,0] = tmp[:,0]+A[i,jj]*np.cos(TimeSeries[:,i]*TimeSeries[:,jj])
                        CoupledCosine[i*Timelength:(i+1)*Timelength,j] = tmp[:,0]
                    if j == 2:
                        tmp = np.zeros(shape=(Timelength,dim))
                        for jj in range(0,Nnodes):
                            tmp[:,0] = tmp[:,0]+A[i,jj]*np.cos((TimeSeries[:,jj]-TimeSeries[:,i]))
                        CoupledCosine[i*Timelength:(i+1)*Timelength,j] = tmp[:,0]
                    if j == 3:
                        tmp = np.zeros(shape=(Timelength,dim))
                        for jj in range(0,Nnodes):
                            tmp[:,0] = tmp[:,0]+A[i,jj]*np.cos(TimeSeries[:,jj])*TimeSeries[:,i]
                        CoupledCosine[i*Timelength:(i+1)*Timelength,j] = tmp[:,0]
            CoupledCosine = pd.DataFrame(data = CoupledCosine, columns = column_values)

        if dim == 2:
            column_values = [f'aggr(cos(sour({var})))' for var in var_names] + [f'aggr(cos(targ({var})*sour({var})))' for var in var_names] + [f'aggr(cos(sour({var})-targ({var})))' for var in var_names] + [f'aggr(targ({var})*cos(sour({var})))' for var in var_names] # ['cosx1j','cosx2j','cosx1ix1j','cosx2ix2j','cosx1jMinusx1i','cosx2jMinusx2i','x1icosx1j','x2icosx2j']
            CoupledCosine = np.zeros(shape=(Timelength*Nnodes,len(column_values)))
            for j in range(len(column_values)):
                for i in range(0,Nnodes):
                    if j == 0:
                        tmp = np.zeros(shape=(Timelength,dim))
                        for jj in range(0,Nnodes):
                            tmp[:,0] = tmp[:,0]+A[i,jj]*np.cos(TimeSeries[:,jj*dim])
                            tmp[:,1] = tmp[:,1]+A[i,jj]*np.cos(TimeSeries[:,jj*dim+1])
                        CoupledCosine[i*Timelength:(i+1)*Timelength,j] = tmp[:,0]
                        CoupledCosine[i*Timelength:(i+1)*Timelength,j+1] = tmp[:,1]
                    if j == 2:
                        tmp = np.zeros(shape=(Timelength,dim))
                        for jj in range(0,Nnodes):
                            tmp[:,0] = tmp[:,0]+A[i,jj]*np.cos(TimeSeries[:,i*dim]*TimeSeries[:,jj*dim])
                            tmp[:,1] = tmp[:,1]+A[i,jj]*np.cos(TimeSeries[:,i*dim+1]*TimeSeries[:,jj*dim+1])
                        CoupledCosine[i*Timelength:(i+1)*Timelength,j] = tmp[:,0]
                        CoupledCosine[i*Timelength:(i+1)*Timelength,j+1] = tmp[:,1]
                    if j == 4:
                        tmp = np.zeros(shape=(Timelength,dim))
                        for jj in range(0,Nnodes):
                            tmp[:,0] = tmp[:,0]+A[i,jj]*np.cos(TimeSeries[:,jj*dim]-TimeSeries[:,i*dim])
                            tmp[:,1] = tmp[:,1]+A[i,jj]*np.cos(TimeSeries[:,jj*dim+1]-TimeSeries[:,i*dim+1])
                        CoupledCosine[i*Timelength:(i+1)*Timelength,j] = tmp[:,0]
                        CoupledCosine[i*Timelength:(i+1)*Timelength,j+1] = tmp[:,1]
                    if j == 6:
                        tmp = np.zeros(shape=(Timelength,dim))
                        for jj in range(0,Nnodes):
                            tmp[:,0] = tmp[:,0]+A[i,jj]*np.cos(TimeSeries[:,jj*dim])*TimeSeries[:,i*dim]
                            tmp[:,1] = tmp[:,1]+A[i,jj]*np.cos(TimeSeries[:,jj*dim+1])*TimeSeries[:,i*dim+1]
                        CoupledCosine[i*Timelength:(i+1)*Timelength,j] = tmp[:,0]
                        CoupledCosine[i*Timelength:(i+1)*Timelength,j+1] = tmp[:,1]

            CoupledCosine = pd.DataFrame(data = CoupledCosine, columns = column_values)

        if dim == 3:
            column_values = [f'aggr(cos(sour({var})))' for var in var_names] + [f'aggr(cos(targ({var})*sour({var})))' for var in var_names] + [f'aggr(cos(sour({var})-targ({var})))' for var in var_names] + [f'aggr(targ({var})*cos(sour({var})))' for var in var_names] # ['cosx1j','cosx2j','cosx3j','cosx1ix1j','cosx2ix2j','cosx3ix3j','cosx1jMinusx1i','cosx2jMinusx2i','cosx3jMinusx3i','x1icosx1j','x2icosx2j','x3icosx3j']
            CoupledCosine = np.zeros(shape=(Timelength*Nnodes,len(column_values)))
            for j in range(len(column_values)):
                for i in range(0,Nnodes):
                    if j == 0:
                        tmp = np.zeros(shape=(Timelength,dim))
                        for jj in range(0,Nnodes):
                            tmp[:,0] = tmp[:,0]+A[i,jj]*np.cos(TimeSeries[:,jj*dim])
                            tmp[:,1] = tmp[:,1]+A[i,jj]*np.cos(TimeSeries[:,jj*dim+1])
                            tmp[:,2] = tmp[:,2]+A[i,jj]*np.cos(TimeSeries[:,jj*dim+2])
                        CoupledCosine[i*Timelength:(i+1)*Timelength,j] = tmp[:,0]
                        CoupledCosine[i*Timelength:(i+1)*Timelength,j+1] = tmp[:,1]
                        CoupledCosine[i*Timelength:(i+1)*Timelength,j+2] = tmp[:,2]

                    if j == 3:
                        tmp = np.zeros(shape=(Timelength,dim))
                        for jj in range(0,Nnodes):
                            tmp[:,0] = tmp[:,0]+A[i,jj]*np.cos(TimeSeries[:,i*dim]*TimeSeries[:,jj*dim])
                            tmp[:,1] = tmp[:,1]+A[i,jj]*np.cos(TimeSeries[:,i*dim+1]*TimeSeries[:,jj*dim+1])
                            tmp[:,2] = tmp[:,2]+A[i,jj]*np.cos(TimeSeries[:,i*dim+2]*TimeSeries[:,jj*dim+2])
                        CoupledCosine[i*Timelength:(i+1)*Timelength,j] = tmp[:,0]
                        CoupledCosine[i*Timelength:(i+1)*Timelength,j+1] = tmp[:,1]
                        CoupledCosine[i*Timelength:(i+1)*Timelength,j+2] = tmp[:,2]
                    if j == 6:
                        tmp = np.zeros(shape=(Timelength,dim))
                        for jj in range(0,Nnodes):
                            tmp[:,0] = tmp[:,0]+A[i,jj]*np.cos(TimeSeries[:,jj*dim]-TimeSeries[:,i*dim])
                            tmp[:,1] = tmp[:,1]+A[i,jj]*np.cos(TimeSeries[:,jj*dim+1]-TimeSeries[:,i*dim+1])
                            tmp[:,2] = tmp[:,2]+A[i,jj]*np.cos(TimeSeries[:,jj*dim+2]-TimeSeries[:,i*dim+2])
                        CoupledCosine[i*Timelength:(i+1)*Timelength,j] = tmp[:,0]
                        CoupledCosine[i*Timelength:(i+1)*Timelength,j+1] = tmp[:,1]
                        CoupledCosine[i*Timelength:(i+1)*Timelength,j+2] = tmp[:,2]
                    if j == 9:
                        tmp = np.zeros(shape=(Timelength,dim))
                        for jj in range(0,Nnodes):
                            tmp[:,0] = tmp[:,0]+A[i,jj]*np.cos(TimeSeries[:,jj*dim])*TimeSeries[:,i*dim]
                            tmp[:,1] = tmp[:,1]+A[i,jj]*np.cos(TimeSeries[:,jj*dim+1])*TimeSeries[:,i*dim+1]
                            tmp[:,2] = tmp[:,2]+A[i,jj]*np.cos(TimeSeries[:,jj*dim+2])*TimeSeries[:,i*dim+2]
                        CoupledCosine[i*Timelength:(i+1)*Timelength,j] = tmp[:,0]
                        CoupledCosine[i*Timelength:(i+1)*Timelength,j+1] = tmp[:,1]
                        CoupledCosine[i*Timelength:(i+1)*Timelength,j+2] = tmp[:,2]

            CoupledCosine = pd.DataFrame(data = CoupledCosine, columns = column_values)

        if dim == 4:
            column_values = [f'aggr(cos(sour({var})))' for var in var_names] + [f'aggr(cos(targ({var})*sour({var})))' for var in var_names] + [f'aggr(cos(sour({var})-targ({var})))' for var in var_names] + [f'aggr(targ({var})*cos(sour({var})))' for var in var_names] # ['cosx1j','cosx2j','cosx3j','cosx4j','cosx1ix1j','cosx2ix2j','cosx3ix3j','cosx4ix4j','cosx1jMinusx1i','cosx2jMinusx2i','cosx3jMinusx3i','cosx4jMinusx4i','x1icosx1j','x2icosx2j','x3icosx3j','x4icosx4j']
            CoupledCosine = np.zeros(shape=(Timelength*Nnodes,len(column_values)))
            for j in range(len(column_values)):
                for i in range(0,Nnodes):
                    if j == 0:
                        tmp = np.zeros(shape=(Timelength,dim))
                        for jj in range(0,Nnodes):
                            tmp[:,0] = tmp[:,0]+A[i,jj]*np.cos(TimeSeries[:,jj*dim])
                            tmp[:,1] = tmp[:,1]+A[i,jj]*np.cos(TimeSeries[:,jj*dim+1])
                            tmp[:,2] = tmp[:,2]+A[i,jj]*np.cos(TimeSeries[:,jj*dim+2])
                            tmp[:,3] = tmp[:,3]+A[i,jj]*np.cos(TimeSeries[:,jj*dim+3])
                        CoupledCosine[i*Timelength:(i+1)*Timelength,j] = tmp[:,0]
                        CoupledCosine[i*Timelength:(i+1)*Timelength,j+1] = tmp[:,1]
                        CoupledCosine[i*Timelength:(i+1)*Timelength,j+2] = tmp[:,2]
                        CoupledCosine[i*Timelength:(i+1)*Timelength,j+3] = tmp[:,3]
                    if j == 4:
                        tmp = np.zeros(shape=(Timelength,dim))
                        for jj in range(0,Nnodes):
                            tmp[:,0] = tmp[:,0]+A[i,jj]*np.cos(TimeSeries[:,i*dim]*TimeSeries[:,jj*dim])
                            tmp[:,1] = tmp[:,1]+A[i,jj]*np.cos(TimeSeries[:,i*dim+1]*TimeSeries[:,jj*dim+1])
                            tmp[:,2] = tmp[:,2]+A[i,jj]*np.cos(TimeSeries[:,i*dim+2]*TimeSeries[:,jj*dim+2])
                            tmp[:,3] = tmp[:,3]+A[i,jj]*np.cos(TimeSeries[:,i*dim+3]*TimeSeries[:,jj*dim+3])
                        CoupledCosine[i*Timelength:(i+1)*Timelength,j] = tmp[:,0]
                        CoupledCosine[i*Timelength:(i+1)*Timelength,j+1] = tmp[:,1]
                        CoupledCosine[i*Timelength:(i+1)*Timelength,j+2] = tmp[:,2]
                        CoupledCosine[i*Timelength:(i+1)*Timelength,j+3] = tmp[:,3]
                    if j == 8:
                        tmp = np.zeros(shape=(Timelength,dim))
                        for jj in range(0,Nnodes):
                            tmp[:,0] = tmp[:,0]+A[i,jj]*np.cos(TimeSeries[:,jj*dim]-TimeSeries[:,i*dim])
                            tmp[:,1] = tmp[:,1]+A[i,jj]*np.cos(TimeSeries[:,jj*dim+1]-TimeSeries[:,i*dim+1])
                            tmp[:,2] = tmp[:,2]+A[i,jj]*np.cos(TimeSeries[:,jj*dim+2]-TimeSeries[:,i*dim+2])
                            tmp[:,3] = tmp[:,3]+A[i,jj]*np.cos(TimeSeries[:,jj*dim+3]-TimeSeries[:,i*dim+3])
                        CoupledCosine[i*Timelength:(i+1)*Timelength,j] = tmp[:,0]
                        CoupledCosine[i*Timelength:(i+1)*Timelength,j+1] = tmp[:,1]
                        CoupledCosine[i*Timelength:(i+1)*Timelength,j+2] = tmp[:,2]
                        CoupledCosine[i*Timelength:(i+1)*Timelength,j+3] = tmp[:,3]
                    if j == 12:
                        tmp = np.zeros(shape=(Timelength,dim))
                        for jj in range(0,Nnodes):
                            tmp[:,0] = tmp[:,0]+A[i,jj]*np.cos(TimeSeries[:,jj*dim])*TimeSeries[:,i*dim]
                            tmp[:,1] = tmp[:,1]+A[i,jj]*np.cos(TimeSeries[:,jj*dim+1])*TimeSeries[:,i*dim+1]
                            tmp[:,2] = tmp[:,2]+A[i,jj]*np.cos(TimeSeries[:,jj*dim+2])*TimeSeries[:,i*dim+2]
                            tmp[:,3] = tmp[:,3]+A[i,jj]*np.cos(TimeSeries[:,jj*dim+3])*TimeSeries[:,i*dim+3]
                        CoupledCosine[i*Timelength:(i+1)*Timelength,j] = tmp[:,0]
                        CoupledCosine[i*Timelength:(i+1)*Timelength,j+1] = tmp[:,1]
                        CoupledCosine[i*Timelength:(i+1)*Timelength,j+2] = tmp[:,2]
                        CoupledCosine[i*Timelength:(i+1)*Timelength,j+3] = tmp[:,3]
                    
            CoupledCosine = pd.DataFrame(data = CoupledCosine, columns = column_values)

        if dim > 4:
            raise ValueError('dim must be less than 5.')
        
    if Tan == True:
        if dim == 1:
            column_values = [f'aggr(tan(sour({var})))' for var in var_names] + [f'aggr(tan(targ({var})*sour({var})))' for var in var_names] + [f'aggr(tan(sour({var})-targ({var})))' for var in var_names] + [f'aggr(targ({var})*tan(sour({var})))' for var in var_names] # ['tanx1j','tanx1ix1j','tanx1jMinusx1i','x1itanx1j']
            CoupledTangent = np.zeros(shape=(Timelength*Nnodes,len(column_values)))
            for j in range(len(column_values)):
                for i in range(0,Nnodes):
                    if j == 0:
                        tmp = np.zeros(shape=(Timelength,dim))
                        for jj in range(0,Nnodes):
                            tmp[:,0] = tmp[:,0]+A[i,jj]*np.tan(TimeSeries[:,jj])
                        CoupledTangent[i*Timelength:(i+1)*Timelength,j] = tmp[:,0]
                    if j == 1:
                        tmp = np.zeros(shape=(Timelength,dim))
                        for jj in range(0,Nnodes):
                            tmp[:,0] = tmp[:,0]+A[i,jj]*np.tan(TimeSeries[:,i]*TimeSeries[:,jj])
                        CoupledTangent[i*Timelength:(i+1)*Timelength,j] = tmp[:,0]
                    if j == 2:
                        tmp = np.zeros(shape=(Timelength,dim))
                        for jj in range(0,Nnodes):
                            tmp[:,0] = tmp[:,0]+A[i,jj]*np.tan((TimeSeries[:,jj]-TimeSeries[:,i]))
                        CoupledTangent[i*Timelength:(i+1)*Timelength,j] = tmp[:,0]
                    if j == 3:
                        tmp = np.zeros(shape=(Timelength,dim))
                        for jj in range(0,Nnodes):
                            tmp[:,0] = tmp[:,0]+A[i,jj]*np.tan(TimeSeries[:,jj])*TimeSeries[:,i]
                        CoupledTangent[i*Timelength:(i+1)*Timelength,j] = tmp[:,0]
            CoupledTangent = pd.DataFrame(data = CoupledTangent, columns = column_values)

        if dim == 2:
            column_values = [f'aggr(tan(sour({var})))' for var in var_names] + [f'aggr(tan(targ({var})*sour({var})))' for var in var_names] + [f'aggr(tan(sour({var})-targ({var})))' for var in var_names] + [f'aggr(targ({var})*tan(sour({var})))' for var in var_names] # ['tanx1j','tanx2j','tanx1ix1j','tanx2ix2j','tanx1jMinusx1i','tanx2jMinusx2i','x1itanx1j','x2itanx2j']
            CoupledTangent = np.zeros(shape=(Timelength*Nnodes,len(column_values)))
            for j in range(len(column_values)):
                for i in range(0,Nnodes):
                    if j == 0:
                        tmp = np.zeros(shape=(Timelength,dim))
                        for jj in range(0,Nnodes):
                            tmp[:,0] = tmp[:,0]+A[i,jj]*np.tan(TimeSeries[:,jj*dim])
                            tmp[:,1] = tmp[:,1]+A[i,jj]*np.tan(TimeSeries[:,jj*dim+1])
                        CoupledTangent[i*Timelength:(i+1)*Timelength,j] = tmp[:,0]
                        CoupledTangent[i*Timelength:(i+1)*Timelength,j+1] = tmp[:,1]
                    if j == 2:
                        tmp = np.zeros(shape=(Timelength,dim))
                        for jj in range(0,Nnodes):
                            tmp[:,0] = tmp[:,0]+A[i,jj]*np.tan(TimeSeries[:,i*dim]*TimeSeries[:,jj*dim])
                            tmp[:,1] = tmp[:,1]+A[i,jj]*np.tan(TimeSeries[:,i*dim+1]*TimeSeries[:,jj*dim+1])
                        CoupledTangent[i*Timelength:(i+1)*Timelength,j] = tmp[:,0]
                        CoupledTangent[i*Timelength:(i+1)*Timelength,j+1] = tmp[:,1]
                    if j == 4:
                        tmp = np.zeros(shape=(Timelength,dim))
                        for jj in range(0,Nnodes):
                            tmp[:,0] = tmp[:,0]+A[i,jj]*np.tan(TimeSeries[:,jj*dim]-TimeSeries[:,i*dim])
                            tmp[:,1] = tmp[:,1]+A[i,jj]*np.tan(TimeSeries[:,jj*dim+1]-TimeSeries[:,i*dim+1])
                        CoupledTangent[i*Timelength:(i+1)*Timelength,j] = tmp[:,0]
                        CoupledTangent[i*Timelength:(i+1)*Timelength,j+1] = tmp[:,1]
                    if j == 6:
                        tmp = np.zeros(shape=(Timelength,dim))
                        for jj in range(0,Nnodes):
                            tmp[:,0] = tmp[:,0]+A[i,jj]*np.tan(TimeSeries[:,jj*dim])*TimeSeries[:,i*dim]
                            tmp[:,1] = tmp[:,1]+A[i,jj]*np.tan(TimeSeries[:,jj*dim+1])*TimeSeries[:,i*dim+1]
                        CoupledTangent[i*Timelength:(i+1)*Timelength,j] = tmp[:,0]
                        CoupledTangent[i*Timelength:(i+1)*Timelength,j+1] = tmp[:,1]

            CoupledTangent = pd.DataFrame(data = CoupledTangent, columns = column_values)

        if dim == 3:
            column_values = [f'aggr(tan(sour({var})))' for var in var_names] + [f'aggr(tan(targ({var})*sour({var})))' for var in var_names] + [f'aggr(tan(sour({var})-targ({var})))' for var in var_names] + [f'aggr(targ({var})*tan(sour({var})))' for var in var_names] # ['tanx1j','tanx2j','tanx3j','tanx1ix1j','tanx2ix2j','tanx3ix3j','tanx1jMinusx1i','tanx2jMinusx2i','tanx3jMinusx3i','x1itanx1j','x2itanx2j','x3itanx3j']
            CoupledTangent = np.zeros(shape=(Timelength*Nnodes,len(column_values)))
            for j in range(len(column_values)):
                for i in range(0,Nnodes):
                    if j == 0:
                        tmp = np.zeros(shape=(Timelength,dim))
                        for jj in range(0,Nnodes):
                            tmp[:,0] = tmp[:,0]+A[i,jj]*np.tan(TimeSeries[:,jj*dim])
                            tmp[:,1] = tmp[:,1]+A[i,jj]*np.tan(TimeSeries[:,jj*dim+1])
                            tmp[:,2] = tmp[:,2]+A[i,jj]*np.tan(TimeSeries[:,jj*dim+2])
                        CoupledTangent[i*Timelength:(i+1)*Timelength,j] = tmp[:,0]
                        CoupledTangent[i*Timelength:(i+1)*Timelength,j+1] = tmp[:,1]
                        CoupledTangent[i*Timelength:(i+1)*Timelength,j+2] = tmp[:,2]

                    if j == 3:
                        tmp = np.zeros(shape=(Timelength,dim))
                        for jj in range(0,Nnodes):
                            tmp[:,0] = tmp[:,0]+A[i,jj]*np.tan(TimeSeries[:,i*dim]*TimeSeries[:,jj*dim])
                            tmp[:,1] = tmp[:,1]+A[i,jj]*np.tan(TimeSeries[:,i*dim+1]*TimeSeries[:,jj*dim+1])
                            tmp[:,2] = tmp[:,2]+A[i,jj]*np.tan(TimeSeries[:,i*dim+2]*TimeSeries[:,jj*dim+2])
                        CoupledTangent[i*Timelength:(i+1)*Timelength,j] = tmp[:,0]
                        CoupledTangent[i*Timelength:(i+1)*Timelength,j+1] = tmp[:,1]
                        CoupledTangent[i*Timelength:(i+1)*Timelength,j+2] = tmp[:,2]
                    if j == 6:
                        tmp = np.zeros(shape=(Timelength,dim))
                        for jj in range(0,Nnodes):
                            tmp[:,0] = tmp[:,0]+A[i,jj]*np.tan(TimeSeries[:,jj*dim]-TimeSeries[:,i*dim])
                            tmp[:,1] = tmp[:,1]+A[i,jj]*np.tan(TimeSeries[:,jj*dim+1]-TimeSeries[:,i*dim+1])
                            tmp[:,2] = tmp[:,2]+A[i,jj]*np.tan(TimeSeries[:,jj*dim+2]-TimeSeries[:,i*dim+2])
                        CoupledTangent[i*Timelength:(i+1)*Timelength,j] = tmp[:,0]
                        CoupledTangent[i*Timelength:(i+1)*Timelength,j+1] = tmp[:,1]
                        CoupledTangent[i*Timelength:(i+1)*Timelength,j+2] = tmp[:,2]
                    if j == 9:
                        tmp = np.zeros(shape=(Timelength,dim))
                        for jj in range(0,Nnodes):
                            tmp[:,0] = tmp[:,0]+A[i,jj]*np.tan(TimeSeries[:,jj*dim])*TimeSeries[:,i*dim]
                            tmp[:,1] = tmp[:,1]+A[i,jj]*np.tan(TimeSeries[:,jj*dim+1])*TimeSeries[:,i*dim+1]
                            tmp[:,2] = tmp[:,2]+A[i,jj]*np.tan(TimeSeries[:,jj*dim+2])*TimeSeries[:,i*dim+2]
                        CoupledTangent[i*Timelength:(i+1)*Timelength,j] = tmp[:,0]
                        CoupledTangent[i*Timelength:(i+1)*Timelength,j+1] = tmp[:,1]
                        CoupledTangent[i*Timelength:(i+1)*Timelength,j+2] = tmp[:,2]

            CoupledTangent = pd.DataFrame(data = CoupledTangent, columns = column_values)

        if dim == 4:
            column_values = [f'aggr(tan(sour({var})))' for var in var_names] + [f'aggr(tan(targ({var})*sour({var})))' for var in var_names] + [f'aggr(tan(sour({var})-targ({var})))' for var in var_names] + [f'aggr(targ({var})*tan(sour({var})))' for var in var_names] # ['tanx1j','tanx2j','tanx3j','tanx4j','tanx1ix1j','tanx2ix2j','tanx3ix3j','tanx4ix4j','tanx1jMinusx1i','tanx2jMinusx2i','tanx3jMinusx3i','tanx4jMinusx4i','x1itanx1j','x2itanx2j','x3itanx3j','x4itanx4j']
            CoupledTangent = np.zeros(shape=(Timelength*Nnodes,len(column_values)))
            for j in range(len(column_values)):
                for i in range(0,Nnodes):
                    if j == 0:
                        tmp = np.zeros(shape=(Timelength,dim))
                        for jj in range(0,Nnodes):
                            tmp[:,0] = tmp[:,0]+A[i,jj]*np.tan(TimeSeries[:,jj*dim])
                            tmp[:,1] = tmp[:,1]+A[i,jj]*np.tan(TimeSeries[:,jj*dim+1])
                            tmp[:,2] = tmp[:,2]+A[i,jj]*np.tan(TimeSeries[:,jj*dim+2])
                            tmp[:,3] = tmp[:,3]+A[i,jj]*np.tan(TimeSeries[:,jj*dim+3])
                        CoupledTangent[i*Timelength:(i+1)*Timelength,j] = tmp[:,0]
                        CoupledTangent[i*Timelength:(i+1)*Timelength,j+1] = tmp[:,1]
                        CoupledTangent[i*Timelength:(i+1)*Timelength,j+2] = tmp[:,2]
                        CoupledTangent[i*Timelength:(i+1)*Timelength,j+3] = tmp[:,3]
                    if j == 4:
                        tmp = np.zeros(shape=(Timelength,dim))
                        for jj in range(0,Nnodes):
                            tmp[:,0] = tmp[:,0]+A[i,jj]*np.tan(TimeSeries[:,i*dim]*TimeSeries[:,jj*dim])
                            tmp[:,1] = tmp[:,1]+A[i,jj]*np.tan(TimeSeries[:,i*dim+1]*TimeSeries[:,jj*dim+1])
                            tmp[:,2] = tmp[:,2]+A[i,jj]*np.tan(TimeSeries[:,i*dim+2]*TimeSeries[:,jj*dim+2])
                            tmp[:,3] = tmp[:,3]+A[i,jj]*np.tan(TimeSeries[:,i*dim+3]*TimeSeries[:,jj*dim+3])
                        CoupledTangent[i*Timelength:(i+1)*Timelength,j] = tmp[:,0]
                        CoupledTangent[i*Timelength:(i+1)*Timelength,j+1] = tmp[:,1]
                        CoupledTangent[i*Timelength:(i+1)*Timelength,j+2] = tmp[:,2]
                        CoupledTangent[i*Timelength:(i+1)*Timelength,j+3] = tmp[:,3]
                    if j == 8:
                        tmp = np.zeros(shape=(Timelength,dim))
                        for jj in range(0,Nnodes):
                            tmp[:,0] = tmp[:,0]+A[i,jj]*np.tan(TimeSeries[:,jj*dim]-TimeSeries[:,i*dim])
                            tmp[:,1] = tmp[:,1]+A[i,jj]*np.tan(TimeSeries[:,jj*dim+1]-TimeSeries[:,i*dim+1])
                            tmp[:,2] = tmp[:,2]+A[i,jj]*np.tan(TimeSeries[:,jj*dim+2]-TimeSeries[:,i*dim+2])
                            tmp[:,3] = tmp[:,3]+A[i,jj]*np.tan(TimeSeries[:,jj*dim+3]-TimeSeries[:,i*dim+3])
                        CoupledTangent[i*Timelength:(i+1)*Timelength,j] = tmp[:,0]
                        CoupledTangent[i*Timelength:(i+1)*Timelength,j+1] = tmp[:,1]
                        CoupledTangent[i*Timelength:(i+1)*Timelength,j+2] = tmp[:,2]
                        CoupledTangent[i*Timelength:(i+1)*Timelength,j+3] = tmp[:,3]
                    if j == 12:
                        tmp = np.zeros(shape=(Timelength,dim))
                        for jj in range(0,Nnodes):
                            tmp[:,0] = tmp[:,0]+A[i,jj]*np.tan(TimeSeries[:,jj*dim])*TimeSeries[:,i*dim]
                            tmp[:,1] = tmp[:,1]+A[i,jj]*np.tan(TimeSeries[:,jj*dim+1])*TimeSeries[:,i*dim+1]
                            tmp[:,2] = tmp[:,2]+A[i,jj]*np.tan(TimeSeries[:,jj*dim+2])*TimeSeries[:,i*dim+2]
                            tmp[:,3] = tmp[:,3]+A[i,jj]*np.tan(TimeSeries[:,jj*dim+3])*TimeSeries[:,i*dim+3]
                        CoupledTangent[i*Timelength:(i+1)*Timelength,j] = tmp[:,0]
                        CoupledTangent[i*Timelength:(i+1)*Timelength,j+1] = tmp[:,1]
                        CoupledTangent[i*Timelength:(i+1)*Timelength,j+2] = tmp[:,2]
                        CoupledTangent[i*Timelength:(i+1)*Timelength,j+3] = tmp[:,3]
                    
            CoupledTangent = pd.DataFrame(data = CoupledTangent, columns = column_values)

        if dim > 4:
            raise ValueError('dim must be less than 5.')

    CoupledTrignometric = pd.DataFrame()
    if Sine == True:
        CoupledTrignometric = pd.concat([CoupledTrignometric,CoupledSine],axis=1)
    if Cos == True:
        CoupledTrignometric = pd.concat([CoupledTrignometric, CoupledCosine], axis=1)
    if Tan == True:
        CoupledTrignometric = pd.concat([CoupledTrignometric, CoupledTangent], axis=1)
    return CoupledTrignometric


def Coupled_Exponential_functions(TimeSeries, var_names, Nnodes, A, Exponential = True):
    Timelength = np.size(TimeSeries, 0)
    dim_multi_Nnodes = np.size(TimeSeries, 1)
    dim = len(var_names)
    if Exponential == True:
        if dim == 1:
            column_values = [f'aggr(exp(sour({var})))' for var in var_names] + [f'aggr(exp(targ({var})*sour({var})))' for var in var_names] + [f'aggr(exp(sour({var})-targ({var})))' for var in var_names] + [f'aggr(targ({var})*exp(sour({var})))' for var in var_names] # ['expx1j','expx1ix1j','expx1jMinusx1i','x1iexpx1j']
            CoupledExp = np.zeros(shape=(Timelength*Nnodes,len(column_values)))
            for j in range(len(column_values)):
                for i in range(0,Nnodes):
                    if j == 0:
                        tmp = np.zeros(shape=(Timelength,dim))
                        for jj in range(0,Nnodes):
                            tmp[:,0] = tmp[:,0]+A[i,jj]*np.exp(TimeSeries[:,jj])
                        CoupledExp[i*Timelength:(i+1)*Timelength,j] = tmp[:,0]
                    if j == 1:
                        tmp = np.zeros(shape=(Timelength,dim))
                        for jj in range(0,Nnodes):
                            tmp[:,0] = tmp[:,0]+A[i,jj]*np.exp(TimeSeries[:,i]*TimeSeries[:,jj])
                        CoupledExp[i*Timelength:(i+1)*Timelength,j] = tmp[:,0]
                    if j == 2:
                        tmp = np.zeros(shape=(Timelength,dim))
                        for jj in range(0,Nnodes):
                            tmp[:,0] = tmp[:,0]+A[i,jj]*np.exp((TimeSeries[:,jj]-TimeSeries[:,i]))
                        CoupledExp[i*Timelength:(i+1)*Timelength,j] = tmp[:,0]
                    if j == 3:
                        tmp = np.zeros(shape=(Timelength,dim))
                        for jj in range(0,Nnodes):
                            tmp[:,0] = tmp[:,0]+A[i,jj]*np.exp(TimeSeries[:,jj])*TimeSeries[:,i]
                        CoupledExp[i*Timelength:(i+1)*Timelength,j] = tmp[:,0]
            CoupledExp = pd.DataFrame(data = CoupledExp, columns = column_values)

        if dim == 2:
            column_values = [f'aggr(exp(sour({var})))' for var in var_names] + [f'aggr(exp(targ({var})*sour({var})))' for var in var_names] + [f'aggr(exp(sour({var})-targ({var})))' for var in var_names] + [f'aggr(targ({var})*exp(sour({var})))' for var in var_names] # ['expx1j','expx2j','expx1ix1j','expx2ix2j','expx1jMinusx1i','expx2jMinusx2i','x1iexpx1j','x2iexpx2j']
            CoupledExp = np.zeros(shape=(Timelength*Nnodes,len(column_values)))
            for j in range(len(column_values)):
                for i in range(0,Nnodes):
                    if j == 0:
                        tmp = np.zeros(shape=(Timelength,dim))
                        for jj in range(0,Nnodes):
                            tmp[:,0] = tmp[:,0]+A[i,jj]*np.exp(TimeSeries[:,jj*dim])
                            tmp[:,1] = tmp[:,1]+A[i,jj]*np.exp(TimeSeries[:,jj*dim+1])
                        CoupledExp[i*Timelength:(i+1)*Timelength,j] = tmp[:,0]
                        CoupledExp[i*Timelength:(i+1)*Timelength,j+1] = tmp[:,1]
                    if j == 2:
                        tmp = np.zeros(shape=(Timelength,dim))
                        for jj in range(0,Nnodes):
                            tmp[:,0] = tmp[:,0]+A[i,jj]*np.exp(TimeSeries[:,i*dim]*TimeSeries[:,jj*dim])
                            tmp[:,1] = tmp[:,1]+A[i,jj]*np.exp(TimeSeries[:,i*dim+1]*TimeSeries[:,jj*dim+1])
                        CoupledExp[i*Timelength:(i+1)*Timelength,j] = tmp[:,0]
                        CoupledExp[i*Timelength:(i+1)*Timelength,j+1] = tmp[:,1]
                    if j == 4:
                        tmp = np.zeros(shape=(Timelength,dim))
                        for jj in range(0,Nnodes):
                            tmp[:,0] = tmp[:,0]+A[i,jj]*np.exp(TimeSeries[:,jj*dim]-TimeSeries[:,i*dim])
                            tmp[:,1] = tmp[:,1]+A[i,jj]*np.exp(TimeSeries[:,jj*dim+1]-TimeSeries[:,i*dim+1])
                        CoupledExp[i*Timelength:(i+1)*Timelength,j] = tmp[:,0]
                        CoupledExp[i*Timelength:(i+1)*Timelength,j+1] = tmp[:,1]
                    if j == 6:
                        tmp = np.zeros(shape=(Timelength,dim))
                        for jj in range(0,Nnodes):
                            tmp[:,0] = tmp[:,0]+A[i,jj]*np.exp(TimeSeries[:,jj*dim])*TimeSeries[:,i*dim]
                            tmp[:,1] = tmp[:,1]+A[i,jj]*np.exp(TimeSeries[:,jj*dim+1])*TimeSeries[:,i*dim+1]
                        CoupledExp[i*Timelength:(i+1)*Timelength,j] = tmp[:,0]
                        CoupledExp[i*Timelength:(i+1)*Timelength,j+1] = tmp[:,1]

            CoupledExp = pd.DataFrame(data = CoupledExp, columns = column_values)

        if dim == 3:
            column_values = [f'aggr(exp(sour({var})))' for var in var_names] + [f'aggr(exp(targ({var})*sour({var})))' for var in var_names] + [f'aggr(exp(sour({var})-targ({var})))' for var in var_names] + [f'aggr(targ({var})*exp(sour({var})))' for var in var_names] # ['expx1j','expx2j','expx3j','expx1ix1j','expx2ix2j','expx3ix3j','expx1jMinusx1i','expx2jMinusx2i','expx3jMinusx3i','x1iexpx1j','x2iexpx2j','x3iexpx3j']
            CoupledExp = np.zeros(shape=(Timelength*Nnodes,len(column_values)))
            for j in range(len(column_values)):
                for i in range(0,Nnodes):
                    if j == 0:
                        tmp = np.zeros(shape=(Timelength,dim))
                        for jj in range(0,Nnodes):
                            tmp[:,0] = tmp[:,0]+A[i,jj]*np.exp(TimeSeries[:,jj*dim])
                            tmp[:,1] = tmp[:,1]+A[i,jj]*np.exp(TimeSeries[:,jj*dim+1])
                            tmp[:,2] = tmp[:,2]+A[i,jj]*np.exp(TimeSeries[:,jj*dim+2])
                        CoupledExp[i*Timelength:(i+1)*Timelength,j] = tmp[:,0]
                        CoupledExp[i*Timelength:(i+1)*Timelength,j+1] = tmp[:,1]
                        CoupledExp[i*Timelength:(i+1)*Timelength,j+2] = tmp[:,2]

                    if j == 3:
                        tmp = np.zeros(shape=(Timelength,dim))
                        for jj in range(0,Nnodes):
                            tmp[:,0] = tmp[:,0]+A[i,jj]*np.exp(TimeSeries[:,i*dim]*TimeSeries[:,jj*dim])
                            tmp[:,1] = tmp[:,1]+A[i,jj]*np.exp(TimeSeries[:,i*dim+1]*TimeSeries[:,jj*dim+1])
                            tmp[:,2] = tmp[:,2]+A[i,jj]*np.exp(TimeSeries[:,i*dim+2]*TimeSeries[:,jj*dim+2])
                        CoupledExp[i*Timelength:(i+1)*Timelength,j] = tmp[:,0]
                        CoupledExp[i*Timelength:(i+1)*Timelength,j+1] = tmp[:,1]
                        CoupledExp[i*Timelength:(i+1)*Timelength,j+2] = tmp[:,2]
                    if j == 6:
                        tmp = np.zeros(shape=(Timelength,dim))
                        for jj in range(0,Nnodes):
                            tmp[:,0] = tmp[:,0]+A[i,jj]*np.exp(TimeSeries[:,jj*dim]-TimeSeries[:,i*dim])
                            tmp[:,1] = tmp[:,1]+A[i,jj]*np.exp(TimeSeries[:,jj*dim+1]-TimeSeries[:,i*dim+1])
                            tmp[:,2] = tmp[:,2]+A[i,jj]*np.exp(TimeSeries[:,jj*dim+2]-TimeSeries[:,i*dim+2])
                        CoupledExp[i*Timelength:(i+1)*Timelength,j] = tmp[:,0]
                        CoupledExp[i*Timelength:(i+1)*Timelength,j+1] = tmp[:,1]
                        CoupledExp[i*Timelength:(i+1)*Timelength,j+2] = tmp[:,2]
                    if j == 9:
                        tmp = np.zeros(shape=(Timelength,dim))
                        for jj in range(0,Nnodes):
                            tmp[:,0] = tmp[:,0]+A[i,jj]*np.exp(TimeSeries[:,jj*dim])*TimeSeries[:,i*dim]
                            tmp[:,1] = tmp[:,1]+A[i,jj]*np.exp(TimeSeries[:,jj*dim+1])*TimeSeries[:,i*dim+1]
                            tmp[:,2] = tmp[:,2]+A[i,jj]*np.exp(TimeSeries[:,jj*dim+2])*TimeSeries[:,i*dim+2]
                        CoupledExp[i*Timelength:(i+1)*Timelength,j] = tmp[:,0]
                        CoupledExp[i*Timelength:(i+1)*Timelength,j+1] = tmp[:,1]
                        CoupledExp[i*Timelength:(i+1)*Timelength,j+2] = tmp[:,2]

            CoupledExp = pd.DataFrame(data = CoupledExp, columns = column_values)

        if dim == 4:
            column_values = [f'aggr(exp(sour({var})))' for var in var_names] + [f'aggr(exp(targ({var})*sour({var})))' for var in var_names] + [f'aggr(exp(sour({var})-targ({var})))' for var in var_names] + [f'aggr(targ({var})*exp(sour({var})))' for var in var_names] # ['expx1j','expx2j','expx3j','expx4j','expx1ix1j','expx2ix2j','expx3ix3j','expx4ix4j','expx1jMinusx1i','expx2jMinusx2i','expx3jMinusx3i','expx4jMinusx4i','x1iexpx1j','x2iexpx2j','x3iexpx3j','x4iexpx4j']
            CoupledExp = np.zeros(shape=(Timelength*Nnodes,len(column_values)))
            for j in range(len(column_values)):
                for i in range(0,Nnodes):
                    if j == 0:
                        tmp = np.zeros(shape=(Timelength,dim))
                        for jj in range(0,Nnodes):
                            tmp[:,0] = tmp[:,0]+A[i,jj]*np.exp(TimeSeries[:,jj*dim])
                            tmp[:,1] = tmp[:,1]+A[i,jj]*np.exp(TimeSeries[:,jj*dim+1])
                            tmp[:,2] = tmp[:,2]+A[i,jj]*np.exp(TimeSeries[:,jj*dim+2])
                            tmp[:,3] = tmp[:,3]+A[i,jj]*np.exp(TimeSeries[:,jj*dim+3])
                        CoupledExp[i*Timelength:(i+1)*Timelength,j] = tmp[:,0]
                        CoupledExp[i*Timelength:(i+1)*Timelength,j+1] = tmp[:,1]
                        CoupledExp[i*Timelength:(i+1)*Timelength,j+2] = tmp[:,2]
                        CoupledExp[i*Timelength:(i+1)*Timelength,j+3] = tmp[:,3]
                    if j == 4:
                        tmp = np.zeros(shape=(Timelength,dim))
                        for jj in range(0,Nnodes):
                            tmp[:,0] = tmp[:,0]+A[i,jj]*np.exp(TimeSeries[:,i*dim]*TimeSeries[:,jj*dim])
                            tmp[:,1] = tmp[:,1]+A[i,jj]*np.exp(TimeSeries[:,i*dim+1]*TimeSeries[:,jj*dim+1])
                            tmp[:,2] = tmp[:,2]+A[i,jj]*np.exp(TimeSeries[:,i*dim+2]*TimeSeries[:,jj*dim+2])
                            tmp[:,3] = tmp[:,3]+A[i,jj]*np.exp(TimeSeries[:,i*dim+3]*TimeSeries[:,jj*dim+3])
                        CoupledExp[i*Timelength:(i+1)*Timelength,j] = tmp[:,0]
                        CoupledExp[i*Timelength:(i+1)*Timelength,j+1] = tmp[:,1]
                        CoupledExp[i*Timelength:(i+1)*Timelength,j+2] = tmp[:,2]
                        CoupledExp[i*Timelength:(i+1)*Timelength,j+3] = tmp[:,3]
                    if j == 8:
                        tmp = np.zeros(shape=(Timelength,dim))
                        for jj in range(0,Nnodes):
                            tmp[:,0] = tmp[:,0]+A[i,jj]*np.exp(TimeSeries[:,jj*dim]-TimeSeries[:,i*dim])
                            tmp[:,1] = tmp[:,1]+A[i,jj]*np.exp(TimeSeries[:,jj*dim+1]-TimeSeries[:,i*dim+1])
                            tmp[:,2] = tmp[:,2]+A[i,jj]*np.exp(TimeSeries[:,jj*dim+2]-TimeSeries[:,i*dim+2])
                            tmp[:,3] = tmp[:,3]+A[i,jj]*np.exp(TimeSeries[:,jj*dim+3]-TimeSeries[:,i*dim+3])
                        CoupledExp[i*Timelength:(i+1)*Timelength,j] = tmp[:,0]
                        CoupledExp[i*Timelength:(i+1)*Timelength,j+1] = tmp[:,1]
                        CoupledExp[i*Timelength:(i+1)*Timelength,j+2] = tmp[:,2]
                        CoupledExp[i*Timelength:(i+1)*Timelength,j+3] = tmp[:,3]
                    if j == 12:
                        tmp = np.zeros(shape=(Timelength,dim))
                        for jj in range(0,Nnodes):
                            tmp[:,0] = tmp[:,0]+A[i,jj]*np.exp(TimeSeries[:,jj*dim])*TimeSeries[:,i*dim]
                            tmp[:,1] = tmp[:,1]+A[i,jj]*np.exp(TimeSeries[:,jj*dim+1])*TimeSeries[:,i*dim+1]
                            tmp[:,2] = tmp[:,2]+A[i,jj]*np.exp(TimeSeries[:,jj*dim+2])*TimeSeries[:,i*dim+2]
                            tmp[:,3] = tmp[:,3]+A[i,jj]*np.exp(TimeSeries[:,jj*dim+3])*TimeSeries[:,i*dim+3]
                        CoupledExp[i*Timelength:(i+1)*Timelength,j] = tmp[:,0]
                        CoupledExp[i*Timelength:(i+1)*Timelength,j+1] = tmp[:,1]
                        CoupledExp[i*Timelength:(i+1)*Timelength,j+2] = tmp[:,2]
                        CoupledExp[i*Timelength:(i+1)*Timelength,j+3] = tmp[:,3]
                    
            CoupledExp = pd.DataFrame(data = CoupledExp, columns = column_values)
        if dim > 4:
            raise ValueError('dim must be less than 5.')

    return CoupledExp

def Coupled_Fractional_functions(TimeSeries, var_names, Nnodes, A, Fractional = True):
    Timelength = np.size(TimeSeries, 0)
    dim_multi_Nnodes = np.size(TimeSeries, 1)
    dim = len(var_names)
    if Fractional == True:
        if dim == 1:
            column_values = [f'aggr(1/sour({var}))' for var in var_names] + [f'aggr(1/(targ({var})*sour({var})))' for var in var_names] + [f'aggr(1/(sour({var})-targ({var})))' for var in var_names] + [f'aggr(targ({var})/sour({var}))' for var in var_names] # ['fracx1j','fracx1ix1j','fracx1jMinusx1i','x1ifracx1j']
            CoupledFraction = np.zeros(shape=(Timelength*Nnodes,len(column_values)))
            for j in range(len(column_values)):
                for i in range(0,Nnodes):
                    if j == 0:
                        tmp = np.zeros(shape=(Timelength,dim))
                        for jj in range(0,Nnodes):
                            tmp[:,0] = tmp[:,0]+A[i,jj]*1/(TimeSeries[:,jj])
                        CoupledFraction[i*Timelength:(i+1)*Timelength,j] = tmp[:,0]
                    if j == 1:
                        tmp = np.zeros(shape=(Timelength,dim))
                        for jj in range(0,Nnodes):
                            tmp[:,0] = tmp[:,0]+A[i,jj]*1/(TimeSeries[:,i]*TimeSeries[:,jj])
                        CoupledFraction[i*Timelength:(i+1)*Timelength,j] = tmp[:,0]
                    if j == 2:
                        tmp = np.zeros(shape=(Timelength,dim))
                        for jj in range(0,Nnodes):
                            for tt in range(0,Timelength):
                                if TimeSeries[tt,jj]-TimeSeries[tt,i] !=0:
                                    tmp[tt,0] = tmp[tt,0]+A[i,jj]*1/(TimeSeries[tt,jj]-TimeSeries[tt,i])
                                else:
                                    tmp[tt,0] = tmp[tt,0]+np.zeros(shape=(1,1))
                        CoupledFraction[i*Timelength:(i+1)*Timelength,j] = tmp[:,0]
                    if j == 3:
                        tmp = np.zeros(shape=(Timelength,dim))
                        for jj in range(0,Nnodes):
                            tmp[:,0] = tmp[:,0]+A[i,jj]*1/(TimeSeries[:,jj])*TimeSeries[:,i]
                        CoupledFraction[i*Timelength:(i+1)*Timelength,j] = tmp[:,0]
            CoupledFraction = pd.DataFrame(data = CoupledFraction, columns = column_values)

        if dim == 2:
            column_values = [f'aggr(1/sour({var}))' for var in var_names] + [f'aggr(1/(targ({var})*sour({var})))' for var in var_names] + [f'aggr(1/(sour({var})-targ({var})))' for var in var_names] + [f'aggr(targ({var})/sour({var}))' for var in var_names] # ['fracx1j','fracx2j','fracx1ix1j','fracx2ix2j','fracx1jMinusx1i','fracx2jMinusx2i','x1ifracx1j','x2ifracx2j']
            CoupledFraction = np.zeros(shape=(Timelength*Nnodes,len(column_values)))
            for j in range(len(column_values)):
                for i in range(0,Nnodes):
                    if j == 0:
                        tmp = np.zeros(shape=(Timelength,dim))
                        for jj in range(0,Nnodes):
                            tmp[:,0] = tmp[:,0]+A[i,jj]*1/(TimeSeries[:,jj*dim])
                            tmp[:,1] = tmp[:,1]+A[i,jj]*1/(TimeSeries[:,jj*dim+1])
                        CoupledFraction[i*Timelength:(i+1)*Timelength,j] = tmp[:,0]
                        CoupledFraction[i*Timelength:(i+1)*Timelength,j+1] = tmp[:,1]
                    if j == 2:
                        tmp = np.zeros(shape=(Timelength,dim))
                        for jj in range(0,Nnodes):
                            tmp[:,0] = tmp[:,0]+A[i,jj]*1/(TimeSeries[:,i*dim]*TimeSeries[:,jj*dim])
                            tmp[:,1] = tmp[:,1]+A[i,jj]*1/(TimeSeries[:,i*dim+1]*TimeSeries[:,jj*dim+1])
                        CoupledFraction[i*Timelength:(i+1)*Timelength,j] = tmp[:,0]
                        CoupledFraction[i*Timelength:(i+1)*Timelength,j+1] = tmp[:,1]
                    if j == 4:
                        tmp = np.zeros(shape=(Timelength,dim))
                        for jj in range(0,Nnodes):
                            for tt in range(0,Timelength):
                                if TimeSeries[tt,jj*dim]-TimeSeries[tt,i*dim] !=0:
                                    tmp[tt,0] = tmp[tt,0]+A[i,jj]*1/(TimeSeries[tt,jj*dim]-TimeSeries[tt,i*dim])
                                else:
                                    tmp[tt,0] = tmp[tt,0]+np.zeros(shape=(1,1))
                                if TimeSeries[tt,jj*dim+1]-TimeSeries[tt,i*dim+1] !=0:
                                    tmp[tt,1] = tmp[tt,1]+A[i,jj]*1/(TimeSeries[tt,jj*dim+1]-TimeSeries[tt,i*dim+1])
                                else:
                                    tmp[tt,1] = tmp[tt,1]+np.zeros(shape=(1,1))
                        CoupledFraction[i*Timelength:(i+1)*Timelength,j] = tmp[:,0]
                        CoupledFraction[i*Timelength:(i+1)*Timelength,j+1] = tmp[:,1]
                    if j == 6:
                        tmp = np.zeros(shape=(Timelength,dim))
                        for jj in range(0,Nnodes):
                            tmp[:,0] = tmp[:,0]+A[i,jj]*1/(TimeSeries[:,jj*dim])*TimeSeries[:,i*dim]
                            tmp[:,1] = tmp[:,1]+A[i,jj]*1/(TimeSeries[:,jj*dim+1])*TimeSeries[:,i*dim+1]
                        CoupledFraction[i*Timelength:(i+1)*Timelength,j] = tmp[:,0]
                        CoupledFraction[i*Timelength:(i+1)*Timelength,j+1] = tmp[:,1]

            CoupledFraction = pd.DataFrame(data = CoupledFraction, columns = column_values)

        if dim == 3:
            column_values = [f'aggr(1/sour({var}))' for var in var_names] + [f'aggr(1/(targ({var})*sour({var})))' for var in var_names] + [f'aggr(1/(sour({var})-targ({var})))' for var in var_names] + [f'aggr(targ({var})/sour({var}))' for var in var_names] # ['fracx1j','fracx2j','fracx3j','fracx1ix1j','fracx2ix2j','fracx3ix3j','fracx1jMinusx1i','fracx2jMinusx2i','fracx3jMinusx3i','x1ifracx1j','x2ifracx2j','x3ifracx3j']
            CoupledFraction = np.zeros(shape=(Timelength*Nnodes,len(column_values)))
            for j in range(len(column_values)):
                for i in range(0,Nnodes):
                    if j == 0:
                        tmp = np.zeros(shape=(Timelength,dim))
                        for jj in range(0,Nnodes):
                            tmp[:,0] = tmp[:,0]+A[i,jj]*1/(TimeSeries[:,jj*dim])
                            tmp[:,1] = tmp[:,1]+A[i,jj]*1/(TimeSeries[:,jj*dim+1])
                            tmp[:,2] = tmp[:,2]+A[i,jj]*1/(TimeSeries[:,jj*dim+2])
                        CoupledFraction[i*Timelength:(i+1)*Timelength,j] = tmp[:,0]
                        CoupledFraction[i*Timelength:(i+1)*Timelength,j+1] = tmp[:,1]
                        CoupledFraction[i*Timelength:(i+1)*Timelength,j+2] = tmp[:,2]

                    if j == 3:
                        tmp = np.zeros(shape=(Timelength,dim))
                        for jj in range(0,Nnodes):
                            tmp[:,0] = tmp[:,0]+A[i,jj]*1/(TimeSeries[:,i*dim]*TimeSeries[:,jj*dim])
                            tmp[:,1] = tmp[:,1]+A[i,jj]*1/(TimeSeries[:,i*dim+1]*TimeSeries[:,jj*dim+1])
                            tmp[:,2] = tmp[:,2]+A[i,jj]*1/(TimeSeries[:,i*dim+2]*TimeSeries[:,jj*dim+2])
                        CoupledFraction[i*Timelength:(i+1)*Timelength,j] = tmp[:,0]
                        CoupledFraction[i*Timelength:(i+1)*Timelength,j+1] = tmp[:,1]
                        CoupledFraction[i*Timelength:(i+1)*Timelength,j+2] = tmp[:,2]
                    if j == 6:
                        tmp = np.zeros(shape=(Timelength,dim))
                        for jj in range(0,Nnodes):
                            for tt in range(0,Timelength):
                                if TimeSeries[tt,jj*dim]-TimeSeries[tt,i*dim] !=0:
                                    tmp[tt,0] = tmp[tt,0]+A[i,jj]*1/(TimeSeries[tt,jj*dim]-TimeSeries[tt,i*dim])
                                else:
                                    tmp[tt,0] = tmp[tt,0]+np.zeros(shape=(1,1))
                                if TimeSeries[tt,jj*dim+1]-TimeSeries[tt,i*dim+1] !=0:
                                    tmp[tt,1] = tmp[tt,1]+A[i,jj]*1/(TimeSeries[tt,jj*dim+1]-TimeSeries[tt,i*dim+1])
                                else:
                                    tmp[tt,1] = tmp[tt,1]+np.zeros(shape=(1,1))
                                if  TimeSeries[tt,jj*dim+2]-TimeSeries[tt,i*dim+2] !=0:
                                    tmp[tt,2] = tmp[tt,2]+A[i,jj]*1/(TimeSeries[tt,jj*dim+2]-TimeSeries[tt,i*dim+2])
                                else:
                                    tmp[tt,2] = tmp[tt,2]+np.zeros(shape=(1,1))
                        CoupledFraction[i*Timelength:(i+1)*Timelength,j] = tmp[:,0]
                        CoupledFraction[i*Timelength:(i+1)*Timelength,j+1] = tmp[:,1]
                        CoupledFraction[i*Timelength:(i+1)*Timelength,j+2] = tmp[:,2]
                    if j == 9:
                        tmp = np.zeros(shape=(Timelength,dim))
                        for jj in range(0,Nnodes):
                            tmp[:,0] = tmp[:,0]+A[i,jj]*1/(TimeSeries[:,jj*dim])*TimeSeries[:,i*dim]
                            tmp[:,1] = tmp[:,1]+A[i,jj]*1/(TimeSeries[:,jj*dim+1])*TimeSeries[:,i*dim+1]
                            tmp[:,2] = tmp[:,2]+A[i,jj]*1/(TimeSeries[:,jj*dim+2])*TimeSeries[:,i*dim+2]
                        CoupledFraction[i*Timelength:(i+1)*Timelength,j] = tmp[:,0]
                        CoupledFraction[i*Timelength:(i+1)*Timelength,j+1] = tmp[:,1]
                        CoupledFraction[i*Timelength:(i+1)*Timelength,j+2] = tmp[:,2]

            CoupledFraction = pd.DataFrame(data = CoupledFraction, columns = column_values)

        if dim == 4:
            column_values = [f'aggr(1/sour({var}))' for var in var_names] + [f'aggr(1/(targ({var})*sour({var})))' for var in var_names] + [f'aggr(1/(sour({var})-targ({var})))' for var in var_names] + [f'aggr(targ({var})/sour({var}))' for var in var_names] # ['fracx1j','fracx2j','fracx3j','fracx4j','fracx1ix1j','fracx2ix2j','fracx3ix3j','fracx4ix4j','fracx1jMinusx1i','fracx2jMinusx2i','fracx3jMinusx3i','fracx4jMinusx4i','x1ifracx1j','x2ifracx2j','x3ifracx3j','x4ifracx4j']
            CoupledFraction = np.zeros(shape=(Timelength*Nnodes,len(column_values)))
            for j in range(len(column_values)):
                for i in range(0,Nnodes):
                    if j == 0:
                        tmp = np.zeros(shape=(Timelength,dim))
                        for jj in range(0,Nnodes):
                            tmp[:,0] = tmp[:,0]+A[i,jj]*1/(TimeSeries[:,jj*dim])
                            tmp[:,1] = tmp[:,1]+A[i,jj]*1/(TimeSeries[:,jj*dim+1])
                            tmp[:,2] = tmp[:,2]+A[i,jj]*1/(TimeSeries[:,jj*dim+2])
                            tmp[:,3] = tmp[:,3]+A[i,jj]*1/(TimeSeries[:,jj*dim+3])
                        CoupledFraction[i*Timelength:(i+1)*Timelength,j] = tmp[:,0]
                        CoupledFraction[i*Timelength:(i+1)*Timelength,j+1] = tmp[:,1]
                        CoupledFraction[i*Timelength:(i+1)*Timelength,j+2] = tmp[:,2]
                        CoupledFraction[i*Timelength:(i+1)*Timelength,j+3] = tmp[:,3]
                    if j == 4:
                        tmp = np.zeros(shape=(Timelength,dim))
                        for jj in range(0,Nnodes):
                            tmp[:,0] = tmp[:,0]+A[i,jj]*1/(TimeSeries[:,i*dim]*TimeSeries[:,jj*dim])
                            tmp[:,1] = tmp[:,1]+A[i,jj]*1/(TimeSeries[:,i*dim+1]*TimeSeries[:,jj*dim+1])
                            tmp[:,2] = tmp[:,2]+A[i,jj]*1/(TimeSeries[:,i*dim+2]*TimeSeries[:,jj*dim+2])
                            tmp[:,3] = tmp[:,3]+A[i,jj]*1/(TimeSeries[:,i*dim+3]*TimeSeries[:,jj*dim+3])
                        CoupledFraction[i*Timelength:(i+1)*Timelength,j] = tmp[:,0]
                        CoupledFraction[i*Timelength:(i+1)*Timelength,j+1] = tmp[:,1]
                        CoupledFraction[i*Timelength:(i+1)*Timelength,j+2] = tmp[:,2]
                        CoupledFraction[i*Timelength:(i+1)*Timelength,j+3] = tmp[:,3]
                    if j == 8:
                        tmp = np.zeros(shape=(Timelength,dim))
                        for jj in range(0,Nnodes):
                            for tt in range(0,Timelength):
                                if TimeSeries[tt,jj*dim]-TimeSeries[tt,i*dim] !=0:
                                    tmp[tt,0] = tmp[tt,0]+A[i,jj]*1/(TimeSeries[tt,jj*dim]-TimeSeries[tt,i*dim])
                                else:
                                    tmp[tt,0] = tmp[tt,0]+np.zeros(shape=(1,1))
                                if TimeSeries[tt,jj*dim+1]-TimeSeries[tt,i*dim+1] !=0:
                                    tmp[tt,1] = tmp[tt,1]+A[i,jj]*1/(TimeSeries[tt,jj*dim+1]-TimeSeries[tt,i*dim+1])
                                else:
                                    tmp[tt,1] = tmp[tt,1]+np.zeros(shape=(1,1))
                                if  TimeSeries[tt,jj*dim+2]-TimeSeries[tt,i*dim+2] !=0:
                                    tmp[tt,2] = tmp[tt,2]+A[i,jj]*1/(TimeSeries[tt,jj*dim+2]-TimeSeries[tt,i*dim+2])
                                else:
                                    tmp[tt,2] = tmp[tt,2]+np.zeros(shape=(1,1))
                                if TimeSeries[tt,jj*dim+3]-TimeSeries[tt,i*dim+3] !=0:
                                    tmp[tt,3] = tmp[tt,3]+A[i,jj]*1/(TimeSeries[tt,jj*dim+3]-TimeSeries[tt,i*dim+3])
                                else:
                                    tmp[tt,3] = tmp[tt,3]+np.zeros(shape=(1,1))
                        CoupledFraction[i*Timelength:(i+1)*Timelength,j] = tmp[:,0]
                        CoupledFraction[i*Timelength:(i+1)*Timelength,j+1] = tmp[:,1]
                        CoupledFraction[i*Timelength:(i+1)*Timelength,j+2] = tmp[:,2]
                        CoupledFraction[i*Timelength:(i+1)*Timelength,j+3] = tmp[:,3]
                    if j == 12:
                        tmp = np.zeros(shape=(Timelength,dim))
                        for jj in range(0,Nnodes):
                            tmp[:,0] = tmp[:,0]+A[i,jj]*1/(TimeSeries[:,jj*dim])*TimeSeries[:,i*dim]
                            tmp[:,1] = tmp[:,1]+A[i,jj]*1/(TimeSeries[:,jj*dim+1])*TimeSeries[:,i*dim+1]
                            tmp[:,2] = tmp[:,2]+A[i,jj]*1/(TimeSeries[:,jj*dim+2])*TimeSeries[:,i*dim+2]
                            tmp[:,3] = tmp[:,3]+A[i,jj]*1/(TimeSeries[:,jj*dim+3])*TimeSeries[:,i*dim+3]
                        CoupledFraction[i*Timelength:(i+1)*Timelength,j] = tmp[:,0]
                        CoupledFraction[i*Timelength:(i+1)*Timelength,j+1] = tmp[:,1]
                        CoupledFraction[i*Timelength:(i+1)*Timelength,j+2] = tmp[:,2]
                        CoupledFraction[i*Timelength:(i+1)*Timelength,j+3] = tmp[:,3]
                    
            CoupledFraction = pd.DataFrame(data = CoupledFraction, columns = column_values)
        if dim > 4:
            raise ValueError('dim must be less than 5.')

    return CoupledFraction

def Coupled_Activation_functions(TimeSeries, var_names, Nnodes, A, Sigmoid = True, Tanh = True, Regulation = True, alpha=[10], beta=[1], gamma=[1,2,5]):
    Timelength = np.size(TimeSeries, 0)
    dim_multi_Nnodes = np.size(TimeSeries, 1)
    dim = len(var_names)
    # alpha = [10]
    # beta = [1]
    # gamma = [1,2,5]
    Numfunc = len(alpha)*len(beta)

    CoupledActivation = pd.DataFrame()


    if Sigmoid == True:
        if dim == 1:
            variables = [f'aggr(sigmoid(sour({var})*({{alpha}})+{{beta}}))' for var in var_names] + [f'aggr(sigmoid(targ({var})*sour({var})*({{alpha}})+{{beta}}))' for var in var_names] + [f'aggr(sigmoid((sour({var})-targ({var}))*({{alpha}})+{{beta}}))' for var in var_names] + [f'aggr(targ({var})*sigmoid(sour({var})*({{alpha}})+{{beta}}))' for var in var_names] # ['sigx1j','sigx1ix1j','sigx1jMinusx1i','x1isigx1j']
            
            
            for i in range(4):
                for ii in range(len(alpha)):
                    for jj in range(len(beta)):
                        if i == 0:
                            for kk in range(dim):
                                TimeSeriesDim = TimeSeries[:,kk::dim]
                                TimeSeriesDimxi = np.reshape(TimeSeriesDim,(-1,1),order='F')
                                TimeSeriesDimxi = np.tile(TimeSeriesDimxi,(1,Nnodes))
                                TimeSeriesDimxj = np.tile(TimeSeriesDim,(Nnodes,1))
                                A_Expand = np.repeat(A,TimeSeries.shape[0],axis=0)
                                CoupledSigmoid = np.sum(sigmoidfun(TimeSeriesDimxj,alpha[ii],beta[jj])*A_Expand,axis=1,keepdims=True)
                                tmp = pd.DataFrame(data=CoupledSigmoid,columns=[variables[i*dim+kk].format(alpha=str(alpha[ii]),beta=str(beta[jj]))])
                                CoupledActivation = pd.concat([CoupledActivation,tmp],axis=1)
                        elif i == 1:
                            for kk in range(dim):
                                TimeSeriesDim = TimeSeries[:,kk::dim]
                                TimeSeriesDimxi = np.reshape(TimeSeriesDim,(-1,1),order='F')
                                TimeSeriesDimxi = np.tile(TimeSeriesDimxi,(1,Nnodes))
                                TimeSeriesDimxj = np.tile(TimeSeriesDim,(Nnodes,1))
                                A_Expand = np.repeat(A,TimeSeries.shape[0],axis=0)
                                CoupledSigmoid = np.sum(sigmoidfun(TimeSeriesDimxi*TimeSeriesDimxj,alpha[ii],beta[jj])*A_Expand,axis=1,keepdims=True)
                                tmp = pd.DataFrame(data=CoupledSigmoid,columns=[variables[i*dim+kk].format(alpha=str(alpha[ii]),beta=str(beta[jj]))])
                                CoupledActivation = pd.concat([CoupledActivation,tmp],axis=1)
                        elif i == 2:
                            for kk in range(dim):
                                TimeSeriesDim = TimeSeries[:,kk::dim]
                                TimeSeriesDimxi = np.reshape(TimeSeriesDim,(-1,1),order='F')
                                TimeSeriesDimxi = np.tile(TimeSeriesDimxi,(1,Nnodes))
                                TimeSeriesDimxj = np.tile(TimeSeriesDim,(Nnodes,1))
                                A_Expand = np.repeat(A,TimeSeries.shape[0],axis=0)
                                CoupledSigmoid = np.sum(sigmoidfun(TimeSeriesDimxj-TimeSeriesDimxi,alpha[ii],beta[jj])*A_Expand,axis=1,keepdims=True)
                                tmp = pd.DataFrame(data=CoupledSigmoid,columns=[variables[i*dim+kk].format(alpha=str(alpha[ii]),beta=str(beta[jj]))])
                                CoupledActivation = pd.concat([CoupledActivation,tmp],axis=1)
                        elif i == 3:
                            for kk in range(dim):
                                TimeSeriesDim = TimeSeries[:,kk::dim]
                                TimeSeriesDimxi = np.reshape(TimeSeriesDim,(-1,1),order='F')
                                TimeSeriesDimxi = np.tile(TimeSeriesDimxi,(1,Nnodes))
                                TimeSeriesDimxj = np.tile(TimeSeriesDim,(Nnodes,1))
                                A_Expand = np.repeat(A,TimeSeries.shape[0],axis=0)
                                CoupledSigmoid = np.sum(TimeSeriesDimxi*sigmoidfun(TimeSeriesDimxj,alpha[ii],beta[jj])*A_Expand,axis=1,keepdims=True)
                                tmp = pd.DataFrame(data=CoupledSigmoid,columns=[variables[i*dim+kk].format(alpha=str(alpha[ii]),beta=str(beta[jj]))])
                                CoupledActivation = pd.concat([CoupledActivation,tmp],axis=1)
    
        if dim == 2:
                variables = [f'aggr(sigmoid(sour({var})*({{alpha}})+{{beta}}))' for var in var_names] + \
                            [f'aggr(sigmoid(targ({var})*sour({var})*({{alpha}})+{{beta}}))' for var in var_names] + \
                            [f'aggr(sigmoid((sour({var})-targ({var}))*({{alpha}})+{{beta}}))' for var in var_names] + \
                            [f'aggr(targ({var})*sigmoid(sour({var})*({{alpha}})+{{beta}}))' for var in var_names] 
                # ['sigx1j','sigx2j','sigx1ix1j','sigx2ix2j','sigx1jMinusx1i','sigx2jMinusx2i','x1isigx1j','x2isigx2j']
                
                
                for i in range(4):
                    for ii in range(len(alpha)):
                        for jj in range(len(beta)):
                            if i == 0:
                                for kk in range(dim):
                                    TimeSeriesDim = TimeSeries[:,kk::dim]
                                    TimeSeriesDimxi = np.reshape(TimeSeriesDim,(-1,1),order='F')
                                    TimeSeriesDimxi = np.tile(TimeSeriesDimxi,(1,Nnodes))
                                    TimeSeriesDimxj = np.tile(TimeSeriesDim,(Nnodes,1))
                                    A_Expand = np.repeat(A,TimeSeries.shape[0],axis=0)
                                    CoupledSigmoid = np.sum(sigmoidfun(TimeSeriesDimxj,alpha[ii],beta[jj])*A_Expand,axis=1,keepdims=True)
                                    tmp = pd.DataFrame(data=CoupledSigmoid,columns=[variables[i*dim+kk].format(alpha=str(alpha[ii]),beta=str(beta[jj]))])
                                    CoupledActivation = pd.concat([CoupledActivation,tmp],axis=1)
                            elif i == 1:
                                for kk in range(dim):
                                    TimeSeriesDim = TimeSeries[:,kk::dim]
                                    TimeSeriesDimxi = np.reshape(TimeSeriesDim,(-1,1),order='F')
                                    TimeSeriesDimxi = np.tile(TimeSeriesDimxi,(1,Nnodes))
                                    TimeSeriesDimxj = np.tile(TimeSeriesDim,(Nnodes,1))
                                    A_Expand = np.repeat(A,TimeSeries.shape[0],axis=0)
                                    CoupledSigmoid = np.sum(sigmoidfun(TimeSeriesDimxi*TimeSeriesDimxj,alpha[ii],beta[jj])*A_Expand,axis=1,keepdims=True)
                                    tmp = pd.DataFrame(data=CoupledSigmoid,columns=[variables[i*dim+kk].format(alpha=str(alpha[ii]),beta=str(beta[jj]))])
                                    CoupledActivation = pd.concat([CoupledActivation,tmp],axis=1)
                            elif i == 2:
                                for kk in range(dim):
                                    TimeSeriesDim = TimeSeries[:,kk::dim]
                                    TimeSeriesDimxi = np.reshape(TimeSeriesDim,(-1,1),order='F')
                                    TimeSeriesDimxi = np.tile(TimeSeriesDimxi,(1,Nnodes))
                                    TimeSeriesDimxj = np.tile(TimeSeriesDim,(Nnodes,1))
                                    A_Expand = np.repeat(A,TimeSeries.shape[0],axis=0)
                                    CoupledSigmoid = np.sum(sigmoidfun(TimeSeriesDimxj-TimeSeriesDimxi,alpha[ii],beta[jj])*A_Expand,axis=1,keepdims=True)
                                    tmp = pd.DataFrame(data=CoupledSigmoid,columns=[variables[i*dim+kk].format(alpha=str(alpha[ii]),beta=str(beta[jj]))])
                                    CoupledActivation = pd.concat([CoupledActivation,tmp],axis=1)
                            elif i == 3:
                                for kk in range(dim):
                                    TimeSeriesDim = TimeSeries[:,kk::dim]
                                    TimeSeriesDimxi = np.reshape(TimeSeriesDim,(-1,1),order='F')
                                    TimeSeriesDimxi = np.tile(TimeSeriesDimxi,(1,Nnodes))
                                    TimeSeriesDimxj = np.tile(TimeSeriesDim,(Nnodes,1))
                                    A_Expand = np.repeat(A,TimeSeries.shape[0],axis=0)
                                    CoupledSigmoid = np.sum(TimeSeriesDimxi*sigmoidfun(TimeSeriesDimxj,alpha[ii],beta[jj])*A_Expand,axis=1,keepdims=True)
                                    tmp = pd.DataFrame(data=CoupledSigmoid,columns=[variables[i*dim+kk].format(alpha=str(alpha[ii]),beta=str(beta[jj]))])
                                    CoupledActivation = pd.concat([CoupledActivation,tmp],axis=1)

        if dim == 3:
                variables = [f'aggr(sigmoid(sour({var})*({{alpha}})+{{beta}}))' for var in var_names] + [f'aggr(sigmoid(targ({var})*sour({var})*({{alpha}})+{{beta}}))' for var in var_names] + [f'aggr(sigmoid((sour({var})-targ({var}))*({{alpha}})+{{beta}}))' for var in var_names] + [f'aggr(targ({var})*sigmoid(sour({var})*({{alpha}})+{{beta}}))' for var in var_names] # ['sigx1j','sigx2j','sigx3j','sigx1ix1j','sigx2ix2j','sigx3ix3j','sigx1jMinusx1i','sigx2jMinusx2i','sigx3jMinusx3i','x1isigx1j','x2isigx2j','x3isigx3j']
                
                for i in range(4):
                    for ii in range(len(alpha)):
                        for jj in range(len(beta)):
                            if i == 0:
                                for kk in range(dim):
                                    TimeSeriesDim = TimeSeries[:,kk::dim]
                                    TimeSeriesDimxi = np.reshape(TimeSeriesDim,(-1,1),order='F')
                                    TimeSeriesDimxi = np.tile(TimeSeriesDimxi,(1,Nnodes))
                                    TimeSeriesDimxj = np.tile(TimeSeriesDim,(Nnodes,1))
                                    A_Expand = np.repeat(A,TimeSeries.shape[0],axis=0)
                                    CoupledSigmoid = np.sum(sigmoidfun(TimeSeriesDimxj,alpha[ii],beta[jj])*A_Expand,axis=1,keepdims=True)
                                    tmp = pd.DataFrame(data=CoupledSigmoid,columns=[variables[i*dim+kk].format(alpha=str(alpha[ii]),beta=str(beta[jj]))])
                                    CoupledActivation = pd.concat([CoupledActivation,tmp],axis=1)
                            elif i == 1:
                                for kk in range(dim):
                                    TimeSeriesDim = TimeSeries[:,kk::dim]
                                    TimeSeriesDimxi = np.reshape(TimeSeriesDim,(-1,1),order='F')
                                    TimeSeriesDimxi = np.tile(TimeSeriesDimxi,(1,Nnodes))
                                    TimeSeriesDimxj = np.tile(TimeSeriesDim,(Nnodes,1))
                                    A_Expand = np.repeat(A,TimeSeries.shape[0],axis=0)
                                    CoupledSigmoid = np.sum(sigmoidfun(TimeSeriesDimxi*TimeSeriesDimxj,alpha[ii],beta[jj])*A_Expand,axis=1,keepdims=True)
                                    tmp = pd.DataFrame(data=CoupledSigmoid,columns=[variables[i*dim+kk].format(alpha=str(alpha[ii]),beta=str(beta[jj]))])
                                    CoupledActivation = pd.concat([CoupledActivation,tmp],axis=1)
                            elif i == 2:
                                for kk in range(dim):
                                    TimeSeriesDim = TimeSeries[:,kk::dim]
                                    TimeSeriesDimxi = np.reshape(TimeSeriesDim,(-1,1),order='F')
                                    TimeSeriesDimxi = np.tile(TimeSeriesDimxi,(1,Nnodes))
                                    TimeSeriesDimxj = np.tile(TimeSeriesDim,(Nnodes,1))
                                    A_Expand = np.repeat(A,TimeSeries.shape[0],axis=0)
                                    CoupledSigmoid = np.sum(sigmoidfun(TimeSeriesDimxj-TimeSeriesDimxi,alpha[ii],beta[jj])*A_Expand,axis=1,keepdims=True)
                                    tmp = pd.DataFrame(data=CoupledSigmoid,columns=[variables[i*dim+kk].format(alpha=str(alpha[ii]),beta=str(beta[jj]))])
                                    CoupledActivation = pd.concat([CoupledActivation,tmp],axis=1)
                            elif i == 3:
                                for kk in range(dim):
                                    TimeSeriesDim = TimeSeries[:,kk::dim]
                                    TimeSeriesDimxi = np.reshape(TimeSeriesDim,(-1,1),order='F')
                                    TimeSeriesDimxi = np.tile(TimeSeriesDimxi,(1,Nnodes))
                                    TimeSeriesDimxj = np.tile(TimeSeriesDim,(Nnodes,1))
                                    A_Expand = np.repeat(A,TimeSeries.shape[0],axis=0)
                                    CoupledSigmoid = np.sum(TimeSeriesDimxi*sigmoidfun(TimeSeriesDimxj,alpha[ii],beta[jj])*A_Expand,axis=1,keepdims=True)
                                    tmp = pd.DataFrame(data=CoupledSigmoid,columns=[variables[i*dim+kk].format(alpha=str(alpha[ii]),beta=str(beta[jj]))])
                                    CoupledActivation = pd.concat([CoupledActivation,tmp],axis=1)

        if dim == 4:
                variables = [f'aggr(sigmoid(sour({var})*({{alpha}})+{{beta}}))' for var in var_names] + [f'aggr(sigmoid(targ({var})*sour({var})*({{alpha}})+{{beta}}))' for var in var_names] + [f'aggr(sigmoid((sour({var})-targ({var}))*({{alpha}})+{{beta}}))' for var in var_names] + [f'aggr(targ({var})*sigmoid(sour({var})*({{alpha}})+{{beta}}))' for var in var_names] # ['sigx1j','sigx2j','sigx3j','sigx4j','sigx1ix1j','sigx2ix2j','sigx3ix3j','sigx4ix4j','sigx1jMinusx1i','sigx2jMinusx2i','sigx3jMinusx3i','sigx4jMinusx4i','x1isigx1j','x2isigx2j','x3isigx3j','x4isigx4j']
                
                
                for i in range(4):
                    for ii in range(len(alpha)):
                        for jj in range(len(beta)):
                            if i == 0:
                                for kk in range(dim):
                                    TimeSeriesDim = TimeSeries[:,kk::dim]
                                    TimeSeriesDimxi = np.reshape(TimeSeriesDim,(-1,1),order='F')
                                    TimeSeriesDimxi = np.tile(TimeSeriesDimxi,(1,Nnodes))
                                    TimeSeriesDimxj = np.tile(TimeSeriesDim,(Nnodes,1))
                                    A_Expand = np.repeat(A,TimeSeries.shape[0],axis=0)
                                    CoupledSigmoid = np.sum(sigmoidfun(TimeSeriesDimxj,alpha[ii],beta[jj])*A_Expand,axis=1,keepdims=True)
                                    tmp = pd.DataFrame(data=CoupledSigmoid,columns=[variables[i*dim+kk].format(alpha=str(alpha[ii]),beta=str(beta[jj]))])
                                    CoupledActivation = pd.concat([CoupledActivation,tmp],axis=1)
                            elif i == 1:
                                for kk in range(dim):
                                    TimeSeriesDim = TimeSeries[:,kk::dim]
                                    TimeSeriesDimxi = np.reshape(TimeSeriesDim,(-1,1),order='F')
                                    TimeSeriesDimxi = np.tile(TimeSeriesDimxi,(1,Nnodes))
                                    TimeSeriesDimxj = np.tile(TimeSeriesDim,(Nnodes,1))
                                    A_Expand = np.repeat(A,TimeSeries.shape[0],axis=0)
                                    CoupledSigmoid = np.sum(sigmoidfun(TimeSeriesDimxi*TimeSeriesDimxj,alpha[ii],beta[jj])*A_Expand,axis=1,keepdims=True)
                                    tmp = pd.DataFrame(data=CoupledSigmoid,columns=[variables[i*dim+kk].format(alpha=str(alpha[ii]),beta=str(beta[jj]))])
                                    CoupledActivation = pd.concat([CoupledActivation,tmp],axis=1)
                            elif i == 2:
                                for kk in range(dim):
                                    TimeSeriesDim = TimeSeries[:,kk::dim]
                                    TimeSeriesDimxi = np.reshape(TimeSeriesDim,(-1,1),order='F')
                                    TimeSeriesDimxi = np.tile(TimeSeriesDimxi,(1,Nnodes))
                                    TimeSeriesDimxj = np.tile(TimeSeriesDim,(Nnodes,1))
                                    A_Expand = np.repeat(A,TimeSeries.shape[0],axis=0)
                                    CoupledSigmoid = np.sum(sigmoidfun(TimeSeriesDimxj-TimeSeriesDimxi,alpha[ii],beta[jj])*A_Expand,axis=1,keepdims=True)
                                    tmp = pd.DataFrame(data=CoupledSigmoid,columns=[variables[i*dim+kk].format(alpha=str(alpha[ii]),beta=str(beta[jj]))])
                                    CoupledActivation = pd.concat([CoupledActivation,tmp],axis=1)
                            elif i == 3:
                                for kk in range(dim):
                                    TimeSeriesDim = TimeSeries[:,kk::dim]
                                    TimeSeriesDimxi = np.reshape(TimeSeriesDim,(-1,1),order='F')
                                    TimeSeriesDimxi = np.tile(TimeSeriesDimxi,(1,Nnodes))
                                    TimeSeriesDimxj = np.tile(TimeSeriesDim,(Nnodes,1))
                                    A_Expand = np.repeat(A,TimeSeries.shape[0],axis=0)
                                    CoupledSigmoid = np.sum(TimeSeriesDimxi*sigmoidfun(TimeSeriesDimxj,alpha[ii],beta[jj])*A_Expand,axis=1,keepdims=True)
                                    tmp = pd.DataFrame(data=CoupledSigmoid,columns=[variables[i*dim+kk].format(alpha=str(alpha[ii]),beta=str(beta[jj]))])
                                    CoupledActivation = pd.concat([CoupledActivation,tmp],axis=1)
        if dim > 4:
            raise ValueError('dim must be less than 5.')
    if Tanh == True:
        if dim == 1:
            column_values = [f'aggr(tanh(sour({var})))' for var in var_names] + [f'aggr(tanh(targ({var})*sour({var})))' for var in var_names] + [f'aggr(tanh(sour({var})-targ({var})))' for var in var_names] + [f'aggr(targ({var})*tanh(sour({var})))' for var in var_names] # ['tanhx1j','tanhx1ix1j','tanhx1jMinusx1i','x1itanhx1j']
            CoupledTanh = np.zeros(shape=(Timelength*Nnodes,len(column_values)))
            for j in range(len(column_values)):
                for i in range(0,Nnodes):
                    if j == 0:
                        tmp = np.zeros(shape=(Timelength,dim))
                        for jj in range(0,Nnodes):
                            tmp[:,0] = tmp[:,0]+A[i,jj]*tangentH(TimeSeries[:,jj])
                        CoupledTanh[i*Timelength:(i+1)*Timelength,j] = tmp[:,0]
                    if j == 1:
                        tmp = np.zeros(shape=(Timelength,dim))
                        for jj in range(0,Nnodes):
                            tmp[:,0] = tmp[:,0]+A[i,jj]*tangentH(TimeSeries[:,i]*TimeSeries[:,jj])
                        CoupledTanh[i*Timelength:(i+1)*Timelength,j] = tmp[:,0]
                    if j == 2:
                        tmp = np.zeros(shape=(Timelength,dim))
                        for jj in range(0,Nnodes):
                            tmp[:,0] = tmp[:,0]+A[i,jj]*tangentH((TimeSeries[:,jj]-TimeSeries[:,i]))
                        CoupledTanh[i*Timelength:(i+1)*Timelength,j] = tmp[:,0]
                    if j == 3:
                        tmp = np.zeros(shape=(Timelength,dim))
                        for jj in range(0,Nnodes):
                            tmp[:,0] = tmp[:,0]+A[i,jj]*tangentH(TimeSeries[:,jj])*TimeSeries[:,i]
                        CoupledTanh[i*Timelength:(i+1)*Timelength,j] = tmp[:,0]
            CoupledTanh = pd.DataFrame(data = CoupledTanh, columns = column_values)

        if dim == 2:
            column_values = [f'aggr(tanh(sour({var})))' for var in var_names] + [f'aggr(tanh(targ({var})*sour({var})))' for var in var_names] + [f'aggr(tanh(sour({var})-targ({var})))' for var in var_names] + [f'aggr(targ({var})*tanh(sour({var})))' for var in var_names] # ['tanhx1j','tanhx2j','tanhx1ix1j','tanhx2ix2j','tanhx1jMinusx1i','tanhx2jMinusx2i','x1itanhx1j','x2itanhx2j']
            CoupledTanh = np.zeros(shape=(Timelength*Nnodes,len(column_values)))
            for j in range(len(column_values)):
                for i in range(0,Nnodes):
                    if j == 0:
                        tmp = np.zeros(shape=(Timelength,dim))
                        for jj in range(0,Nnodes):
                            tmp[:,0] = tmp[:,0]+A[i,jj]*tangentH(TimeSeries[:,jj*dim])
                            tmp[:,1] = tmp[:,1]+A[i,jj]*tangentH(TimeSeries[:,jj*dim+1])
                        CoupledTanh[i*Timelength:(i+1)*Timelength,j] = tmp[:,0]
                        CoupledTanh[i*Timelength:(i+1)*Timelength,j+1] = tmp[:,1]
                    if j == 2:
                        tmp = np.zeros(shape=(Timelength,dim))
                        for jj in range(0,Nnodes):
                            tmp[:,0] = tmp[:,0]+A[i,jj]*tangentH(TimeSeries[:,i*dim]*TimeSeries[:,jj*dim])
                            tmp[:,1] = tmp[:,1]+A[i,jj]*tangentH(TimeSeries[:,i*dim+1]*TimeSeries[:,jj*dim+1])
                        CoupledTanh[i*Timelength:(i+1)*Timelength,j] = tmp[:,0]
                        CoupledTanh[i*Timelength:(i+1)*Timelength,j+1] = tmp[:,1]
                    if j == 4:
                        tmp = np.zeros(shape=(Timelength,dim))
                        for jj in range(0,Nnodes):
                            tmp[:,0] = tmp[:,0]+A[i,jj]*tangentH(TimeSeries[:,jj*dim]-TimeSeries[:,i*dim])
                            tmp[:,1] = tmp[:,1]+A[i,jj]*tangentH(TimeSeries[:,jj*dim+1]-TimeSeries[:,i*dim+1])
                        CoupledTanh[i*Timelength:(i+1)*Timelength,j] = tmp[:,0]
                        CoupledTanh[i*Timelength:(i+1)*Timelength,j+1] = tmp[:,1]
                    if j == 6:
                        tmp = np.zeros(shape=(Timelength,dim))
                        for jj in range(0,Nnodes):
                            tmp[:,0] = tmp[:,0]+A[i,jj]*tangentH(TimeSeries[:,jj*dim])*TimeSeries[:,i*dim]
                            tmp[:,1] = tmp[:,1]+A[i,jj]*tangentH(TimeSeries[:,jj*dim+1])*TimeSeries[:,i*dim+1]
                        CoupledTanh[i*Timelength:(i+1)*Timelength,j] = tmp[:,0]
                        CoupledTanh[i*Timelength:(i+1)*Timelength,j+1] = tmp[:,1]

            CoupledTanh = pd.DataFrame(data = CoupledTanh, columns = column_values)

        if dim == 3:
            column_values = [f'aggr(tanh(sour({var})))' for var in var_names] + [f'aggr(tanh(targ({var})*sour({var})))' for var in var_names] + [f'aggr(tanh(sour({var})-targ({var})))' for var in var_names] + [f'aggr(targ({var})*tanh(sour({var})))' for var in var_names] # ['tanhx1j','tanhx2j','tanhx3j','tanhx1ix1j','tanhx2ix2j','tanhx3ix3j','tanhx1jMinusx1i','tanhx2jMinusx2i','tanhx3jMinusx3i','x1itanhx1j','x2itanhx2j','x3itanhx3j']
            CoupledTanh = np.zeros(shape=(Timelength*Nnodes,len(column_values)))
            for j in range(len(column_values)):
                for i in range(0,Nnodes):
                    if j == 0:
                        tmp = np.zeros(shape=(Timelength,dim))
                        for jj in range(0,Nnodes):
                            tmp[:,0] = tmp[:,0]+A[i,jj]*tangentH(TimeSeries[:,jj*dim])
                            tmp[:,1] = tmp[:,1]+A[i,jj]*tangentH(TimeSeries[:,jj*dim+1])
                            tmp[:,2] = tmp[:,2]+A[i,jj]*tangentH(TimeSeries[:,jj*dim+2])
                        CoupledTanh[i*Timelength:(i+1)*Timelength,j] = tmp[:,0]
                        CoupledTanh[i*Timelength:(i+1)*Timelength,j+1] = tmp[:,1]
                        CoupledTanh[i*Timelength:(i+1)*Timelength,j+2] = tmp[:,2]

                    if j == 3:
                        tmp = np.zeros(shape=(Timelength,dim))
                        for jj in range(0,Nnodes):
                            tmp[:,0] = tmp[:,0]+A[i,jj]*tangentH(TimeSeries[:,i*dim]*TimeSeries[:,jj*dim])
                            tmp[:,1] = tmp[:,1]+A[i,jj]*tangentH(TimeSeries[:,i*dim+1]*TimeSeries[:,jj*dim+1])
                            tmp[:,2] = tmp[:,2]+A[i,jj]*tangentH(TimeSeries[:,i*dim+2]*TimeSeries[:,jj*dim+2])
                        CoupledTanh[i*Timelength:(i+1)*Timelength,j] = tmp[:,0]
                        CoupledTanh[i*Timelength:(i+1)*Timelength,j+1] = tmp[:,1]
                        CoupledTanh[i*Timelength:(i+1)*Timelength,j+2] = tmp[:,2]
                    if j == 6:
                        tmp = np.zeros(shape=(Timelength,dim))
                        for jj in range(0,Nnodes):
                            tmp[:,0] = tmp[:,0]+A[i,jj]*tangentH(TimeSeries[:,jj*dim]-TimeSeries[:,i*dim])
                            tmp[:,1] = tmp[:,1]+A[i,jj]*tangentH(TimeSeries[:,jj*dim+1]-TimeSeries[:,i*dim+1])
                            tmp[:,2] = tmp[:,2]+A[i,jj]*tangentH(TimeSeries[:,jj*dim+2]-TimeSeries[:,i*dim+2])
                        CoupledTanh[i*Timelength:(i+1)*Timelength,j] = tmp[:,0]
                        CoupledTanh[i*Timelength:(i+1)*Timelength,j+1] = tmp[:,1]
                        CoupledTanh[i*Timelength:(i+1)*Timelength,j+2] = tmp[:,2]
                    if j == 9:
                        tmp = np.zeros(shape=(Timelength,dim))
                        for jj in range(0,Nnodes):
                            tmp[:,0] = tmp[:,0]+A[i,jj]*tangentH(TimeSeries[:,jj*dim])*TimeSeries[:,i*dim]
                            tmp[:,1] = tmp[:,1]+A[i,jj]*tangentH(TimeSeries[:,jj*dim+1])*TimeSeries[:,i*dim+1]
                            tmp[:,2] = tmp[:,2]+A[i,jj]*tangentH(TimeSeries[:,jj*dim+2])*TimeSeries[:,i*dim+2]
                        CoupledTanh[i*Timelength:(i+1)*Timelength,j] = tmp[:,0]
                        CoupledTanh[i*Timelength:(i+1)*Timelength,j+1] = tmp[:,1]
                        CoupledTanh[i*Timelength:(i+1)*Timelength,j+2] = tmp[:,2]

            CoupledTanh = pd.DataFrame(data = CoupledTanh, columns = column_values)

        if dim == 4:
            column_values = [f'aggr(tanh(sour({var})))' for var in var_names] + [f'aggr(tanh(targ({var})*sour({var})))' for var in var_names] + [f'aggr(tanh(sour({var})-targ({var})))' for var in var_names] + [f'aggr(targ({var})*tanh(sour({var})))' for var in var_names] # ['tanhx1j','tanhx2j','tanhx3j','tanhx4j','tanhx1ix1j','tanhx2ix2j','tanhx3ix3j','tanhx4ix4j','tanhx1jMinusx1i','tanhx2jMinusx2i','tanhx3jMinusx3i','tanhx4jMinusx4i','x1itanhx1j','x2itanhx2j','x3itanhx3j','x4itanhx4j']
            CoupledTanh = np.zeros(shape=(Timelength*Nnodes,len(column_values)))
            for j in range(len(column_values)):
                for i in range(0,Nnodes):
                    if j == 0:
                        tmp = np.zeros(shape=(Timelength,dim))
                        for jj in range(0,Nnodes):
                            tmp[:,0] = tmp[:,0]+A[i,jj]*tangentH(TimeSeries[:,jj*dim])
                            tmp[:,1] = tmp[:,1]+A[i,jj]*tangentH(TimeSeries[:,jj*dim+1])
                            tmp[:,2] = tmp[:,2]+A[i,jj]*tangentH(TimeSeries[:,jj*dim+2])
                            tmp[:,3] = tmp[:,3]+A[i,jj]*tangentH(TimeSeries[:,jj*dim+3])
                        CoupledTanh[i*Timelength:(i+1)*Timelength,j] = tmp[:,0]
                        CoupledTanh[i*Timelength:(i+1)*Timelength,j+1] = tmp[:,1]
                        CoupledTanh[i*Timelength:(i+1)*Timelength,j+2] = tmp[:,2]
                        CoupledTanh[i*Timelength:(i+1)*Timelength,j+3] = tmp[:,3]
                    if j == 4:
                        tmp = np.zeros(shape=(Timelength,dim))
                        for jj in range(0,Nnodes):
                            tmp[:,0] = tmp[:,0]+A[i,jj]*tangentH(TimeSeries[:,i*dim]*TimeSeries[:,jj*dim])
                            tmp[:,1] = tmp[:,1]+A[i,jj]*tangentH(TimeSeries[:,i*dim+1]*TimeSeries[:,jj*dim+1])
                            tmp[:,2] = tmp[:,2]+A[i,jj]*tangentH(TimeSeries[:,i*dim+2]*TimeSeries[:,jj*dim+2])
                            tmp[:,3] = tmp[:,3]+A[i,jj]*tangentH(TimeSeries[:,i*dim+3]*TimeSeries[:,jj*dim+3])
                        CoupledTanh[i*Timelength:(i+1)*Timelength,j] = tmp[:,0]
                        CoupledTanh[i*Timelength:(i+1)*Timelength,j+1] = tmp[:,1]
                        CoupledTanh[i*Timelength:(i+1)*Timelength,j+2] = tmp[:,2]
                        CoupledTanh[i*Timelength:(i+1)*Timelength,j+3] = tmp[:,3]
                    if j == 8:
                        tmp = np.zeros(shape=(Timelength,dim))
                        for jj in range(0,Nnodes):
                            tmp[:,0] = tmp[:,0]+A[i,jj]*tangentH(TimeSeries[:,jj*dim]-TimeSeries[:,i*dim])
                            tmp[:,1] = tmp[:,1]+A[i,jj]*tangentH(TimeSeries[:,jj*dim+1]-TimeSeries[:,i*dim+1])
                            tmp[:,2] = tmp[:,2]+A[i,jj]*tangentH(TimeSeries[:,jj*dim+2]-TimeSeries[:,i*dim+2])
                            tmp[:,3] = tmp[:,3]+A[i,jj]*tangentH(TimeSeries[:,jj*dim+3]-TimeSeries[:,i*dim+3])
                        CoupledTanh[i*Timelength:(i+1)*Timelength,j] = tmp[:,0]
                        CoupledTanh[i*Timelength:(i+1)*Timelength,j+1] = tmp[:,1]
                        CoupledTanh[i*Timelength:(i+1)*Timelength,j+2] = tmp[:,2]
                        CoupledTanh[i*Timelength:(i+1)*Timelength,j+3] = tmp[:,3]
                    if j == 12:
                        tmp = np.zeros(shape=(Timelength,dim))
                        for jj in range(0,Nnodes):
                            tmp[:,0] = tmp[:,0]+A[i,jj]*tangentH(TimeSeries[:,jj*dim])*TimeSeries[:,i*dim]
                            tmp[:,1] = tmp[:,1]+A[i,jj]*tangentH(TimeSeries[:,jj*dim+1])*TimeSeries[:,i*dim+1]
                            tmp[:,2] = tmp[:,2]+A[i,jj]*tangentH(TimeSeries[:,jj*dim+2])*TimeSeries[:,i*dim+2]
                            tmp[:,3] = tmp[:,3]+A[i,jj]*tangentH(TimeSeries[:,jj*dim+3])*TimeSeries[:,i*dim+3]
                        CoupledTanh[i*Timelength:(i+1)*Timelength,j] = tmp[:,0]
                        CoupledTanh[i*Timelength:(i+1)*Timelength,j+1] = tmp[:,1]
                        CoupledTanh[i*Timelength:(i+1)*Timelength,j+2] = tmp[:,2]
                        CoupledTanh[i*Timelength:(i+1)*Timelength,j+3] = tmp[:,3]
                    
            CoupledTanh = pd.DataFrame(data = CoupledTanh, columns = column_values)
        
        if dim > 4:
            raise ValueError('dim must be less than 5.')

    if Regulation == True:
        if dim == 1:
            variables = [f'aggr(regular(sour({var}), {{gamma}}))' for var in var_names] + [f'aggr(regular(targ({var})*sour({var}), {{gamma}}))' for var in var_names] + [f'aggr(regular(sour({var})-targ({var}), {{gamma}}))' for var in var_names] + [f'aggr(targ({var})*regular(sour({var}), {{gamma}}))' for var in var_names] # ['regx1j','regx1ix1j','regx1jMinusx1i','x1iregx1j']
            
            
            for i in range(4):
                for ii in range(len(gamma)):
                    if i == 0:
                        for kk in range(dim):
                            TimeSeriesDim = TimeSeries[:,kk::dim]
                            TimeSeriesDimxi = np.reshape(TimeSeriesDim,(-1,1),order='F')
                            TimeSeriesDimxi = np.tile(TimeSeriesDimxi,(1,Nnodes))
                            TimeSeriesDimxj = np.tile(TimeSeriesDim,(Nnodes,1))
                            A_Expand = np.repeat(A,TimeSeries.shape[0],axis=0)
                            CoupledRegulation = np.sum(Regulation_func(TimeSeriesDimxj,gamma[ii])*A_Expand,axis=1,keepdims=True)
                            tmp = pd.DataFrame(data=CoupledRegulation,columns=[variables[i*dim+kk].format(gamma=str(gamma[ii]))])
                            CoupledActivation = pd.concat([CoupledActivation,tmp],axis=1)
                    elif i == 1:
                        for kk in range(dim):
                            TimeSeriesDim = TimeSeries[:,kk::dim]
                            TimeSeriesDimxi = np.reshape(TimeSeriesDim,(-1,1),order='F')
                            TimeSeriesDimxi = np.tile(TimeSeriesDimxi,(1,Nnodes))
                            TimeSeriesDimxj = np.tile(TimeSeriesDim,(Nnodes,1))
                            A_Expand = np.repeat(A,TimeSeries.shape[0],axis=0)
                            CoupledRegulation = np.sum(Regulation_func(TimeSeriesDimxi*TimeSeriesDimxj,gamma[ii])*A_Expand,axis=1,keepdims=True)
                            tmp = pd.DataFrame(data=CoupledRegulation,columns=[variables[i*dim+kk].format(gamma=str(gamma[ii]))])
                            CoupledActivation = pd.concat([CoupledActivation,tmp],axis=1)
                    elif i == 2:
                        for kk in range(dim):
                            TimeSeriesDim = TimeSeries[:,kk::dim]
                            TimeSeriesDimxi = np.reshape(TimeSeriesDim,(-1,1),order='F')
                            TimeSeriesDimxi = np.tile(TimeSeriesDimxi,(1,Nnodes))
                            TimeSeriesDimxj = np.tile(TimeSeriesDim,(Nnodes,1))
                            A_Expand = np.repeat(A,TimeSeries.shape[0],axis=0)
                            CoupledRegulation = np.sum(Regulation_func(TimeSeriesDimxj-TimeSeriesDimxi,gamma[ii])*A_Expand,axis=1,keepdims=True)
                            tmp = pd.DataFrame(data=CoupledRegulation,columns=[variables[i*dim+kk].format(gamma=str(gamma[ii]))])
                            CoupledActivation = pd.concat([CoupledActivation,tmp],axis=1)
                    elif i == 3:
                        for kk in range(dim):
                            TimeSeriesDim = TimeSeries[:,kk::dim]
                            TimeSeriesDimxi = np.reshape(TimeSeriesDim,(-1,1),order='F')
                            TimeSeriesDimxi = np.tile(TimeSeriesDimxi,(1,Nnodes))
                            TimeSeriesDimxj = np.tile(TimeSeriesDim,(Nnodes,1))
                            A_Expand = np.repeat(A,TimeSeries.shape[0],axis=0)
                            CoupledRegulation = np.sum(TimeSeriesDimxi*Regulation_func(TimeSeriesDimxj,gamma[ii])*A_Expand,axis=1,keepdims=True)
                            tmp = pd.DataFrame(data=CoupledRegulation,columns=[variables[i*dim+kk].format(gamma=str(gamma[ii]))])
                            CoupledActivation = pd.concat([CoupledActivation,tmp],axis=1)
    
        if dim == 2:
                variables = [f'aggr(regular(sour({var}), {{gamma}}))' for var in var_names] + [f'aggr(regular(targ({var})*sour({var}), {{gamma}}))' for var in var_names] + [f'aggr(regular(sour({var})-targ({var}), {{gamma}}))' for var in var_names] + [f'aggr(targ({var})*regular(sour({var}), {{gamma}}))' for var in var_names] # ['regx1j','regx2j','regx1ix1j','regx2ix2j','regx1jMinusx1i','regx2jMinusx2i','x1iregx1j','x2iregx2j']
                
                
                for i in range(4):
                    for ii in range(len(gamma)):
                        if i == 0:
                            for kk in range(dim):
                                TimeSeriesDim = TimeSeries[:,kk::dim]
                                TimeSeriesDimxi = np.reshape(TimeSeriesDim,(-1,1),order='F')
                                TimeSeriesDimxi = np.tile(TimeSeriesDimxi,(1,Nnodes))
                                TimeSeriesDimxj = np.tile(TimeSeriesDim,(Nnodes,1))
                                A_Expand = np.repeat(A,TimeSeries.shape[0],axis=0)
                                CoupledRegulation = np.sum(Regulation_func(TimeSeriesDimxj,gamma[ii])*A_Expand,axis=1,keepdims=True)
                                tmp = pd.DataFrame(data=CoupledRegulation,columns=[variables[i*dim+kk].format(gamma=str(gamma[ii]))])
                                CoupledActivation = pd.concat([CoupledActivation,tmp],axis=1)
                        elif i == 1:
                            for kk in range(dim):
                                TimeSeriesDim = TimeSeries[:,kk::dim]
                                TimeSeriesDimxi = np.reshape(TimeSeriesDim,(-1,1),order='F')
                                TimeSeriesDimxi = np.tile(TimeSeriesDimxi,(1,Nnodes))
                                TimeSeriesDimxj = np.tile(TimeSeriesDim,(Nnodes,1))
                                A_Expand = np.repeat(A,TimeSeries.shape[0],axis=0)
                                CoupledRegulation = np.sum(Regulation_func(TimeSeriesDimxi*TimeSeriesDimxj,gamma[ii])*A_Expand,axis=1,keepdims=True)
                                tmp = pd.DataFrame(data=CoupledRegulation,columns=[variables[i*dim+kk].format(gamma=str(gamma[ii]))])
                                CoupledActivation = pd.concat([CoupledActivation,tmp],axis=1)
                        elif i == 2:
                            for kk in range(dim):
                                TimeSeriesDim = TimeSeries[:,kk::dim]
                                TimeSeriesDimxi = np.reshape(TimeSeriesDim,(-1,1),order='F')
                                TimeSeriesDimxi = np.tile(TimeSeriesDimxi,(1,Nnodes))
                                TimeSeriesDimxj = np.tile(TimeSeriesDim,(Nnodes,1))
                                A_Expand = np.repeat(A,TimeSeries.shape[0],axis=0)
                                CoupledRegulation = np.sum(Regulation_func(TimeSeriesDimxj-TimeSeriesDimxi,gamma[ii])*A_Expand,axis=1,keepdims=True)
                                tmp = pd.DataFrame(data=CoupledRegulation,columns=[variables[i*dim+kk].format(gamma=str(gamma[ii]))])
                                CoupledActivation = pd.concat([CoupledActivation,tmp],axis=1)
                        elif i == 3:
                            for kk in range(dim):
                                TimeSeriesDim = TimeSeries[:,kk::dim]
                                TimeSeriesDimxi = np.reshape(TimeSeriesDim,(-1,1),order='F')
                                TimeSeriesDimxi = np.tile(TimeSeriesDimxi,(1,Nnodes))
                                TimeSeriesDimxj = np.tile(TimeSeriesDim,(Nnodes,1))
                                A_Expand = np.repeat(A,TimeSeries.shape[0],axis=0)
                                CoupledRegulation = np.sum(TimeSeriesDimxi*Regulation_func(TimeSeriesDimxj,gamma[ii])*A_Expand,axis=1,keepdims=True)
                                tmp = pd.DataFrame(data=CoupledRegulation,columns=[variables[i*dim+kk].format(gamma=str(gamma[ii]))])
                                CoupledActivation = pd.concat([CoupledActivation,tmp],axis=1)

        if dim == 3:
                variables = [f'aggr(regular(sour({var}), {{gamma}}))' for var in var_names] + [f'aggr(regular(targ({var})*sour({var}), {{gamma}}))' for var in var_names] + [f'aggr(regular(sour({var})-targ({var}), {{gamma}}))' for var in var_names] + [f'aggr(targ({var})*regular(sour({var}), {{gamma}}))' for var in var_names] # ['regx1j','regx2j','regx3j','regx1ix1j','regx2ix2j','regx3ix3j','regx1jMinusx1i','regx2jMinusx2i','regx3jMinusx3i','x1iregx1j','x2iregx2j','x3iregx3j']
                
                
                for i in range(4):
                    for ii in range(len(gamma)):
                        if i == 0:
                            for kk in range(dim):
                                TimeSeriesDim = TimeSeries[:,kk::dim]
                                TimeSeriesDimxi = np.reshape(TimeSeriesDim,(-1,1),order='F')
                                TimeSeriesDimxi = np.tile(TimeSeriesDimxi,(1,Nnodes))
                                TimeSeriesDimxj = np.tile(TimeSeriesDim,(Nnodes,1))
                                A_Expand = np.repeat(A,TimeSeries.shape[0],axis=0)
                                CoupledRegulation = np.sum(Regulation_func(TimeSeriesDimxj,gamma[ii])*A_Expand,axis=1,keepdims=True)
                                tmp = pd.DataFrame(data=CoupledRegulation,columns=[variables[i*dim+kk].format(gamma=str(gamma[ii]))])
                                CoupledActivation = pd.concat([CoupledActivation,tmp],axis=1)
                        elif i == 1:
                            for kk in range(dim):
                                TimeSeriesDim = TimeSeries[:,kk::dim]
                                TimeSeriesDimxi = np.reshape(TimeSeriesDim,(-1,1),order='F')
                                TimeSeriesDimxi = np.tile(TimeSeriesDimxi,(1,Nnodes))
                                TimeSeriesDimxj = np.tile(TimeSeriesDim,(Nnodes,1))
                                A_Expand = np.repeat(A,TimeSeries.shape[0],axis=0)
                                CoupledRegulation = np.sum(Regulation_func(TimeSeriesDimxi*TimeSeriesDimxj,gamma[ii])*A_Expand,axis=1,keepdims=True)
                                tmp = pd.DataFrame(data=CoupledRegulation,columns=[variables[i*dim+kk].format(gamma=str(gamma[ii]))])
                                CoupledActivation = pd.concat([CoupledActivation,tmp],axis=1)
                        elif i == 2:
                            for kk in range(dim):
                                TimeSeriesDim = TimeSeries[:,kk::dim]
                                TimeSeriesDimxi = np.reshape(TimeSeriesDim,(-1,1),order='F')
                                TimeSeriesDimxi = np.tile(TimeSeriesDimxi,(1,Nnodes))
                                TimeSeriesDimxj = np.tile(TimeSeriesDim,(Nnodes,1))
                                A_Expand = np.repeat(A,TimeSeries.shape[0],axis=0)
                                CoupledRegulation = np.sum(Regulation_func(TimeSeriesDimxj-TimeSeriesDimxi,gamma[ii])*A_Expand,axis=1,keepdims=True)
                                tmp = pd.DataFrame(data=CoupledRegulation,columns=[variables[i*dim+kk].format(gamma=str(gamma[ii]))])
                                CoupledActivation = pd.concat([CoupledActivation,tmp],axis=1)
                        elif i == 3:
                            for kk in range(dim):
                                TimeSeriesDim = TimeSeries[:,kk::dim]
                                TimeSeriesDimxi = np.reshape(TimeSeriesDim,(-1,1),order='F')
                                TimeSeriesDimxi = np.tile(TimeSeriesDimxi,(1,Nnodes))
                                TimeSeriesDimxj = np.tile(TimeSeriesDim,(Nnodes,1))
                                A_Expand = np.repeat(A,TimeSeries.shape[0],axis=0)
                                CoupledRegulation = np.sum(TimeSeriesDimxi*Regulation_func(TimeSeriesDimxj,gamma[ii])*A_Expand,axis=1,keepdims=True)
                                tmp = pd.DataFrame(data=CoupledRegulation,columns=[variables[i*dim+kk].format(gamma=str(gamma[ii]))])
                                CoupledActivation = pd.concat([CoupledActivation,tmp],axis=1)

        if dim == 4:
                variables = [f'aggr(regular(sour({var}), {{gamma}}))' for var in var_names] + [f'aggr(regular(targ({var})*sour({var}), {{gamma}}))' for var in var_names] + [f'aggr(regular(sour({var})-targ({var}), {{gamma}}))' for var in var_names] + [f'aggr(targ({var})*regular(sour({var}), {{gamma}}))' for var in var_names] # ['regx1j','regx2j','regx3j','regx4j','regx1ix1j','regx2ix2j','regx3ix3j','regx4ix4j','regx1jMinusx1i','regx2jMinusx2i','regx3jMinusx3i','regx4jMinusx4i','x1iregx1j','x2iregx2j','x3iregx3j','x4iregx4j']
                
                
                for i in range(4):
                    for ii in range(len(gamma)):
                        if i == 0:
                            for kk in range(dim):
                                TimeSeriesDim = TimeSeries[:,kk::dim]
                                TimeSeriesDimxi = np.reshape(TimeSeriesDim,(-1,1),order='F')
                                TimeSeriesDimxi = np.tile(TimeSeriesDimxi,(1,Nnodes))
                                TimeSeriesDimxj = np.tile(TimeSeriesDim,(Nnodes,1))
                                A_Expand = np.repeat(A,TimeSeries.shape[0],axis=0)
                                CoupledRegulation = np.sum(Regulation_func(TimeSeriesDimxj,gamma[ii])*A_Expand,axis=1,keepdims=True)
                                tmp = pd.DataFrame(data=CoupledRegulation,columns=[variables[i*dim+kk].format(gamma=str(gamma[ii]))])
                                CoupledActivation = pd.concat([CoupledActivation,tmp],axis=1)
                        elif i == 1:
                            for kk in range(dim):
                                TimeSeriesDim = TimeSeries[:,kk::dim]
                                TimeSeriesDimxi = np.reshape(TimeSeriesDim,(-1,1),order='F')
                                TimeSeriesDimxi = np.tile(TimeSeriesDimxi,(1,Nnodes))
                                TimeSeriesDimxj = np.tile(TimeSeriesDim,(Nnodes,1))
                                A_Expand = np.repeat(A,TimeSeries.shape[0],axis=0)
                                CoupledRegulation = np.sum(Regulation_func(TimeSeriesDimxi*TimeSeriesDimxj,gamma[ii])*A_Expand,axis=1,keepdims=True)
                                tmp = pd.DataFrame(data=CoupledRegulation,columns=[variables[i*dim+kk].format(gamma=str(gamma[ii]))])
                                CoupledActivation = pd.concat([CoupledActivation,tmp],axis=1)
                        elif i == 2:
                            for kk in range(dim):
                                TimeSeriesDim = TimeSeries[:,kk::dim]
                                TimeSeriesDimxi = np.reshape(TimeSeriesDim,(-1,1),order='F')
                                TimeSeriesDimxi = np.tile(TimeSeriesDimxi,(1,Nnodes))
                                TimeSeriesDimxj = np.tile(TimeSeriesDim,(Nnodes,1))
                                A_Expand = np.repeat(A,TimeSeries.shape[0],axis=0)
                                CoupledRegulation = np.sum(Regulation_func(TimeSeriesDimxj-TimeSeriesDimxi,gamma[ii])*A_Expand,axis=1,keepdims=True)
                                tmp = pd.DataFrame(data=CoupledRegulation,columns=[variables[i*dim+kk].format(gamma=str(gamma[ii]))])
                                CoupledActivation = pd.concat([CoupledActivation,tmp],axis=1)
                        elif i == 3:
                            for kk in range(dim):
                                TimeSeriesDim = TimeSeries[:,kk::dim]
                                TimeSeriesDimxi = np.reshape(TimeSeriesDim,(-1,1),order='F')
                                TimeSeriesDimxi = np.tile(TimeSeriesDimxi,(1,Nnodes))
                                TimeSeriesDimxj = np.tile(TimeSeriesDim,(Nnodes,1))
                                A_Expand = np.repeat(A,TimeSeries.shape[0],axis=0)
                                CoupledRegulation = np.sum(TimeSeriesDimxi*Regulation_func(TimeSeriesDimxj,gamma[ii])*A_Expand,axis=1,keepdims=True)
                                tmp = pd.DataFrame(data=CoupledRegulation,columns=[variables[i*dim+kk].format(gamma=str(gamma[ii]))])
                                CoupledActivation = pd.concat([CoupledActivation,tmp],axis=1)
    
        if dim > 4:
            raise ValueError('dim must be less than 5.')

    return pd.concat([CoupledActivation,CoupledTanh], axis = 1)

def Coupled_Rescaling_functions(TimeSeries, var_names, Nnodes, A, Rescaling = True):
    Timelength = np.size(TimeSeries, 0)
    dim_multi_Nnodes = np.size(TimeSeries, 1)
    dim = len(var_names)

    CoupledRescaling = pd.DataFrame()

    if Rescaling == True:
        if dim == 1:
            variables = [f'(aggr(sour({var})))/(aggr(1)+1e-6)' for var in var_names] + [f'(aggr(targ({var})*sour({var})))/(aggr(1)+1e-6)' for var in var_names] + [f'(aggr(sour({var})-targ({var})))/(aggr(1)+1e-6)' for var in var_names] # ['rescx1j','rescx1ix1j','rescx1jMinusx1i']
            
            for i in range(3):
                if i == 0:
                    for kk in range(dim):
                        TimeSeriesDim = TimeSeries[:,kk::dim]
                        TimeSeriesDimxi = np.reshape(TimeSeriesDim,(-1,1),order='F')
                        TimeSeriesDimxi = np.tile(TimeSeriesDimxi,(1,Nnodes))
                        TimeSeriesDimxj = np.tile(TimeSeriesDim,(Nnodes,1))
                        A_Expand = np.repeat(A,TimeSeries.shape[0],axis=0)
                        CoupledRescal = np.sum(TimeSeriesDimxj*A_Expand,axis=1,keepdims=True)/np.sum(A_Expand,axis=1,keepdims=True)
                        tmp = pd.DataFrame(data=CoupledRescal,columns=[variables[i*dim+kk]])
                        CoupledRescaling = pd.concat([CoupledRescaling,tmp],axis=1)
                elif i == 1:
                    for kk in range(dim):
                        TimeSeriesDim = TimeSeries[:,kk::dim]
                        TimeSeriesDimxi = np.reshape(TimeSeriesDim,(-1,1),order='F')
                        TimeSeriesDimxi = np.tile(TimeSeriesDimxi,(1,Nnodes))
                        TimeSeriesDimxj = np.tile(TimeSeriesDim,(Nnodes,1))
                        A_Expand = np.repeat(A,TimeSeries.shape[0],axis=0)
                        CoupledRescal = np.sum(TimeSeriesDimxi*TimeSeriesDimxj*A_Expand,axis=1,keepdims=True)/np.sum(A_Expand,axis=1,keepdims=True)
                        tmp = pd.DataFrame(data=CoupledRescal,columns=[variables[i*dim+kk]])
                        CoupledRescaling = pd.concat([CoupledRescaling,tmp],axis=1)
                elif i == 2:
                    for kk in range(dim):
                        TimeSeriesDim = TimeSeries[:,kk::dim]
                        TimeSeriesDimxi = np.reshape(TimeSeriesDim,(-1,1),order='F')
                        TimeSeriesDimxi = np.tile(TimeSeriesDimxi,(1,Nnodes))
                        TimeSeriesDimxj = np.tile(TimeSeriesDim,(Nnodes,1))
                        A_Expand = np.repeat(A,TimeSeries.shape[0],axis=0)
                        CoupledRescal = np.sum((TimeSeriesDimxj-TimeSeriesDimxi)*A_Expand,axis=1,keepdims=True)/np.sum(A_Expand,axis=1,keepdims=True)
                        tmp = pd.DataFrame(data=CoupledRescal,columns=[variables[i*dim+kk]])
                        CoupledRescaling = pd.concat([CoupledRescaling,tmp],axis=1)
    
        if dim == 2:
            variables = [f'(aggr(sour({var})))/(aggr(1)+1e-6)' for var in var_names] + [f'(aggr(targ({var})*sour({var})))/(aggr(1)+1e-6)' for var in var_names] + [f'(aggr(sour({var})-targ({var})))/(aggr(1)+1e-6)' for var in var_names] # ['rescx1j','rescx2j','rescx1ix1j','rescx2ix2j','rescx1jMinusx1i','rescx2jMinusx2i','x2irescx2j']
            
            
            for i in range(3):
                if i == 0:
                    for kk in range(dim):
                        TimeSeriesDim = TimeSeries[:,kk::dim]
                        TimeSeriesDimxi = np.reshape(TimeSeriesDim,(-1,1),order='F')
                        TimeSeriesDimxi = np.tile(TimeSeriesDimxi,(1,Nnodes))
                        TimeSeriesDimxj = np.tile(TimeSeriesDim,(Nnodes,1))
                        A_Expand = np.repeat(A,TimeSeries.shape[0],axis=0)
                        CoupledRescal = np.sum(TimeSeriesDimxj*A_Expand,axis=1,keepdims=True)/np.sum(A_Expand,axis=1,keepdims=True)
                        tmp = pd.DataFrame(data=CoupledRescal,columns=[variables[i*dim+kk]])
                        CoupledRescaling = pd.concat([CoupledRescaling,tmp],axis=1)
                elif i == 1:
                    for kk in range(dim):
                        TimeSeriesDim = TimeSeries[:,kk::dim]
                        TimeSeriesDimxi = np.reshape(TimeSeriesDim,(-1,1),order='F')
                        TimeSeriesDimxi = np.tile(TimeSeriesDimxi,(1,Nnodes))
                        TimeSeriesDimxj = np.tile(TimeSeriesDim,(Nnodes,1))
                        A_Expand = np.repeat(A,TimeSeries.shape[0],axis=0)
                        CoupledRescal = np.sum(TimeSeriesDimxi*TimeSeriesDimxj*A_Expand,axis=1,keepdims=True)/np.sum(A_Expand,axis=1,keepdims=True)
                        tmp = pd.DataFrame(data=CoupledRescal,columns=[variables[i*dim+kk]])
                        CoupledRescaling = pd.concat([CoupledRescaling,tmp],axis=1)
                elif i == 2:
                    for kk in range(dim):
                        TimeSeriesDim = TimeSeries[:,kk::dim]
                        TimeSeriesDimxi = np.reshape(TimeSeriesDim,(-1,1),order='F')
                        TimeSeriesDimxi = np.tile(TimeSeriesDimxi,(1,Nnodes))
                        TimeSeriesDimxj = np.tile(TimeSeriesDim,(Nnodes,1))
                        A_Expand = np.repeat(A,TimeSeries.shape[0],axis=0)
                        CoupledRescal = np.sum((TimeSeriesDimxj-TimeSeriesDimxi)*A_Expand,axis=1,keepdims=True)/np.sum(A_Expand,axis=1,keepdims=True)
                        tmp = pd.DataFrame(data=CoupledRescal,columns=[variables[i*dim+kk]])
                        CoupledRescaling = pd.concat([CoupledRescaling,tmp],axis=1)


        if dim == 3:
            variables = [f'(aggr(sour({var})))/(aggr(1)+1e-6)' for var in var_names] + [f'(aggr(targ({var})*sour({var})))/(aggr(1)+1e-6)' for var in var_names] + [f'(aggr(sour({var})-targ({var})))/(aggr(1)+1e-6)' for var in var_names] # ['rescx1j','rescx2j','rescx3j','rescx1ix1j','rescx2ix2j','rescx3ix3j','rescx1jMinusx1i','rescx2jMinusx2i','rescx3jMinusx3i']
            
            
            for i in range(3):
                if i == 0:
                    for kk in range(dim):
                        TimeSeriesDim = TimeSeries[:,kk::dim]
                        TimeSeriesDimxi = np.reshape(TimeSeriesDim,(-1,1),order='F')
                        TimeSeriesDimxi = np.tile(TimeSeriesDimxi,(1,Nnodes))
                        TimeSeriesDimxj = np.tile(TimeSeriesDim,(Nnodes,1))
                        A_Expand = np.repeat(A,TimeSeries.shape[0],axis=0)
                        CoupledRescal = np.sum(TimeSeriesDimxj*A_Expand,axis=1,keepdims=True)/np.sum(A_Expand,axis=1,keepdims=True)
                        tmp = pd.DataFrame(data=CoupledRescal,columns=[variables[i*dim+kk]])
                        CoupledRescaling = pd.concat([CoupledRescaling,tmp],axis=1)
                elif i == 1:
                    for kk in range(dim):
                        TimeSeriesDim = TimeSeries[:,kk::dim]
                        TimeSeriesDimxi = np.reshape(TimeSeriesDim,(-1,1),order='F')
                        TimeSeriesDimxi = np.tile(TimeSeriesDimxi,(1,Nnodes))
                        TimeSeriesDimxj = np.tile(TimeSeriesDim,(Nnodes,1))
                        A_Expand = np.repeat(A,TimeSeries.shape[0],axis=0)
                        CoupledRescal = np.sum(TimeSeriesDimxi*TimeSeriesDimxj*A_Expand,axis=1,keepdims=True)/np.sum(A_Expand,axis=1,keepdims=True)
                        tmp = pd.DataFrame(data=CoupledRescal,columns=[variables[i*dim+kk]])
                        CoupledRescaling = pd.concat([CoupledRescaling,tmp],axis=1)
                elif i == 2:
                    for kk in range(dim):
                        TimeSeriesDim = TimeSeries[:,kk::dim]
                        TimeSeriesDimxi = np.reshape(TimeSeriesDim,(-1,1),order='F')
                        TimeSeriesDimxi = np.tile(TimeSeriesDimxi,(1,Nnodes))
                        TimeSeriesDimxj = np.tile(TimeSeriesDim,(Nnodes,1))
                        A_Expand = np.repeat(A,TimeSeries.shape[0],axis=0)
                        CoupledRescal = np.sum((TimeSeriesDimxj-TimeSeriesDimxi)*A_Expand,axis=1,keepdims=True)/np.sum(A_Expand,axis=1,keepdims=True)
                        tmp = pd.DataFrame(data=CoupledRescal,columns=[variables[i*dim+kk]])
                        CoupledRescaling = pd.concat([CoupledRescaling,tmp],axis=1)


        if dim == 4:
            variables = [f'(aggr(sour({var})))/(aggr(1)+1e-6)' for var in var_names] + [f'(aggr(targ({var})*sour({var})))/(aggr(1)+1e-6)' for var in var_names] + [f'(aggr(sour({var})-targ({var})))/(aggr(1)+1e-6)' for var in var_names] # ['rescx1j','rescx2j','rescx3j','rescx4j','rescx1ix1j','rescx2ix2j','rescx3ix3j','rescx4ix4j','rescx1jMinusx1i','rescx2jMinusx2i','rescx3jMinusx3i','rescx4jMinusx4i']
            
            
            for i in range(3):
                if i == 0:
                    for kk in range(dim):
                        TimeSeriesDim = TimeSeries[:,kk::dim]
                        TimeSeriesDimxi = np.reshape(TimeSeriesDim,(-1,1),order='F')
                        TimeSeriesDimxi = np.tile(TimeSeriesDimxi,(1,Nnodes))
                        TimeSeriesDimxj = np.tile(TimeSeriesDim,(Nnodes,1))
                        A_Expand = np.repeat(A,TimeSeries.shape[0],axis=0)
                        CoupledRescal = np.sum(TimeSeriesDimxj*A_Expand,axis=1,keepdims=True)/np.sum(A_Expand,axis=1,keepdims=True)
                        tmp = pd.DataFrame(data=CoupledRescal,columns=[variables[i*dim+kk]])
                        CoupledRescaling = pd.concat([CoupledRescaling,tmp],axis=1)
                elif i == 1:
                    for kk in range(dim):
                        TimeSeriesDim = TimeSeries[:,kk::dim]
                        TimeSeriesDimxi = np.reshape(TimeSeriesDim,(-1,1),order='F')
                        TimeSeriesDimxi = np.tile(TimeSeriesDimxi,(1,Nnodes))
                        TimeSeriesDimxj = np.tile(TimeSeriesDim,(Nnodes,1))
                        A_Expand = np.repeat(A,TimeSeries.shape[0],axis=0)
                        CoupledRescal = np.sum(TimeSeriesDimxi*TimeSeriesDimxj*A_Expand,axis=1,keepdims=True)/np.sum(A_Expand,axis=1,keepdims=True)
                        tmp = pd.DataFrame(data=CoupledRescal,columns=[variables[i*dim+kk]])
                        CoupledRescaling = pd.concat([CoupledRescaling,tmp],axis=1)
                elif i == 2:
                    for kk in range(dim):
                        TimeSeriesDim = TimeSeries[:,kk::dim]
                        TimeSeriesDimxi = np.reshape(TimeSeriesDim,(-1,1),order='F')
                        TimeSeriesDimxi = np.tile(TimeSeriesDimxi,(1,Nnodes))
                        TimeSeriesDimxj = np.tile(TimeSeriesDim,(Nnodes,1))
                        A_Expand = np.repeat(A,TimeSeries.shape[0],axis=0)
                        CoupledRescal = np.sum((TimeSeriesDimxj-TimeSeriesDimxi)*A_Expand,axis=1,keepdims=True)/np.sum(A_Expand,axis=1,keepdims=True)
                        tmp = pd.DataFrame(data=CoupledRescal,columns=[variables[i*dim+kk]])
                        CoupledRescaling = pd.concat([CoupledRescaling,tmp],axis=1)
        if dim > 4:
            raise ValueError('dim must be less than 5.')
    return CoupledRescaling.fillna(0)


# %%
if __name__ == '__main__':
    var_names = ['x1', 'x2']
    dim = len(var_names)
    Nnodes = 10
    Timelength = 100
    PolyOrder = 3
    coupled_PolyOrder = 2
    TimeSeries = np.random.rand(Timelength,Nnodes,dim).reshape(Timelength,Nnodes*dim)
    A = np.random.rand(Nnodes,Nnodes) > 0.5

    # %%
    tmp1 = Polynomial_functions(TimeSeries, var_names, Nnodes, PolyOrder)
    print(tmp1.columns)

    # %%
    tmp2 = Trigonometric(TimeSeries, var_names, Nnodes)
    print(tmp2.columns)

    # %%
    tmp3 = Exponential(TimeSeries, var_names, Nnodes)
    print(tmp3.columns)

    # %%
    tmp4 = Fractional(TimeSeries, var_names, Nnodes)
    print(tmp4.columns)

    # %%
    tmp5 = Activation(TimeSeries, var_names, Nnodes, Sigmoid = True, Tanh = True, Regulation = True)
    print(tmp5.columns)

    # %%
    tmp6 = rescaling(TimeSeries, var_names, Nnodes, A, Rescal = True)
    print(tmp6.columns)

    # %%
    tmp7 = coupled_Polynomial_functions(TimeSeries, var_names, Nnodes, A, coupled_PolyOrder)
    print(tmp7.columns)
    
    # %%
    tmp8 = Coupled_Trigonometric_functions(TimeSeries, var_names, Nnodes, A)
    print(tmp8.columns)

    # %%
    tmp9 = Coupled_Exponential_functions(TimeSeries, var_names, Nnodes, A)
    print(tmp9.columns)

    # %%
    tmp10 = Coupled_Fractional_functions(TimeSeries, var_names, Nnodes, A)
    print(tmp10.columns)
    
    # %%
    tmp11 = Coupled_Activation_functions(TimeSeries, var_names, Nnodes, A, Sigmoid = True, Tanh = True, Regulation = True)
    print(tmp11.columns)

    # %%
    tmp12 = Coupled_Rescaling_functions(TimeSeries, var_names, Nnodes, A, Rescaling = True)
    print(tmp12.columns)

    # %%
    tmp = pd.concat([tmp1,tmp2,tmp3,tmp4,tmp5,tmp6,tmp7,tmp8,tmp9,tmp10,tmp11,tmp12],axis=1)
    print(tmp.columns)

    # %%
    x1 = TimeSeries.reshape(Timelength, Nnodes, dim)[:, :, (0,)]
    x2 = TimeSeries.reshape(Timelength, Nnodes, dim)[:, :, (1,)]
    G = np.stack(np.nonzero(A.T),axis=1)
    Nedges = G.shape[0]
    def aggr(e):
        if not isinstance(e, np.ndarray):
            e = np.full((1, Nedges, 1), e)
        v = np.zeros((Timelength, Nnodes, e.shape[-1]))
        for i in range(Nedges):
            v[:, G[i, 1], :] += e[:, i, :]
        return v
    def sour(v):
        if not isinstance(v, np.ndarray):
            v = np.full((1, Nnodes, 1), v)
        return v[:, G[:, 0], :]
    def targ(v):
        if not isinstance(v, np.ndarray):
            v = np.full((1, Nnodes, 1), v)
        return v[:, G[:, 1], :]
    sin = np.sin
    cos = np.cos
    tan = np.tan
    exp = np.exp
    sigmoid = lambda x, alpha, beta: 1/(1+np.exp(-alpha*(x-beta)))
    tanh = np.tanh
    regulation = lambda x, gamma: (x**gamma)/(x**gamma+1)
    rescaling = lambda x: x / aggr(sour(1))

    # %%
    for column in tmp:
        pred = eval(column).T.reshape(-1)
        true = tmp[column].to_numpy()
        diff = pred - true
        diff[diff == np.inf] = 0
        diff[diff == -np.inf] = 0
        print(np.abs(diff).max(), column)
