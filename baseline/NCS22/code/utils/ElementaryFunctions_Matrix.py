# -*- coding: UTF-8 -*-
"""
Created on Wed July 9 01:13:52 2021
@authors: Tingting Gao, Gang Yan

"""

import pandas as pd
import numpy as np
import sys
from .ElementaryFunctionsPool import *

def ElementaryFunctions_Matrix(TimeSeries, var_names, Nnodes, A, 
                               polynomial_order=3, coupled_polynomial_order=2,
                               activation_alpha=[1,5,10], activation_beta=[0, 1, 5, 10], activation_gamma=[1,2,5,10],
                               coupled_activation_alpha=[10], coupled_activation_beta=[1], coupled_activation_gamma=[1,2,5],
                               PolynomialIndex = True, TrigonometricIndex = True, ExponentialIndex = True, 
                               FractionalIndex = True, ActivationIndex = True, RescalingIndex = True, 
                               CoupledPolynomialIndex = True, CoupledTrigonometricIndex = True, CoupledExponentialIndex = True, 
                               CoupledFractionalIndex = True, CoupledActivationIndex = True, CoupledRescalingIndex = True):
    
    ElementaryMatrix = pd.DataFrame()
    if PolynomialIndex == True:
        ElementaryMatrix = pd.concat([ElementaryMatrix,Polynomial_functions(TimeSeries, var_names, Nnodes, PolyOrder=polynomial_order)],axis=1)
    if TrigonometricIndex == True:
        ElementaryMatrix = pd.concat([ElementaryMatrix, Trigonometric(TimeSeries, var_names, Nnodes, Sin = True, Cos = True, Tan = True)],axis=1)
    if ExponentialIndex == True:
        ElementaryMatrix = pd.concat([ElementaryMatrix, Exponential(TimeSeries, var_names, Nnodes, expomential = True)],axis=1)
    if FractionalIndex == True:
        ElementaryMatrix = pd.concat([ElementaryMatrix, Fractional(TimeSeries, var_names, Nnodes, fractional = True)],axis=1)
    if ActivationIndex == True:
        ElementaryMatrix = pd.concat([ElementaryMatrix, Activation(TimeSeries, var_names, Nnodes, Sigmoid = True, Tanh = True, Regulation = True, alpha=activation_alpha, beta=activation_beta, gamma=activation_gamma)],axis=1)
    if RescalingIndex == True:
        ElementaryMatrix = pd.concat([ElementaryMatrix, rescaling(TimeSeries, var_names, Nnodes, A, Rescal = True)],axis=1)
    if CoupledPolynomialIndex == True:
        ElementaryMatrix = pd.concat([ElementaryMatrix, coupled_Polynomial_functions(TimeSeries, var_names, Nnodes, A, PolyOrder=coupled_polynomial_order)],axis=1)
    if CoupledTrigonometricIndex == True:
        ElementaryMatrix = pd.concat([ElementaryMatrix, Coupled_Trigonometric_functions(TimeSeries, var_names, Nnodes, A, Sine = True, Cos = False, Tan = False)],axis=1)
    if CoupledExponentialIndex == True:
        ElementaryMatrix = pd.concat([ElementaryMatrix, Coupled_Exponential_functions(TimeSeries, var_names, Nnodes, A, Exponential = True)],axis=1)
    if CoupledFractionalIndex == True:
        ElementaryMatrix = pd.concat([ElementaryMatrix, Coupled_Fractional_functions(TimeSeries, var_names, Nnodes, A, Fractional = True)],axis=1)
    if CoupledActivationIndex == True:
        ElementaryMatrix = pd.concat([ElementaryMatrix, Coupled_Activation_functions(TimeSeries, var_names, Nnodes, A, Sigmoid = True, Tanh = True, Regulation = True, alpha=coupled_activation_alpha, beta=coupled_activation_beta, gamma=coupled_activation_gamma)],axis=1)
    if CoupledRescalingIndex == True:
        ElementaryMatrix = pd.concat([ElementaryMatrix, Coupled_Rescaling_functions(TimeSeries, var_names, Nnodes, A, Rescaling = True)],axis=1)
        
    return ElementaryMatrix
