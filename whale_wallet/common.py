import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import norm, pareto, t, lognorm, expon, weibull_min, ks_2samp, kstest
import arch
from scipy.interpolate import CubicSpline
import statsmodels.api as sm
import warnings
from scipy.integrate import trapz
from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score, mean_squared_error
import plotly.express as px