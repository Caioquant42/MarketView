from typing import Any
from scipy.stats import kstest
from scipy.stats import pareto
from scipy.stats import expon
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
from scipy.stats import weibull_min
from scipy.integrate import trapz
import plotly.express as px
from datetime import datetime, date

class Survival:
    def __init__(self,underlying_df,threshold: float = 0.0):
        self.underlying_df = underlying_df
        self.threshold = threshold
        
    def _interarrival(self):
        below = self.underlying_df[self.underlying_df['log_return'] < self.threshold]      
        timestamps = below.index
        timestamps = pd.Series(timestamps)
        # Sort the timestamps in ascending order
        timestamps_sorted = timestamps.sort_values()
        interarrival_time = timestamps_sorted.diff()
        interarrival_time = interarrival_time.dropna()      
        self.interarrival_days = interarrival_time.dt.days

        return self.interarrival_days

    def _plot_interarrival(self, title: str = None):
        below = self.underlying_df[self.underlying_df['log_return'] < self.threshold]      
        timestamps = below.index
        timestamps = pd.Series(timestamps)
        # Sort the timestamps in ascending order
        timestamps_sorted = timestamps.sort_values()
        interarrival_time = timestamps_sorted.diff()
        interarrival_time = interarrival_time.dropna()
        today = datetime.today().date()
        last_available_date = timestamps_sorted.max()
        week_mask = '1111111'  # all days of the week included
        running_days = np.busday_count(last_available_date.date(), today, weekmask=week_mask)
        interarrival_days = interarrival_time.dt.days
        
        # Plot the histogram of interarrival_time
        plt.figure(figsize=(10, 6))
        plt.hist(interarrival_days, bins=40, color='skyblue', edgecolor='black')
        plt.title(f'{title} Interarrival Time negative returns below {self.threshold}%')
        plt.xlabel('Interarrival Time (days)')
        plt.ylabel('Frequency')
        plt.grid(True)

        # Plot the vertical line for running_days
        plt.axvline(x=running_days, color='red', linestyle='--', label=f'Running Days of last incident: {running_days}')
        plt.legend()

        plt.show()
    def _plot_occurence(self, title: str = None):
        below = self.underlying_df[self.underlying_df['log_return'] < self.threshold]
        
        fig = px.scatter(below, x=below.index, y='log_return', color='log_return',
                        labels={'log_return': 'Log Return', 'x': 'Date'},
                        title=f'Retornos Abaixo de {self.threshold}') 
        
        fig.update_traces(marker=dict(size=8, symbol='circle', color='red', line=dict(width=1)),
                        selector=dict(mode='markers'))
        
        fig.update_layout(xaxis=dict(title='Date'), yaxis=dict(title='Log Return'),
                        legend=dict(title=f'Log Returns < {self.threshold}'))
        
        fig.show()

    def mean_excess_loss(self,threshold=-0.05):
        loss = self.underlying_df[self.underlying_df['log_return'] < threshold]
        # Calculate the excess loss for each observation
        excess_losses = loss - threshold
        
        # Filter out non-positive excess losses (losses below threshold)
        positive_excess_losses = excess_losses[excess_losses > 0]
        
        # Calculate the mean excess loss
        mean_excess_loss = np.mean(positive_excess_losses)
    
        return mean_excess_loss
    def equilibrium_distribution(self, cdf_values, threshold): #### ADDD THISS LATERR
        # Calculate the proportion of exceedances over the threshold
        excess_proportion = (1 - cdf_values) / (1 - cdf_values[threshold])
        
        # Ensure that the proportion is bounded between 0 and 1
        equilibrium_values = np.maximum(0, np.minimum(1, excess_proportion))
        
        return equilibrium_values
    def fit_pareto(self):
        self._interarrival()
        # Fit the Pareto distribution
        shape_pareto, loc_pareto, scale_pareto = pareto.fit(self.interarrival_days.dropna())

        # Statistical Tests
        # Calculate AIC for Pareto distribution
        n = len(self.interarrival_days.dropna())
        k_pareto = 3  # 3 parameters: shape, loc, scale
        LL_pareto = pareto.logpdf(self.interarrival_days.dropna(), shape_pareto, loc_pareto, scale_pareto).sum()
        AIC_pareto = 2 * k_pareto - 2 * LL_pareto
        # Calculate BIC for Pareto distribution
        BIC_pareto = k_pareto * np.log(n) - 2 * LL_pareto
        # Perform Kolmogorov-Smirnov test
        ks_statistic, ks_pvalue = kstest(self.interarrival_days.dropna(), 'pareto', args=(shape_pareto, loc_pareto, scale_pareto))

        # Print the result of the Kolmogorov-Smirnov test
        print("Kolmogorov-Smirnov test statistic:", ks_statistic)
        print("Kolmogorov-Smirnov test p-value:", ks_pvalue)
        print("AIC for Pareto distribution:", AIC_pareto)
        print("BIC for Pareto distribution:", BIC_pareto)

        # Plot histogram of interarrival times
        plt.figure(figsize=(8, 6))
        plt.hist(self.interarrival_days, bins=40, color='skyblue', edgecolor='black', density=True, label='Histogram')

        # Plot the fitted Pareto distribution
        x_values = np.linspace(self.interarrival_days.min(), self.interarrival_days.max(), len(self.underlying_df))

        # Calculate the PDF, CDF, and SF using the parameters of the fitted Pareto distribution
        pdf_values = pareto.pdf(x_values, shape_pareto, loc_pareto, scale_pareto)
        self.cdf_values = pareto.cdf(x_values, shape_pareto, loc_pareto, scale_pareto)
        sf_values = 1 - self.cdf_values  # Calculate SF as 1 - CDF        
        
        # Calculate the hazard function
        self.hazard_values = pdf_values / sf_values
        # Calculate the cumulative hazard function using numerical integration
        self.cumulative_hazard_values = trapz(self.hazard_values, x_values)

        # Calculate the mean residual life (MRL) at each time point
        self.mean_residual_life_values = self.cumulative_hazard_values / sf_values

        mean_excess_loss = self.mean_excess_loss()
        #equilibrium_values = self.equilibrium_distribution(self.cdf_values, self.threshold)
        plt.plot(x_values, pdf_values, 'r-', lw=2, label='Fitted Pareto Distribution')

        plt.title('Distribution of Interarrival Times with Fitted Pareto Distribution')
        plt.xlabel('Interarrival Time (days)')
        plt.ylabel('Density')
        plt.legend()
        plt.grid(True)
        plt.show()

        # Print the parameters of the fitted Pareto distribution
        print("Shape parameter (α):", shape_pareto)
        print("Location parameter (β):", loc_pareto)
        print("Scale parameter (xm):", scale_pareto)
        params_df = pd.DataFrame({
            'Shape Pareto (α)': [shape_pareto],
            'Location Pareto (β)': [loc_pareto],
            'Scale Pareto (xm)': [scale_pareto]
            })
        
        print(self.hazard_values,self.cumulative_hazard_values,f'MRL:{self.mean_residual_life_values}',
              f'MEL:{mean_excess_loss}')#f'Integrated Tail: {equilibrium_values}')
        return params_df
    def fit_exponencial(self):
        self._interarrival()

        # Fit the exponential distribution to interarrival_days
        loc, scale = expon.fit(self.interarrival_days.dropna())

        # Plot histogram of interarrival times
        plt.figure(figsize=(8, 6))
        plt.hist(self.interarrival_days, bins=40, color='skyblue', edgecolor='black', density=True, label='Histogram')

        # Plot the fitted exponential distribution
        x = np.linspace(self.interarrival_days.min(), self.interarrival_days.max(), len(self.underlying_df))
        pdf = expon.pdf(x, loc, scale)
        plt.plot(x, pdf, 'r-', lw=2, label='Fitted Exponential Distribution')

        # Perform a KS test using an exponential distribution with specific parameters
        ks_statistic, ks_pvalue = kstest(self.interarrival_days.dropna(), 'expon', args=(loc, scale))

        # Calculate AIC for Exponential distribution
        k_exp = 2  # 2 parameters: loc, scale
        LL_exp = expon.logpdf(self.interarrival_days.dropna(), loc, scale).sum()
        AIC_exp = 2 * k_exp - 2 * LL_exp

        n = len(self.interarrival_days.dropna())
        # Calculate BIC for Exponential distribution
        BIC_exp = k_exp * np.log(n) - 2 * LL_exp
        print("Kolmogorov-Smirnov test statistic:", ks_statistic)
        print("Kolmogorov-Smirnov test p-value:", ks_pvalue)
        print("AIC for Exponential distribution:", AIC_exp)
        print("BIC for Exponential distribution:", BIC_exp)

        plt.title('Distribution of Interarrival Times with Fitted Exponential Distribution')
        plt.xlabel('Interarrival Time (days)')
        plt.ylabel('Density')
        plt.legend()
        plt.grid(True)
        plt.show()

        # Print the parameters of the fitted exponential distribution
        print("Location parameter (loc):", loc)
        print("Scale parameter (scale):", scale)
        params_df = pd.DataFrame({            
            'Location Exponencial': [loc],
            'Scale Exponencial': [scale]
            })
        return params_df
    
    def fit_weibull(self):
        self._interarrival()

        # Fit the Weibull distribution to interarrival_days
        shape, loc, scale = weibull_min.fit(self.interarrival_days.dropna(), floc=0)

        # Plot histogram of interarrival times
        plt.figure(figsize=(8, 6))
        plt.hist(self.interarrival_days, bins=40, color='skyblue', edgecolor='black', density=True, label='Histogram')

        # Plot the fitted Weibull distribution
        x_values = np.linspace(self.interarrival_days.min(), self.interarrival_days.max(), len(self.underlying_df))

        pdf_values = weibull_min.pdf(x_values, shape, loc, scale)

        self.cdf_values = weibull_min.cdf(x_values, shape, loc, scale)
        sf_values = 1 - self.cdf_values  # Calculate SF as 1 - CDF        
        
        # Calculate the hazard function
        self.hazard_values = pdf_values / sf_values
        # Calculate the cumulative hazard function using numerical integration
        self.cumulative_hazard_values = trapz(self.hazard_values, x_values)

        # Calculate the mean residual life (MRL) at each time point
        self.mean_residual_life_values = self.cumulative_hazard_values / sf_values

        mean_excess_loss = self.mean_excess_loss()

        plt.plot(x_values, pdf_values, 'r-', lw=2, label='Fitted Weibull Distribution')

        # Perform a KS test using a Weibull distribution with specific parameters
        ks_statistic, ks_pvalue = kstest(self.interarrival_days.dropna(), 'weibull_min', args=(shape, loc, scale))

        # Calculate AIC for Weibull distribution
        k_weibull = 3  # 3 parameters: shape, loc, scale
        LL_weibull = weibull_min.logpdf(self.interarrival_days.dropna(), shape, loc, scale).sum()
        AIC_weibull = 2 * k_weibull - 2 * LL_weibull

        n = len(self.interarrival_days.dropna())
        # Calculate BIC for Weibull distribution
        BIC_weibull = k_weibull * np.log(n) - 2 * LL_weibull

        print("Kolmogorov-Smirnov test statistic:", ks_statistic)
        print("Kolmogorov-Smirnov test p-value:", ks_pvalue)
        print("AIC for Weibull distribution:", AIC_weibull)
        print("BIC for Weibull distribution:", BIC_weibull)

        plt.title(f'Distribution of Interarrival Times with Fitted Weibull Distribution {self.threshold}')
        plt.xlabel('Interarrival Time (days)')
        plt.ylabel('Density')
        plt.legend()
        plt.grid(True)
        plt.show()

        # Print the parameters of the fitted Weibull distribution
        print("Shape parameter (k):", shape)
        print("Location parameter (loc):", loc)
        print("Scale parameter (scale):", scale)

        params_df = pd.DataFrame({            
            'Shape Weibull (k)': [shape],
            'Location Weibull': [loc],
            'Scale Weibull': [scale]
        })
        print(self.hazard_values,self.cumulative_hazard_values,f'MRL:{self.mean_residual_life_values}',
              f'MEL:{mean_excess_loss}')#f'Integrated Tail: {equilibrium_values}')
        return params_df
    




    
    
    
