import numpy as np
import pandas as pd
import statsmodels.api as sm
from scipy.stats import norm, t, lognorm , ks_2samp 
import matplotlib.pyplot as plt
import arch
from sklearn.preprocessing import MinMaxScaler
from scipy.interpolate import CubicSpline


class ReturnsCalculator:
    def __init__(self, asset_df, price_column='Fechamento'):
        self.asset_df = asset_df
        self.price_column = price_column
        self.drift_estimate = None
        self.sorted_values = float()
        self.interpolated_cdf_values = None
        self.sf_interpolated = None
        self.S0 = float()
        self.stdev = None

    def fit_dist(self,returns:float):
        # Fit the distributions
        shape_lognorm, loc_lognorm, scale_lognorm = lognorm.fit(np.exp(returns))
        mu_normal, std_normal = norm.fit(returns)
        degrees_of_freedom_t, loc_t, scale_t = t.fit(returns)
        self.shape_lognorm, self.loc_lognorm, self.scale_lognorm  = shape_lognorm, loc_lognorm, scale_lognorm
        self.mu_normal,self.std_normal = mu_normal,std_normal
        self.degrees_of_freedom_t,self.loc_t,self.scale_t = degrees_of_freedom_t, loc_t, scale_t

        # Kolmogorov-Smirnov Test
        ks_stat_lognorm, ks_pvalue_lognorm = ks_2samp(np.exp(returns), lognorm.rvs(shape_lognorm, loc_lognorm, scale_lognorm, size=len(returns)))
        ks_stat_normal, ks_pvalue_normal = ks_2samp(returns, norm.rvs(loc=mu_normal, scale=std_normal, size=len(returns)))
        ks_stat_t, ks_pvalue_t = ks_2samp(returns, t.rvs(degrees_of_freedom_t, loc_t, scale_t, size=len(returns)))
        # Print results
        print("Kolmogorov-Smirnov Test:")
        print(f"Log-Normal: KS Statistic = {ks_stat_lognorm}, p-value = {ks_pvalue_lognorm}")
        print(f"Normal: KS Statistic = {ks_stat_normal}, p-value = {ks_pvalue_normal}")
        print(f"Student's t: KS Statistic = {ks_stat_t}, p-value = {ks_pvalue_t}")
        print("Resultados Salvos.")
        # Plotting the data
        x_lognorm = np.linspace(returns.min(), returns.max(), len(returns))

        plt.figure(figsize=(8, 6))
        # plotting the data
        plt.hist(returns, bins=60, density=True, alpha=0.7, color='blue', label='Log Returns Data')

        # Plot the PDF of the log-normal distribution
        pdf_lognorm = lognorm.pdf(np.exp(x_lognorm), shape_lognorm, loc_lognorm, scale_lognorm)
        plt.plot(x_lognorm, pdf_lognorm, 'r-', label='Log-Normal PDF')

        # Plot the PDF of the normal distribution
        pdf_normal = norm.pdf(x_lognorm, mu_normal, std_normal)
        plt.plot(x_lognorm, pdf_normal, 'g-', label='Normal PDF')

        # Plot the PDF of the Student's t-distribution
        pdf_t = t.pdf(x_lognorm, degrees_of_freedom_t, loc_t, scale_t)
        plt.plot(x_lognorm, pdf_t, 'black', label='Student\'s t PDF')
        
        # Add labels and a legend
        plt.title("Comparison of Distribution Fits")
        plt.xlabel("Log Returns")
        plt.ylabel("Probability Density")
        plt.legend()

        # Show the plot
        plt.show()

        return {
            'mu_normal': self.mu_normal,
            'std_normal': self.std_normal,
            'shape_lognorm': self.shape_lognorm,
            'loc_lognorm': self.loc_lognorm,
            'scale_lognorm': self.scale_lognorm,
            'degrees_of_freedom_t': self.degrees_of_freedom_t,
            'loc_t': self.loc_t,
            'scale_t': self.scale_t
        }

    def calculate_returns(self):
        import warnings
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", FutureWarning)
            # Calculate simple_return, log_return, drift_estimate, and histogram_data
            self.asset_df['simple_return'] = self.asset_df[self.price_column].pct_change()
            self.asset_df['log_return'] = np.log(1 + self.asset_df['simple_return'])
            self.asset_df.dropna(subset=['simple_return', 'log_return'], inplace=True)
            

            # Calculate drift estimate
            X = sm.add_constant(np.arange(len(self.asset_df['log_return'])))
            model = sm.OLS(self.asset_df['log_return'], X).fit()
            self.drift_estimate = model.params[1]
            self.stdev = np.std(self.asset_df['log_return'])*np.sqrt(252)

            # Store calculated returns
            self.log_returns = self.asset_df['log_return']
            self.simple_returns = self.asset_df['simple_return']
            self.cdf_values()            
            return self.asset_df
    def cdf_values(self):
        self.interpolated_cdf_values = []  # Initialize the list here
        self.asset_df['simple_return'] = self.asset_df[self.price_column].pct_change()
        self.asset_df['log_return'] = np.log(1 + self.asset_df['simple_return'])
        self.asset_df.dropna(subset=['simple_return', 'log_return'], inplace=True)

        sorted_values = np.unique(np.sort(self.asset_df['log_return']))
        ecdf = np.arange(1, len(sorted_values) + 1) / len(sorted_values)
        cs = CubicSpline(sorted_values, ecdf)
        self.interpolated_cdf_values = cs(sorted_values)  # Convert the list to a NumPy array

        # Ensure that both arrays have the same length
        min_length = min(len(sorted_values), len(self.interpolated_cdf_values))
        self.sorted_values = sorted_values[:min_length]
        self.interpolated_cdf_values = self.interpolated_cdf_values[:min_length]
        self.sf_interpolated = 1 - self.interpolated_cdf_values
        self.S0 = self.asset_df[self.price_column].iloc[-1]
        self.mu = np.median(self.asset_df['log_return'])

        # Return a dictionary containing all the values
        return {
            'S0': self.S0,
            'sf_interpolated': self.sf_interpolated,
            'sorted_values': self.sorted_values,
            'interpolated_cdf_values': self.interpolated_cdf_values,
            'mu': self.mu
        }
    def print_options(self):
        print("Current price (S0): .S0")
        print("Simple Returns: .simple_returns")
        print("Log Returns: .log_returns")
        print("Drift Estimate: .drift_estimate")        
        print("Sorted Values: .sorted_values")
        print("Interpolated CDF Values: .interpolated_cdf_values")
        print("SF Interpolated: .sf_interpolated")
        print("SF Interpolated: .mu")



