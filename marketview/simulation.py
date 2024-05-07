import matplotlib.pyplot as plt 
import numpy as np
import pandas as pd
import datetime
import scipy.stats as stats
from scipy.stats import norm , t, qmc



class Simulation:
    def __init__(self, S0, cdf_empirical_interpolated, sorted_values):
        self.cdf_empirical_interpolated = cdf_empirical_interpolated
        self.sorted_values = sorted_values
        self.S0 = S0

    def _run_simulation(self, t_intervals, iterations=10000, additional_iterations=1):
        t_intervals += 1
        # Initialize the array to store all simulations data
        all_simulations_data = np.zeros((t_intervals, iterations * (additional_iterations + 1)))
        price_list = np.zeros((t_intervals, iterations))
        # Store the data from the initial simulation
        all_simulations_data[:, :iterations] = price_list

        # Set the starting index for additional iterations
        start_index = iterations

        # Create additional arrays to store final values for each simulation
        additional_final_values = []

    
        for j in range(additional_iterations):
            # Initialize the price_list array
            price_list = np.zeros((t_intervals, iterations))
            price_list[0] = self.S0

            for i in range(1,t_intervals):
                # Calculate the z_sample value at the specified quantile for each iteration
                quantile = np.random.rand(iterations)
                z_sample_at_quantile_manual = np.interp(quantile, self.cdf_empirical_interpolated, self.sorted_values)

                # Modify this line to handle broadcasting correctly
                daily_returns = np.exp(z_sample_at_quantile_manual)
                # Generating Paths
                price_list[i, :] = price_list[i - 1, :] * daily_returns

            # Get the final values (arrival values) for each simulation
            final_values_t = price_list[-1, :]

            # Store the final values for the additional simulation
            additional_final_values.append(final_values_t)

            # Plot the simulated paths for each additional simulation
            plt.figure(figsize=(10, 6))
            plt.plot(price_list, alpha=0.5)
            plt.title(f'Simulated Bootstrap Stock Price Paths - Iteration {j + 1}')
            plt.xlabel('Time (Days)')
            plt.ylabel('Stock Price')
            plt.show()

        # Flatten the list of arrays into a single 1D array
        all_simulations_data = np.sort(np.concatenate(additional_final_values))
        sims = len(all_simulations_data)
        print("Total Simulations:", sims)

        return all_simulations_data
    
class MCBootstrap:
    def __init__(self, S0, cdf_empirical_interpolated, sorted_values):
        self.cdf_empirical_interpolated = cdf_empirical_interpolated
        self.sorted_values = sorted_values
        self.S0 = S0

    def _run_simulation(self, time_steps, iterations):                     
        U = np.random.rand(time_steps, iterations)
        Z = np.interp(U, self.cdf_empirical_interpolated, self.sorted_values)
        factor = np.exp(Z)
        paths = self.S0*np.cumprod(factor,axis=0)
        arrival_values = paths[-1]
        return arrival_values
        
        
        
        

class MCNormal:
    def __init__(self, S0, r, T, volatility):
        self.S0 = S0
        self.r = r
        self.volatility = volatility
        self.T = T

    def _run_simulation(self, time_steps, iterations):
        # Precompute constants
        dt = self.T / time_steps
        nudt = (self.r - 0.5*self.volatility**2)*dt
        volsdt = self.volatility*np.sqrt(dt)
        lnS = np.log(self.S0)
        price_list = np.zeros((iterations, time_steps + 1))  # Store price evolution for each iteration        
        
        # Monte Carlo Method
        for i in range(iterations):
            lnSt = lnS
            price_list[i, 0] = self.S0  # Set initial price for each path
            for j in range(time_steps):
                lnSt = lnSt + nudt + volsdt*np.random.normal()
                St = np.exp(lnSt)
                price_list[i, j+1] = St
                
        # Plot each path
        plt.figure(figsize=(8, 4))
        for i in range(iterations):
            plt.plot(np.arange(time_steps + 1) * dt, price_list[i], alpha=0.6)
        plt.title("Monte Carlo Simulation of Normal Distribution")
        plt.xlabel("Time")
        plt.ylabel("Price")
        plt.show()
        
        # Calculate and return final price of each path
        ST = price_list[:, -1]        
        return ST
    
class MCNormal_Sobol:
    def __init__(self, S0, r, T, volatility):
        self.S0 = S0
        self.r = r
        self.volatility = volatility
        self.T = T

    def _run_simulation(self, time_steps, iterations):
        # Precompute constants
        dt = self.T / time_steps
        nudt = (self.r - 0.5*self.volatility**2)*dt
        volsdt = self.volatility*np.sqrt(dt)
        lnS = np.log(self.S0)
        price_list = np.zeros((iterations, time_steps + 1))  # Store price evolution for each iteration        
        
        # Monte Carlo Method
        for i in range(iterations):
            lnSt = lnS
            price_list[i, 0] = self.S0  # Set initial price for each path
            Z = sobol_norm(time_steps)  # Generate Sobol sequences
            for j in range(time_steps):
                lnSt = lnSt + nudt + volsdt*Z[j, i]
                St = np.exp(lnSt)
                price_list[i, j+1] = St
                
        # Plot each path
        plt.figure(figsize=(8, 4))
        for i in range(iterations):
            plt.plot(np.arange(time_steps + 1) * dt, price_list[i], alpha=0.6)
        plt.title("Monte Carlo Simulation of Normal Distribution")
        plt.xlabel("Time")
        plt.ylabel("Price")
        plt.show()
        
        # Calculate and return final price of each path
        ST = price_list[:, -1]        
        return ST

class MCNormal_vectorized:
    def __init__(self, S0, r, T, volatility):
        self.S0 = S0
        self.r = r
        self.volatility = volatility
        self.T = T

    def _run_simulation(self, time_steps, iterations):
        time_steps = 1 #Skipping to the end
        # Precompute constants
        dt = self.T / time_steps
        nudt = (self.r - 0.5*self.volatility**2)*dt
        volsdt = self.volatility*np.sqrt(dt)
        lnS = np.log(self.S0)
        # Monte Carlo Method
        Z = np.random.normal(size=(time_steps, iterations))
        delta_lnSt = nudt + volsdt*Z
        lnSt = lnS + np.cumsum(delta_lnSt, axis=0)
        lnSt = np.concatenate( (np.full(shape=(1, iterations), fill_value=lnS), lnSt ) )
        ST = np.exp(lnSt)
        arrival_values = ST[-1]
        return arrival_values
    
class MCNormal_vectorized_Sobol:
    def __init__(self, S0, r, T, volatility):
        self.S0 = S0
        self.r = r
        self.volatility = volatility
        self.T = T

    def _run_simulation(self, time_steps, iterations):
        # Precompute constants
        dt = self.T / time_steps
        nudt = (self.r - 0.5*self.volatility**2)*dt
        volsdt = self.volatility*np.sqrt(dt)
        lnS = np.log(self.S0)

        # Monte Carlo Method
        Z = sobol_norm(time_steps * iterations, d=time_steps)
        Z = np.reshape(Z, (time_steps, iterations))
        delta_lnSt = nudt + volsdt*Z
        lnSt = lnS + np.cumsum(delta_lnSt, axis=0)
        lnSt = np.concatenate( (np.full(shape=(1, iterations), fill_value=lnS), lnSt ) )
        ST = np.exp(lnSt)
        arrival_values = ST[-1]
        return arrival_values

    
class MCTstudent:
    def __init__(self, S0, mu, volatility, degrees_of_freedom_t, loc, scale):
        self.S0 = S0
        self.mu = mu
        self.volatility = volatility
        self.degrees_of_freedom_t = degrees_of_freedom_t
        self.loc = loc
        self.scale = scale

    def run_simulation(self, t_intervals, iterations=10000):
        stdev = np.mean(self.volatility / np.sqrt(252))
        drift = self.mu - 0.5 * stdev ** 2
        price_list = np.zeros((t_intervals, iterations))
        price_list[0] = self.S0

        # Generate random samples from the t-distribution
        t_samples = t.rvs(self.degrees_of_freedom_t, loc=self.loc, scale=self.scale, size=(t_intervals, iterations))

        # Standardize the data
        z_samples = (t_samples - self.loc) / self.scale

        for i in range(1, t_intervals):  # Corrected loop condition
            daily_returns = np.exp(drift + stdev * z_samples[i])
            price_list[i] = price_list[i - 1] * daily_returns

        plt.figure(figsize=(8, 4))
        plt.plot(price_list)
        plt.title("Monte Carlo Simulation of T Distribution")
        plt.xlabel("Time")
        plt.ylabel("Price")
        plt.show()

        final_values_t = price_list[-1]
        return final_values_t

import numpy as np
import matplotlib.pyplot as plt

class MCLognormal:
    def __init__(self, S0, mu, volatility):
        self.S0 = S0
        self.mu = mu
        self.volatility = volatility

    def run_simulation(self, t_intervals, iterations=10000):
        stdev = np.mean(self.volatility / np.sqrt(252))
        drift = self.mu - 0.5 * self.volatility ** 2
        price_list = np.zeros((t_intervals, iterations))
        price_list[0] = self.S0

        for i in range(1, t_intervals):
            daily_returns = np.exp(drift + stdev * np.random.normal(0, 1, iterations))
            price_list[i] = price_list[i - 1] * daily_returns

        plt.figure(figsize=(8, 4))
        plt.plot(price_list)
        plt.title("Monte Carlo Simulation of Lognormal Distribution")
        plt.xlabel("Time")
        plt.ylabel("Price")
        plt.show()

        final_values_lognormal = price_list[-1]
        return final_values_lognormal
    
def sobol_norm(m, d=1):
    sampler = qmc.Sobol(d, scramble=True)
    x_sobol = sampler.random_base2(m)
    return stats.norm.ppf(x_sobol)