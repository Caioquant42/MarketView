import arch
import numpy as np
import pandas as pd
import warnings
import matplotlib.pyplot as plt
class VolatilityModel:
    def __init__(self, underlying_df):
        self.underlying_df = underlying_df


    def std(self,window:int):
        rolling_stdev = self.underlying_df['log_return'].rolling(window=window).std()*np.sqrt(252)
        self.underlying_df['STD'] = rolling_stdev
        return self.underlying_df
 
    def ewma(self, t_intervals, decay=0.94):
        if not isinstance(t_intervals, int):
            raise ValueError("t_intervals should be an integer")

        returns = self.underlying_df['log_return']
        ewma_volatility = returns.ewm(span=t_intervals, min_periods=t_intervals).std() * np.sqrt(252)
        self.underlying_df['EWMA'] = ewma_volatility
        return self.underlying_df
    
    def garch(self, window:int):
        if not isinstance(window, int):
            raise ValueError("window should be an integer")
        
                # Suppress warnings
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            returns = self.underlying_df['log_return']
            garch_volatility = returns.rolling(window=window).apply(lambda x: arch.arch_model(x, vol='Garch', p=1, q=1).fit().conditional_volatility.iloc[-1])*np.sqrt(252)
            self.underlying_df['GARCH(1,1)'] = garch_volatility
        return self.underlying_df   
    def garch_forecast(self, window: int, forecast_horizon: int):
        if not isinstance(window, int) or not isinstance(forecast_horizon, int):
            raise ValueError("window and forecast_horizon should be integers")

        # Suppress warnings
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            returns = self.underlying_df['log_return']            
            garch_model = arch.arch_model(returns.iloc[-window:], vol='Garch', p=1, q=1)
            results = garch_model.fit()
            forecast = results.forecast(start=None, horizon=forecast_horizon)
            self.garch_forecast_values = np.sqrt(forecast.variance.values)* np.sqrt(252)
            print(self.garch_forecast_values)
            garch_volatility =  self.underlying_df['GARCH(1,1)']      
            combined_volatility = np.concatenate((garch_volatility, self.garch_forecast_values[-1]))    
            date_range = pd.date_range(start=self.underlying_df.index[0], periods=len(combined_volatility), freq='D')
            # Plot historical and forecasted volatility
            plt.figure(figsize=(12, 6))
            plt.plot(date_range, combined_volatility, label='Historical Volatility', color='purple')
            plt.fill_between(date_range[-forecast_horizon:], combined_volatility[-forecast_horizon:], forecast.variance.values[-1], color='orange', alpha=0.5, label='Forecasted Volatility')
            plt.title('Historical and Forecasted Volatility')
            plt.xlabel('Date')
            plt.ylabel('Volatility')
            plt.legend()
            plt.grid(True)
            plt.show()                      
            return self.garch_forecast_values
    
    def rogers_satchell_volatility(self, window):
        if not isinstance(window, int):
            raise ValueError("window should be an integer")

        O = self.underlying_df['Abertura']
        H = self.underlying_df['Máxima']
        L = self.underlying_df['Mínima']
        C = self.underlying_df['Fechamento']

        # Calculate rs for each window
        rs = np.log(H / C) * np.log(H / O) + np.log(L / C) * np.log(L / O)

        # Calculate variance of rs using rolling window
        variance_rs = rs.rolling(window=window, min_periods=window).mean()

        # Calculate standard deviation of rs
        std_deviation_rs = np.sqrt(variance_rs)
        self.underlying_df['RS'] = std_deviation_rs*np.sqrt(252)
        return self.underlying_df

    
    def all_vol_models(self,window):
        if not isinstance(window, int):
            raise ValueError("window should be an integer")
        self.std(window)
        self.ewma(window)
        self.rogers_satchell_volatility(window)
        #self.garch(window)
        return self.underlying_df


