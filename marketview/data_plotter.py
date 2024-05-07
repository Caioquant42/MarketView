import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import pandas as pd
import plotly.express as px
from scipy.stats import kurtosis, skew
from datetime import date
from scipy.stats import probplot
class DataPlotter:


    def __init__(self,data):
        self.data = data        
    def plot_data(self, column_name='Fechamento'):
        fig = px.line(x=self.data.index, y=self.data[column_name], labels={'x': 'Date', 'y': 'Preço'}, title=f"{column_name}")
        fig.update_layout(showlegend=True)
        fig.show()      

    def plot_histogram(self, data_returns,bins=60,color: str = 'blue'):
        median = np.median(data_returns.values)
        data_mad = np.mean(np.abs(data_returns.values - median))
        data_kurtosis = kurtosis(data_returns.values)
        data_skewness = skew(data_returns.values)

        plt.figure(figsize=(8, 6))
        sns.histplot(data_returns.values, bins=bins, kde=True, color=color, label='Histograma', stat='probability')
        plt.title('Histograma')
        plt.xlabel('Retorno')
        plt.ylabel('Probabilidade')
        plt.legend()

        # Annotate plot with kurtosis, skewness, MAD, and median values
        plt.text(0.05, 0.75, f'Kurtosis: {data_kurtosis:.2f}', transform=plt.gca().transAxes)
        plt.text(0.05, 0.80, f'Skewness: {data_skewness:.2f}', transform=plt.gca().transAxes)
        plt.text(0.05, 0.85, f'MAD: {data_mad:.2f}', transform=plt.gca().transAxes)
        plt.text(0.05, 0.90, f'Median: {median:.2f}', transform=plt.gca().transAxes)

        plt.show()


    def plot_cdf(self, sorted_values, interpolated_cdf_values):
        # Plot the interpolated CDF
        plt.figure(figsize=(8, 6))
        plt.plot(sorted_values, interpolated_cdf_values, label='Interpolated CDF', color='blue')
        plt.title('Interpolated Cumulative Distribution Function (CDF)')
        plt.xlabel('Values')
        plt.ylabel('Interpolated CDF')
        plt.legend()
        plt.show()

def plot_boxplot(S0: float, all_simulations_data: np.ndarray):
        

        median = np.median(all_simulations_data)
        q25 = np.percentile(all_simulations_data, 25)
        q75 = np.percentile(all_simulations_data, 75)

        # Plot the box plot with MAD
        plt.figure(figsize=(10, 6))
        plt.boxplot(all_simulations_data, vert=False, labels=['Final Values'])

        # Add median line
        plt.axvline(median, color='red', linestyle='dashed', linewidth=2, label=f'Median: {median:.2f}')

        # Add quartiles
        plt.axvline(q25, color='orange', linestyle='dashed', linewidth=2, label=f'Q25: {q25:.2f}')
        plt.axvline(q75, color='orange', linestyle='dashed', linewidth=2, label=f'Q75: {q75:.2f}')

        # Calculate the MAD
        mad = np.mean(np.abs(all_simulations_data - median))
        # Add MAD line
        plt.axvline(median + mad, color='green', linestyle='dashed', linewidth=2, label=f'Median ± MAD: {median + mad:.2f}')
        plt.axvline(median - mad, color='green', linestyle='dashed', linewidth=2, label=f'Median ± MAD: {median - mad:.2f}')

        # Calculate and print the interquartile range (IQR)
        iqr = q75 - q25
        print(f"Interquartile Range (IQR): {iqr:.2f}")
        print(f"Mean Absolute Difference (MAD): {mad:.2f}")

        plt.title('Box Plot of Final Stock Price Values with MAD')
        plt.xlabel('Stock Price')
        plt.legend()
        plt.show()

        # Print the median move
        median_move = ((median - S0) / S0) * 100  # Access S0 using self.S0
        median_move = round(median_move, 4)
        print("Preço Atual:", S0)
        print(f"Variação Esperada: {median_move}%")
        
def _POP_sweet_spot(option_df):
        option_df['POP_derivative'] = np.gradient(option_df['POP_Ratio'], option_df['Strike']) 
        return option_df        


def plot_sweet_spot(option_df, option_type: str):
    if option_type == 'call':
        #option_df = _POP_sweet_spot(option_df)
        min_derivative_index = option_df['POP_derivative'].idxmin()
        min_derivative_strike = option_df.loc[min_derivative_index, 'Strike']        
        plt.figure(figsize=(10, 5))
        plt.plot(option_df['Strike'], option_df['POP_derivative'], label='Derivative', color='blue', linestyle='-', marker='o')
        plt.scatter(min_derivative_strike, option_df.loc[min_derivative_index, 'POP_derivative'], color='red', label=f'Minimum at {min_derivative_strike}', marker='x')
        plt.title('POP - Sweet Spots Calls')
        plt.xlabel('Call Strike')
        plt.ylabel('Derivative')
        plt.legend()
        plt.tight_layout()
        plt.show()


    elif option_type == 'put':
        #option_df = _POP_sweet_spot(option_df)
        min_derivative_index = option_df['POP_derivative'].idxmin()
        min_derivative_strike = option_df.loc[min_derivative_index, 'Strike']
        plt.figure(figsize=(10, 5))
        plt.plot(option_df['Strike'], option_df['POP_derivative'], label='Derivative', color='blue', linestyle='-', marker='o')
        plt.scatter(min_derivative_strike, option_df.loc[min_derivative_index, 'POP_derivative'], color='red', label=f'Minimum at {min_derivative_strike}', marker='x')
        plt.title('POP Sweet Spot - Puts')
        plt.xlabel('Put Delta')
        plt.ylabel('Derivative')
        plt.legend()
        plt.tight_layout()
        plt.show()

def sloppery (call_df, put_df, lower_bound:float = 0.7, upper_bound:float = 1.1):
    filtered_option_df = call_df[(call_df['Moneyness'] >= lower_bound) & (call_df['Moneyness'] <= upper_bound)]


def plot_parity(parity_df):
    fig = px.scatter_3d(parity_df, x='Parity', y='Put Call Ratio', z='Call Strike', color='Normalized Total Volume',
                        labels={'Parity': 'Parity', 'Put Call Ratio': 'Put Call Ratio', 'Call Strike': 'Call Strike',
                                'Total Volume': 'Total Volume'},
                        title='3D Scatter Plot with Parity, Put Call Ratio, and Call Strike')
    fig.show(width=600, height=800) 

    fig.write_html("Parity_3d.html")


def create_filtered_df(option_df,lower_bound, upper_bound):
    return option_df[(option_df['Moneyness'] >= lower_bound) & (option_df['Moneyness'] <= upper_bound)]

def plot_aws(option_df, lower_bound:float = 0.8, upper_bound:float=1.10):
    filtered_option_df = create_filtered_df(option_df,lower_bound, upper_bound)
    fig = px.scatter_3d(filtered_option_df, x='Delta Difference', y='MC_Ratio', z='Strike',title='3D Plot, and Leverage',color='Leverage')
    fig.update_traces(marker=dict(size=5))
    fig.show(width=300, height=1000) 
    fig.write_html("aws_3d.html")
      

def step_plot(x, y, label:str='title',color:str='gray'):
    plt.step(x, y, label=label, color=color, linestyle='-', marker='o')
    plt.show()

def line_plot(x,y,label:str='title',color:str='gray'):
    # Plot the absolute difference between call_sf_interpolated and call_strike_market
    plt.figure(figsize=(10, 6))
    plt.plot(x, y, color=color, label=label)
    plt.grid(True)
    plt.show()

def scatter_plot(x,y,label:str='title',color:str='gray'):
    plt.scatter(x,y,label=label,color=color)
    plt.show()

def plot_vol(underlying_df,which:str=None):
    # Plotting
    plt.figure(figsize=(16, 8))

    if which =='Std':
        plt.plot(underlying_df.index, underlying_df['Std'], label='Std', color='purple')
        plt.plot(underlying_df.index, underlying_df['RS'], label='RS', color='blue')

    elif which =='EWMA':
            plt.plot(underlying_df.index, underlying_df['RS'], label='RS', color='blue')
            plt.plot(underlying_df.index, underlying_df['EWMA'], label='EWMA', color='green')
    elif which == 'GARCH':
            plt.plot(underlying_df.index, underlying_df['RS'], label='RS', color='blue')
            plt.plot(underlying_df.index, underlying_df['GARCH'], label='GARCH', color='red')

    else:
        # Plot RS volatility
        plt.plot(underlying_df.index, underlying_df['RS'], label='RS', color='blue')

        # Plot EWMA volatility
        plt.plot(underlying_df.index, underlying_df['EWMA'], label='EWMA', color='green')

        # Plot GARCH volatility
        plt.plot(underlying_df.index, underlying_df['GARCH'], label='GARCH', color='red')

        # Plot Std volatility
        plt.plot(underlying_df.index, underlying_df['Std'], label='Std', color='purple')






    plt.title('Volatility Comparison')
    plt.xlabel('Date')
    plt.ylabel('Volatility')
    plt.legend()
    plt.grid(True)
    plt.xticks(rotation=45)
    plt.tight_layout()
    plt.show()


def plot_tail_market(taleb_df):
    if 'Market Prices' in taleb_df.columns:
        # Plot using Plotly Express
        fig = px.line(taleb_df, x='Strikes', y=['Market Prices', 'Model Prices (Taleb_2019)'],
                    labels={'value': 'Price', 'Strikes': 'Strike'},
                    title='Market Prices vs Model Prices (Taleb 2019)')

        # Show the plot
        fig.show()
        taleb_df['Ratio'] = taleb_df['Market Prices'] / taleb_df['Model Prices (Taleb_2019)']

        fig = px.line(taleb_df, x='Strikes', y=['Ratio'],
            labels={'value': 'Ratio', 'Strikes': 'Strike'},
            title='Ratio - (Market / Model)')

        # Show the plot
        fig.show()
    else:
        fig = px.line(taleb_df, x='Theoretical Strikes', y=[ 'Model Prices (Taleb_2019)'],
            labels={'value': 'Price', 'Theoretical Strikes': 'Theoretical Strike'},
            title='Theoretical Prices (Taleb 2019)')

        # Show the plot
        fig.show()



    # Here is the final result comparing with market prices, must have input of excel file with last and strike prices.
    # Also interective 2D
def plot_histogram(data_returns,bins=60,color: str = 'blue'):
    median = np.median(data_returns)
    data_mad = np.mean(np.abs(data_returns - median))
    data_kurtosis = kurtosis(data_returns)
    data_skewness = skew(data_returns)

    plt.figure(figsize=(8, 6))
    sns.histplot(data_returns, bins=bins, kde=True, color=color, label='Histograma', stat='probability')
    plt.title('Histograma')
    plt.xlabel('Retorno')
    plt.ylabel('Probabilidade')
    plt.legend()

    # Annotate plot with kurtosis, skewness, MAD, and median values
    plt.text(0.05, 0.75, f'Kurtosis: {data_kurtosis:.2f}', transform=plt.gca().transAxes)
    plt.text(0.05, 0.80, f'Skewness: {data_skewness:.2f}', transform=plt.gca().transAxes)
    plt.text(0.05, 0.85, f'MAD: {data_mad:.2f}', transform=plt.gca().transAxes)
    plt.text(0.05, 0.90, f'Median: {median:.2f}', transform=plt.gca().transAxes)

    plt.show()

def plot_3d(option_df,x_axis,y_axis,z_axis, title:str = 'Option Plot'):    
    fig = px.scatter_3d(option_df, x=x_axis, y=y_axis, z=z_axis,title=f'3D {title}')
    fig.update_traces(marker=dict(size=5))
    fig.show(width=300, height=1000) 
    #fig.write_html("3d.html") #save optional

def plot_4d(option_df,x_axis,y_axis,z_axis,w_axis, title:str = 'Option Plot'):    
    fig = px.scatter_3d(option_df, x=x_axis, y=y_axis, z=z_axis,title=f'4D {title}',color = w_axis)
    fig.update_traces(marker=dict(size=5))
    fig.show(width=300, height=1000) 
    fig.write_html("4d.html")
    #fig.write_html("4d.html") #save optional

def X_plot(call_df, put_df):
    # X plot
    plt.figure(figsize=(12, 6))
    # Plot calls
    plt.step(call_df['Strike'], call_df['POP-normal'], label='Call BlackScholes-POP', color='black', linestyle='-', marker='o')
    plt.step(call_df['Strike'], call_df['POP-bootstrap'], label='Call Model-POP', color='blue', linestyle='-', marker='o')
    # Plot Puts
    plt.step(put_df['Strike'], put_df['POP-normal'], label='Put BlackScholes-POP', color='black', linestyle='-', marker='o')
    plt.step(put_df['Strike'], put_df['POP-bootstrap'], label='Put Model-POP', color='red', linestyle='-', marker='o')
    plt.title('Calls and Puts: Empirical and Interpolated Survival Function')
    plt.xlabel('Final Values / Strike Market')
    plt.ylabel('Survival Probability')
    plt.legend()
    plt.show()


def overall_price(option_df, type: str):
    if type == 'call':
        #Plottin the Market Price Vs Theorical Price        
        plt.step(option_df['Strike'], option_df['Último'], 'o-', label='Market', color='black')
        plt.step(option_df['Strike'], option_df['MC_Call_Price'], 'o-', label='MC_model', color='blue')
        plt.step(option_df['Strike'], option_df['Black Scholes'], 'o-', label='B&S', color='gray')
        plt.title('Call Price')
        plt.xlabel('Strike')
        plt.ylabel('Call-Prices')
        plt.legend()
        plt.show()

    elif type == 'put':
        # Plot market and MC prices
        plt.step(option_df['Strike'], option_df['Último'], 'o-', label='Market', color='black')
        plt.step(option_df['Strike'], option_df['MC_Put_Price'], 'o-', label='MC', color='red')
        plt.step(option_df['Strike'], option_df['Black Scholes'], 'o-', label='B&S', color='gray')
        plt.title('Put Price')
        plt.xlabel('Strike')
        plt.ylabel('Put-Prices')
        plt.legend()
        plt.show()


def parity_plot(parity_dataframe):
    # Extracting data from the dataframe
    y_axis = parity_dataframe['Parity']
    x_axis = parity_dataframe['Call Delta']
    temperature_heat = parity_dataframe['Normalized Total Volume']   

    # Creating the scatter plot
    plt.figure(figsize=(10, 6))
    plt.scatter(x_axis, y_axis, s=temperature_heat, alpha=1)  # Using size parameter for scatter plot
    
    # Customizing the plot
    plt.title('Parity vs Call Delta', fontsize=16)
    plt.xlabel('Call Delta', fontsize=14)
    plt.ylabel('Parity', fontsize=14)
    plt.colorbar(label='Normalized Volume')
    plt.grid(True)
    
    # Displaying the plot
    plt.show()



def put_call_ratio_plot(parity_dataframe, daily_ratio):
    # Drop rows containing NaN values
    parity_dataframe = parity_dataframe.dropna()
    
    # Extracting data from the dataframe
    y_axis = parity_dataframe['Put Call Ratio']
    x_axis = parity_dataframe['Call Delta']
    temperature_heat = parity_dataframe['Normalized Total Volume']       

    # Creating the scatter plot
    plt.figure(figsize=(10, 6))
    plt.scatter(x_axis, y_axis, s=temperature_heat, alpha=1)  # Using size parameter for scatter plot
    
    # Customizing the plot
    plt.title('Put Call Ratio', fontsize=16)
    plt.xlabel('Call Delta', fontsize=14)
    plt.ylabel('PCR', fontsize=14)
    plt.colorbar(label='Normalized Volume')
    plt.grid(True)
    
    # Annotating the plot with sum_pc_ratio
    if not np.isnan(daily_ratio):
        plt.text(0.5, 0.5, f'Sum of Put/Call Ratio: {daily_ratio:.2f}', 
                 fontsize=12, ha='center', va='center', transform=plt.gca().transAxes)
    else:
        plt.text(0.5, 0.5, 'Invalid Put/Call Ratio', 
                 fontsize=12, ha='center', va='center', transform=plt.gca().transAxes, color='red')
    
    # Adding a horizontal line at y = 0.7
    plt.axhline(y=0.7, color='red', linestyle='--', linewidth=1, label='Neutral')
    
    # Displaying the plot
    plt.legend()  # Add legend to show the label of the horizontal line
    plt.show()






