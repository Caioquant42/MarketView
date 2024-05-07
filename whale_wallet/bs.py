import numpy as np
from scipy.stats import norm
import QuantLib as ql
from QuantLib import Date, EuropeanExercise
import pandas as pd
import datetime
class BlackScholes:
    def __init__(self, option_df, option_type):
        self.option_df = option_df
        self.option_type = option_type

    def _parse_date(self,string_date):
        date_tuple = datetime.datetime.strptime(string_date, '%Y-%m-%d').timetuple()
        return ql.Date(*date_tuple[:3])

    def black_scholes_with_greeks(self, spot_price, maturity, volatility, interest_rate, dividend_yield=0.0, b:float = None):
        """
        Calculate the Black-Scholes option price and Greeks for each option in the dataframe.

        Parameters:
        - spot_price: current stock price
        - maturity: maturity date of the options (in QuantLib.Date format)
        - volatility: annualized volatility of the underlying stock
        - interest_rate: risk-free interest rate (annualized)
        - dividend_yield: dividend yield of the underlying stock (default is 0.0)

        Returns:
        - DataFrame: DataFrame with option prices and Greeks added as new columns
        """
        expiration_date = datetime.datetime.strptime(maturity, '%Y-%m-%d').date()        
        # Calculate time to maturity in years
        self.option_df['Time to Maturity'] = ((expiration_date - datetime.date.today()).days + 1) / 365
        
        # Iterate over rows in the dataframe
        for index, row in self.option_df.iterrows():
            strike_price = row['Strike']
            time_to_maturity = row['Time to Maturity']
            option_price, delta, gamma, theta, vega, rho, pop = self._black_scholes_option_price_and_greeks(
                option_type = self.option_type , underlying_price = spot_price , strike_price = strike_price, maturity = time_to_maturity, risk_free_rate = interest_rate , volatility = volatility)
            # Update dataframe with calculated option parameters
            self.option_df.at[index, 'Option Price BS'] = option_price
            self.option_df.at[index, 'Delta2'] = delta
            self.option_df.at[index, 'Gamma2'] = gamma
            self.option_df.at[index, 'Theta2'] = theta
            self.option_df.at[index, 'Vega2'] = vega
            self.option_df.at[index, 'Rho2'] = rho
            self.option_df.at[index, 'POP'] = pop

            # Calculate Implied Volatility From Last option price.
            #implied_vol = self.implied_volatility(option_last_price,spot_price,strike_price,interest_rate,time_to_maturity)
            #self.option_df.at[index, 'Implied Vol BS'] = implied_vol

        return self.option_df

    def _black_scholes_option_price_and_greeks(self, option_type, underlying_price, strike_price, maturity, risk_free_rate, volatility):
        # Set evaluation date
        valuation_date = ql.Date.todaysDate()
        ql.Settings.instance().evaluationDate = valuation_date

        # Define the option type
        option_type_map = {'call': ql.Option.Call, 'put': ql.Option.Put}
        option_type = option_type_map[option_type.lower()]
         
        # Define the option
        payoff = ql.PlainVanillaPayoff(option_type, strike_price)
        maturity_date = datetime.datetime.strptime(maturity, '%Y-%m-%d').date()
        maturity_date = maturity_date.strftime('%Y-%m-%d')
        exercise = ql.EuropeanExercise(maturity_date)
        option = ql.VanillaOption(payoff, exercise)

        # Define the market data
        spot_handle = ql.QuoteHandle(ql.SimpleQuote(underlying_price))
        flat_ts = ql.YieldTermStructureHandle(ql.FlatForward(valuation_date, ql.QuoteHandle(ql.SimpleQuote(risk_free_rate)), ql.Actual360()))
        flat_vol_ts = ql.BlackVolTermStructureHandle(ql.BlackConstantVol(valuation_date, ql.NullCalendar(), ql.QuoteHandle(ql.SimpleQuote(volatility)), ql.Actual360()))
        bsm_process = ql.BlackScholesProcess(spot_handle, flat_ts, flat_vol_ts)

        # Calculate option price and Greeks using Black-Scholes formula
        engine = ql.AnalyticEuropeanEngine(bsm_process)
        option.setPricingEngine(engine)
        
        # Option price
        option_price = option.NPV()
        
        # Delta
        delta = option.delta()
        
        # Gamma
        gamma = option.gamma()
        
        # Vega
        vega = option.vega() / 100
        
        # Theta
        theta = option.theta() / 365
        
        # Rho
        rho = option.rho() / 100
        
        # Probability of exercise
        d1 = (ql.blackFormula(option_type, underlying_price, strike_price, volatility, (maturity - valuation_date) / 365.0))**(-1)*(ql.blackFormula(option_type, underlying_price, strike_price, volatility, (maturity - valuation_date) / 365.0))
        d2 = d1 - (volatility * np.sqrt((maturity - valuation_date) / 365.0))
        if option_type == ql.Option.Call:
            prob_exercise = norm.cdf(d2)
        else:
            prob_exercise = norm.cdf(-d2)
        
        return option_price, delta, gamma, vega, theta, rho, prob_exercise







