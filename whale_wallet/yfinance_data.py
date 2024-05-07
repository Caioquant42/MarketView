import yfinance as yf
import pandas as pd

'''
Coleta dados históricos da biblioteca y_finance

ticker_symbol: PETR4

Valid intervals: 1m, 2m, 5m, 15m, 30m, 60m, 90m, 1h, 1d, 5d, 1wk, 1mo, 3mo

'''

class ydata:
    def __init__(self, ticker_symbol, interval:str = '1d',period:int = 'max',world:bool = False):
        self.ticker_symbol = ticker_symbol
        self.interval = interval
        self.period = period
        self.world = world
        
    def _add_sa_to_tickers(self, tickers):
        if self.world == False:
            return tickers + '.SA'
        else:
            return tickers
    def check_interval(self):
        valid_intervals = ['1m', '2m', '5m', '15m', '30m', '60m', '90m', '1h', '1d', '5d', '1wk', '1mo', '3mo']
        if self.interval not in valid_intervals:
            raise ValueError("Intervalo não disponível, opções válidas: 1m, 2m, 5m, 15m, 30m, 60m, 90m, 1h, 1d, 5d, 1wk, 1mo, 3mo")
        if self.period == 'max':
            if self.interval == '1h':
                self.period = '730d'
            elif self.interval =='1m':
                self.period = '7d'
            elif self.interval in ['1d', '1wk', '1mo', '3mo']:
                self.period = 'max'
            elif self.interval in ['90m', '30m', '15m', '5m']:
                self.period = '60d' 
            else:
                raise ValueError("Erro: Período Inválido.")
            return self.period
        else:
            return self.period

    def get_stock_data(self):
        ticker = self._add_sa_to_tickers(self.ticker_symbol)
        stock_data = yf.Ticker(ticker)
        period = self.check_interval()
        historical_data = stock_data.history(period=period, interval=self.interval)
        rename_cols = ['Abertura', 'Máxima', 'Mínima', 'Fechamento', 'Volume', 'Dividendos', 'Desdobramentos']
        historical_data = historical_data.rename(columns=dict(zip(historical_data.columns, rename_cols)))
        return historical_data




        
    