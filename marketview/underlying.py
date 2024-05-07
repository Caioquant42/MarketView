import pandas as pd
import numpy as np
class Underlying:
    def __init__(self, asset: str):
        self.asset = asset
    def load_data(self, start_date: str = None, end_date: str = None, revert: bool = True, ext: str = 'xlsm') -> pd.DataFrame:
        file_name = f'{self.asset}.{ext}'
        df = pd.read_excel(file_name, index_col='Data')           
        if revert:
            df = df.iloc[::-1]
        if start_date is None:
            return df
        else:
            if end_date is None:
                end_date = df.index[-1].strftime('%Y-%m-%d')
            df = df.loc[start_date:end_date].dropna()
        return df




