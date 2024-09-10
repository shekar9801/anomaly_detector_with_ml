import numpy as np
import pandas as pd
import warnings 
warnings.filterwarnings('ignore')
import matplotlib.pyplot as plt
pd.set_option('display.max_columns', None)
class data_preprocessing:
    
    def __init__(self, data_file_path, float_columns):
        self.data_file_path = data_file_path
        self.float_columns = float_columns
        
    
    def data_loading(self):
        try:
            data = pd.read_csv(self.data_file_path)
            print('data has been loaded....')
        except Exception as e:
            print('error in loading the data :', e)
            
        return data
    
    
    def data_cleaning(self):
        try:
            data = self.data_loading()
            # if data.isnull().values.any():
            #     data[self.float_columns].fillna(data[self.float_columns].mean(), inplace=True)
            data.fillna(method='ffill', inplace=True)
                
            data['time'] = pd.to_datetime(data['time'])
            non_numeric_rows = data[self.float_columns].apply(pd.to_numeric, errors='coerce')
            non_numeric_rows = data[self.float_columns][non_numeric_rows.isnull().any(axis=1)]
            data[self.float_columns] = data[self.float_columns].apply(pd.to_numeric, errors='coerce')
            data = data.dropna(subset=self.float_columns)
            data[self.float_columns] = data[self.float_columns].astype(np.float64)
            print('data cleaning is completed.....')
        except Exception as e:
            print('Exception is:', e)
            
        return data
        
if __name__ == '__main__':
    
    data_file_path = r".\datasets\data.csv"
    float_columns = ['Cyclone_Inlet_Gas_Temp', 'Cyclone_Gas_Outlet_Temp', 'Cyclone_Outlet_Gas_draft', 
                     'Cyclone_cone_draft', 'Cyclone_Inlet_Draft', 'Cyclone_Material_Temp']
    
    dp = data_preprocessing(data_file_path, float_columns)
    data = dp.data_cleaning()
    print(data.head())
    
    
        
        
        
        
        
