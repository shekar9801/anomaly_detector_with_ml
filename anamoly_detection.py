import os
import warnings 
import pandas as pd
import seaborn as sns  
warnings.filterwarnings('ignore')
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
from data_preprocessing import data_preprocessing


class anamoly_detection:
    
    def __init__(self, data_file_path, float_columns, data, plots_path) -> None:
        self.data_file_path = data_file_path
        self.float_columns = float_columns
        self.data = data
        self.plots_path = plots_path
        
    def line_plot(self):
        print('plotting the line plots.....')
        for column in self.float_columns:
            plt.figure(figsize=(30, 5))
            plt.plot(data['time'], data[column], label=column)
            plt.title(column)
            plt.xlabel('Timestamp')
            plt.ylabel(column)
            plt.legend()
            column = f'_line_plot_{column}.png'
            path = os.path.join(self.plots_path, column)
            plt.savefig(path, dpi=300, bbox_inches='tight')
            plt.close()
                
    
    def scatter_plot(self):
        print('plotting the scatter plots.....')
        for column in self.float_columns:
            plt.figure(figsize=(30, 5))
            plt.scatter(data['time'], data[column], label=column)
            plt.title(column)
            plt.xlabel('Timestamp')
            plt.ylabel(column)
            plt.legend()
            column = f'_scatter_plot_{column}.png'
            path = os.path.join(self.plots_path, column)
            plt.savefig(path, dpi=300, bbox_inches='tight')
            plt.close()
            
    
    
    def pdf_plots(self):
        print('plotting the pdf plots.....')
        for column in self.float_columns:
            plt.figure(figsize=(30, 5))  
            sns.kdeplot(data[column], label=column, fill=True)  
            plt.title(f'Probability Density Function (PDF) of {column}')  
            plt.xlabel(column)  
            plt.ylabel('Density')  
            plt.legend()  
            column = f'_pdf_plot_{column}.png'
            path = os.path.join(self.plots_path, column)
            plt.savefig(path, dpi=300, bbox_inches='tight')
            plt.close()
            
    def standardize_data(self):
        scaler = StandardScaler()
        df_scaled = scaler.fit_transform(data[self.float_columns])
        df_scaled = pd.DataFrame(df_scaled, columns=self.float_columns)
        print('data standardization is over....')
        return df_scaled
    
    
    def detect_anomaly(self):
        from sklearn.ensemble import IsolationForest
        iso_forest = IsolationForest(contamination=0.01, random_state=42)
        df_scaled = self.standardize_data()
        data['anomaly'] = iso_forest.fit_predict(df_scaled)
        data['anomaly'] = data['anomaly'].apply(lambda x: 1 if x == -1 else 0)
        data.to_csv(r'.\datasets\anamoly_added_data.csv', index=False)
        print('anamoly detection is over......')
        print(data['anomaly'].value_counts())
        
        return data
    
    def anamoly_plot(self):
        data =  self.detect_anomaly()
        data.set_index('time', inplace=True)
        
        plt.figure(figsize=(50, 8))  
        plt.plot(data.index, data['Cyclone_Inlet_Gas_Temp'], label='Cyclone_Inlet_Gas_Temp')

        plt.scatter(data[data['anomaly'] == 1].index, 
                    data[data['anomaly'] == 1]['Cyclone_Inlet_Gas_Temp'], 
                    color='red', label='Anomalies', s=30, marker='x')

        plt.title('Cyclone Inlet Gas Temperature with Anomalies Detected Over Time')
        plt.xlabel('Timestamp') 
        plt.ylabel('Cyclone Inlet Gas Temperature')
        plt.legend()
        column = f'anamoly_detected_time_period_plot.png'
        path = os.path.join(self.plots_path, column)
        plt.savefig(path, dpi=300, bbox_inches='tight')
        plt.close()
        print('Plotting the final anamoly identified plot....')
        

                            
        
if __name__ == '__main__':
    
    plots_path = r'.\plots'
    data_file_path = r".\datasets\data.csv"
    float_columns = ['Cyclone_Inlet_Gas_Temp', 'Cyclone_Gas_Outlet_Temp', 'Cyclone_Outlet_Gas_draft', 
                     'Cyclone_cone_draft', 'Cyclone_Inlet_Draft', 'Cyclone_Material_Temp']

    dp = data_preprocessing(data_file_path, float_columns)
    data = dp.data_cleaning()
    
    ad = anamoly_detection(data_file_path, float_columns, data, plots_path)
    ad.line_plot()
    ad.scatter_plot()
    ad.pdf_plots()
    print(data.head())
    ad.anamoly_plot()


    

