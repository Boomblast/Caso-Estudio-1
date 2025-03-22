import pandas as pd
import numpy as np
from prophet import Prophet
from typing import Union, Optional
import matplotlib.pyplot as plt
import seaborn as sns

class TimeSeriesAnalyzer:
    def __init__(self, data: Union[str, pd.DataFrame], date_column: str):
        """
        Initialize the TimeSeriesAnalyzer with data and date column name.
        
        Args:
            data: Either a path to a CSV file or a pandas DataFrame
            date_column: Name of the column containing dates
        """
        if isinstance(data, str):
            self.data = pd.read_csv(data)
        else:
            self.data = data.copy()
            
        self.date_column = date_column
        self.data[date_column] = pd.to_datetime(self.data[date_column])
        self.data.set_index(date_column, inplace=True)
        
    def get_statistical_summary(self) -> pd.DataFrame:
        """
        Generate a statistical summary of the time series data.
        
        Returns:
            DataFrame containing statistical measures
        """
        summary = self.data.describe()
        summary.loc['missing_values'] = self.data.isnull().sum()
        summary.loc['unique_values'] = self.data.nunique()
        return summary
    
    def plot_time_series(self, columns: Optional[list] = None):
        """
        Plot the time series data.
        
        Args:
            columns: List of columns to plot. If None, plots all numeric columns.
        """
        if columns is None:
            columns = self.data.select_dtypes(include=[np.number]).columns
            
        plt.figure(figsize=(15, 8))
        for col in columns:
            plt.plot(self.data.index, self.data[col], label=col)
            
        plt.title('Time Series Plot')
        plt.xlabel('Date')
        plt.ylabel('Value')
        plt.legend()
        plt.grid(True)
        plt.show()
        
    def forecast(self, target_column: str, periods: int = 30, 
                changepoint_prior_scale: float = 0.05) -> pd.DataFrame:
        """
        Generate forecasts using Facebook Prophet.
        
        Args:
            target_column: Name of the column to forecast
            periods: Number of periods to forecast
            changepoint_prior_scale: Flexibility of the trend (higher = more flexible)
            
        Returns:
            DataFrame containing the forecast
        """
        # Prepare data for Prophet
        prophet_data = pd.DataFrame({
            'ds': self.data.index,
            'y': self.data[target_column]
        }).reset_index(drop=True)
        
        # Fit Prophet model
        model = Prophet(changepoint_prior_scale=changepoint_prior_scale)
        model.fit(prophet_data)
        
        # Make future dataframe
        future = model.make_future_dataframe(periods=periods)
        
        # Generate forecast
        forecast = model.predict(future)
        
        return forecast
    
    def plot_forecast(self, target_column: str, periods: int = 30):
        """
        Plot the original data and forecast.
        
        Args:
            target_column: Name of the column to forecast
            periods: Number of periods to forecast
        """
        forecast = self.forecast(target_column, periods)
        
        plt.figure(figsize=(15, 8))
        plt.plot(self.data.index, self.data[target_column], 
                label='Historical Data', color='blue')
        plt.plot(forecast['ds'], forecast['yhat'], 
                label='Forecast', color='red', linestyle='--')
        plt.fill_between(forecast['ds'], 
                        forecast['yhat_lower'], 
                        forecast['yhat_upper'], 
                        color='red', alpha=0.1)
        
        plt.title(f'Time Series Forecast for {target_column}')
        plt.xlabel('Date')
        plt.ylabel('Value')
        plt.legend()
        plt.grid(True)
        plt.show()

# Example usage
if __name__ == "__main__":
    # Example with the mock_kaggle.csv dataset
    analyzer = TimeSeriesAnalyzer("mock_kaggle.csv", "data")
    
    # Get statistical summary
    print("\nStatistical Summary:")
    print(analyzer.get_statistical_summary())
    
    # Plot time series
    analyzer.plot_time_series()
    
    # Generate and plot forecast for 'venda' column
    analyzer.plot_forecast('venda', periods=30) 