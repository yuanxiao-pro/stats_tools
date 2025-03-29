import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')  # 切换到无 GUI 后端
import matplotlib.pyplot as plt
from backtesting import Backtest, Strategy
from backtesting.lib import crossover

class ExponentialSmoothingStrategy(Strategy):
    """
    Exponential Smoothing Trading Strategy
    
    Parameters:
    -----------
    alpha : float
        Smoothing factor (0 < alpha < 1). Higher values mean more weight to recent prices.
    """
    
    def init(self):
        # Precompute the exponential moving averages
        alpha = self.params.alpha
        self.smoothed = self.I(self.exponential_smoothing, self.data.Close, alpha)
        
    def exponential_smoothing(self, series, alpha):
        """
        Perform exponential smoothing on a series.
        
        series: array-like
            The input data to smooth
        alpha: float
            Smoothing factor (0 < alpha < 1)
        """
        result = np.zeros_like(series)
        result[0] = series[0]
        for i in range(1, len(series)):
            result[i] = alpha * series[i] + (1 - alpha) * result[i-1]
        return result
    
    def next(self):
        # If we don't have enough data yet, return
        if len(self.smoothed) < 2:
            return
        
        # Generate signals
        current_price = self.data.Close[-1]
        current_smooth = self.smoothed[-1]
        prev_smooth = self.smoothed[-2]
        
        # Buy signal: price crosses above smoothed line
        if current_price > current_smooth and prev_smooth >= self.data.Close[-2]:
            if not self.position.is_long:
                self.buy()
        
        # Sell signal: price crosses below smoothed line
        elif current_price < current_smooth and prev_smooth <= self.data.Close[-2]:
            if self.position.is_long:
                self.sell()

# Example usage
if __name__ == "__main__":
    # Load sample data (replace with your own data)
    from backtesting.test import GOOG
    data = GOOG
    print("data shape", data.shape)
    # Run backtest
    bt = Backtest(data, ExponentialSmoothingStrategy, commission=.002)
    
    # Optimize alpha parameter
    param_grid = {'alpha': np.linspace(0.05, 0.5, 10)}
    print(param_grid)
    optimization_results = bt.optimize(**param_grid)
    print("optimization_results", optimization_results)

    # Get the best parameters
    best_alpha = optimization_results._strategy_param('alpha')
    print(f"Best alpha: {best_alpha}")
    
    # Run with best parameters
    results = bt.run(alpha=best_alpha)
    print(results)
    
    # Plot the results
    bt.plot()