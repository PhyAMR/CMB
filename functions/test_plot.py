import pandas as pd
import numpy as np
import os
import sys

# Ensure the plots module can be imported
from .plots import CorrelationPlots
from .data import Data_loader

def test_create_histogram_grid_with_different_datasets():
    """
    Tests the create_histogram_grid function by generating two distinct
    datasets (representing S12 and XIV statistics) and plotting them 
    together for visual comparison. This test ensures that the function 
    can correctly overlay different datasets on the same histogram plots.
    """
    # Create two DataFrames with different statistical distributions
    np.random.seed(0)
    DL = Data_loader()
    s12_data = {
        'stat_1': np.random.normal(loc=10, scale=2, size=1000),
        'stat_2': np.random.normal(loc=-4, scale=1.5, size=1000),
        'stat_3': np.random.gamma(shape=3, scale=2.5, size=1000),
    }
    df_s12 = pd.DataFrame(s12_data)

    xiv_data = {
        'stat_1': np.random.normal(loc=0, scale=2.5, size=1000),
        'stat_2': np.random.normal(loc=5, scale=1, size=1000),
        'stat_3': np.random.uniform(low=-6, high=6, size=1000),
    }
    df_xiv = pd.DataFrame(xiv_data)

    # Define the labels, title, and save path for the plot
    labels = ['stat_1', 'stat_2', 'stat_3']
    title = "Test Histogram Grid: S12 (Model) vs. XIV (Data)"
    save_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "test_histogram_comparison.png")

    # Instantiate the plotter and call the method
    plotter = CorrelationPlots(data_loader=DL)
    plotter.create_histogram_grid(
        df=df_s12,
        labels=labels,
        title=title,
        comparison_data=df_s12,
        save_path=save_path,
        bins=25
    )

    print(f"Test plot saved to {save_path}")
    # Verify that the plot file was created
    assert os.path.exists(save_path), f"Plot file was not created at {save_path}"

if __name__ == '__main__':
    test_create_histogram_grid_with_different_datasets()
