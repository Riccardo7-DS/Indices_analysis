import numpy as np
from sklearn.linear_model import LinearRegression
import torch
import os
from utils.xarray_functions import ndvi_colormap
from analysis.configs.config_models import config_autodime as model_config
from utils.function_clns import init_logging
from analysis import mask_mbe, mask_mse, compute_image_loss_plot
from torch.nn import MSELoss
import torch
import torch.nn.functional as F
import pandas as pd

import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
from utils.xarray_functions import ndvi_colormap
from analysis.configs.config_models import config_ddim as model_config

cmap = ndvi_colormap("diverging")

def calculate_performance_metrics(real, predicted, metric='rmse'):
    if isinstance(real, np.ndarray):
        real = torch.from_numpy(real)
    if isinstance(predicted, np.ndarray):
        predicted = torch.from_numpy(predicted)

    real_flat = real.reshape(-1)
    pred_flat = predicted.reshape(-1)
    
    percentiles = torch.arange(0, 1.1, 0.1)
    boundaries = torch.quantile(real_flat, percentiles)
    
    metric_values, variances = [], []
    
    for i in range(len(percentiles) - 1):
        lower = boundaries[i]
        upper = boundaries[i+1]
        mask = (real_flat >= lower) & (real_flat < upper)
        
        if metric.lower() == 'rmse':
            value = torch.sqrt(torch.mean((real_flat[mask] - pred_flat[mask])**2))

        elif metric.lower() == 'bias':
            value = torch.mean(real_flat[mask] - pred_flat[mask])

        elif metric.lower() == 'mse':
            value = torch.mean((real_flat[mask] - pred_flat[mask])**2)

        elif metric.lower() == 'nse':
            numerator = torch.sum((real_flat[mask] - pred_flat[mask])**2)
            denominator = torch.sum((real_flat[mask] - torch.mean(real_flat[mask]))**2)
            value = 1 - (numerator / denominator)
            value = 1/(2 - value)
            variances.append(denominator.item())
        else:
            raise ValueError("Metric must be either 'rmse', 'nse' or 'bias'")
        
        metric_values.append(value.item())
        
    
    return metric_values, percentiles[:-1], boundaries[:-1]

def plot_performance_vs_percentile_multi_days(df, 
                                              metric, 
                                              prediction_days_list, 
                                              cmap="Greens",
                                              path=None):
    # Create a plot
    fig, ax = plt.subplots(figsize=(8, 4))
    
    # Colors and line styles for each prediction day
    colors = ['blue', 'red', 'violet']
    linestyles = ['-', '--', '-.']
    
    # Create a colormap and normalize it
    cmap = plt.get_cmap(cmap)
    percentiles = sorted(df['percentile'].unique())
    norm = mcolors.Normalize(vmin=0, vmax=len(percentiles)) 
    
    # Add colored background for each percentile range, making it continuous
    # Plot colored background for each percentile range
    for i in range(len(percentiles)):
        ax.axvspan(percentiles[i], percentiles[i] + 0.1, 
                   facecolor=cmap(norm(i)), alpha=0.3)
    
    # Plot the metric values for each prediction day
    for i, prediction_days in enumerate(prediction_days_list):
        # Filter the dataframe for the given metric and prediction_days
        df_filtered = df[(df['metric'] == metric) & (df['prediction_days'] == prediction_days)]
        
        # Extract the required columns
        metric_values = df_filtered['value'].values
        percentile_ranges = df_filtered['percentile'].values
        boundary_values = df_filtered['ndvi_value'].values

        # Plot the metric values for this prediction day
        ax.plot(percentile_ranges, metric_values, marker='o', color=colors[i], 
                linestyle=linestyles[i], label=f'{prediction_days} days', zorder=5)
        
        # Add annotations for each point (only for the first line, as in the original)
        if i == 0:
            for x, y, val in zip(percentile_ranges, metric_values, boundary_values):
                ax.annotate(f'{val:.2f}', (x, y), textcoords="offset points", 
                            xytext=(0,10), ha='center', va='bottom', zorder=6)
    
    # Labels and title
    ax.set_xlabel('Percentile')
    ax.set_ylabel(metric.upper())
    ax.set_title(f'{metric.upper()} as a function of Percentile Range for different prediction days')
    ax.grid(True, zorder=0)
    
    # Ensure smooth x-axis (optional: you can control the tick frequency as needed)
    ax.set_xticks(np.linspace(percentiles[0], percentiles[-1], num=10))  # Smooth tick marks

    # Add a legend to differentiate the days
    ax.legend(title="Prediction Days")
    
    # Display the plot
    plt.tight_layout()
    if path is not None:
        plt.savefig(os.path.join(path, f"{metric}_comparison.png"))
    
    plt.close()

def plot_performance_vs_percentile(metric_values, 
                                   percentile_ranges, 
                                   boundary_values, 
                                   metric,
                                   cmap="Greens"):
    fig, ax = plt.subplots(figsize=(6, 4))
    
    # Create a sequential colormap
    cmap = plt.get_cmap(cmap)
    norm = mcolors.Normalize(vmin=0, vmax=len(percentile_ranges))
    
    # Plot colored background for each percentile range
    for i in range(len(percentile_ranges)):
        ax.axvspan(percentile_ranges[i], percentile_ranges[i] + 0.1, 
                   facecolor=cmap(norm(i)), alpha=0.3)
    
    # Plot the metric values
    ax.plot(percentile_ranges, metric_values, marker='o', color='blue', zorder=5)
    
    # Add annotations
    for i, (x, y, val) in enumerate(zip(percentile_ranges, metric_values, boundary_values)):
        ax.annotate(f'{val:.2f}', (x, y), textcoords="offset points", 
                    xytext=(0,10), ha='center', va='bottom', zorder=6)
    
    ax.set_xlabel('Percentile')
    ax.set_ylabel(metric.upper())
    ax.set_title(f'{metric.upper()} as a function of Percentile Range')
    ax.set_xticks(percentile_ranges)
    ax.grid(True, zorder=0)
    
    # Add colorbar
    sm = plt.cm.ScalarMappable(cmap=cmap, norm=norm)
    sm.set_array([])
    #cbar = fig.colorbar(sm, ax=ax, label='Percentile Range', ticks=range(len(percentile_ranges)))
    #cbar.set_ticklabels([f'{p:.1f}-{p+0.1:.1f}' for p in percentile_ranges])
    
    plt.tight_layout()
    plt.show()


def gaussian_kernel(window_size, sigma):
    """Create a 2D Gaussian kernel."""
    kernel = torch.tensor([-(x - window_size // 2)**2 / float(2 * sigma**2) for x in range(window_size)])
    kernel = torch.exp(kernel)
    kernel = kernel / kernel.sum()  # Normalize the kernel
    kernel_2d = kernel.unsqueeze(1).mm(kernel.unsqueeze(0))  # Create 2D Gaussian
    return kernel_2d.unsqueeze(0).unsqueeze(0)  # Reshape for 4D convolution

def ssim(img1, img2, window_size=11, sigma=1.5, L=2.0):
    """Calculate the SSIM between two images with pixel range [-1, 1]."""
    # Make sure the images are 4D tensors (batch_size, channels, height, width)
    if len(img1.shape) == 3:
        img1 = img1.unsqueeze(0)
    if len(img2.shape) == 3:
        img2 = img2.unsqueeze(0)

    # Set the channel size
    channels = img1.size(1)
    
    # Create a Gaussian window (filter) and apply it to each channel
    window = gaussian_kernel(window_size, sigma).repeat(channels, 1, 1, 1)
    
    # Apply padding
    padding = window_size // 2
    mu1 = F.conv2d(img1, window, padding=padding, groups=channels)
    mu2 = F.conv2d(img2, window, padding=padding, groups=channels)
    
    # Calculate variances and covariances
    mu1_sq = mu1.pow(2)
    mu2_sq = mu2.pow(2)
    mu1_mu2 = mu1 * mu2
    sigma1_sq = F.conv2d(img1 * img1, window, padding=padding, groups=channels) - mu1_sq
    sigma2_sq = F.conv2d(img2 * img2, window, padding=padding, groups=channels) - mu2_sq
    sigma12 = F.conv2d(img1 * img2, window, padding=padding, groups=channels) - mu1_mu2
    
    # Define constants (small values to stabilize division)
    C1 = (0.01 * L) ** 2  # Adjusted for L=2 (range [-1, 1])
    C2 = (0.03 * L) ** 2  # Adjusted for L=2 (range [-1, 1])
    
    # SSIM formula
    ssim_map = ((2 * mu1_mu2 + C1) * (2 * sigma12 + C2)) / ((mu1_sq + mu2_sq + C1) * (sigma1_sq + sigma2_sq + C2))
    
    return ssim_map.mean().item()

def nse_error(pred_flat, real_flat):
    numerator = torch.sum((real_flat - pred_flat)**2)
    denominator = torch.sum((real_flat - torch.mean(real_flat))**2)
    nse_val = 1 - numerator / denominator
    return nse_val.mean().item()

def compute_coefficients(y_pred, y_true):
    s, w, h = y_true.shape
    # Create linear regression model
    model = LinearRegression()
    model.fit(y_pred.reshape( s*w*h).reshape(-1, 1), 
              y_true.reshape(s*w*h).reshape(-1, 1))

    def shift_data(data, intercept, coefficient):
        return data*coefficient[0] + intercept[0]

    # Compute bias
    coefficient = model.coef_  # Coefficients of the model
    intercept = model.intercept_  # Intercept of the model

    print(f"Coefficients: {coefficient}")
    print(f"Intercept: {intercept}")

    return shift_data(y_pred, intercept, coefficient)



class MetricCollection():
    def __init__(self):
        # Initialize dataframes to store single metrics and percentiles
        self.single_metrics = pd.DataFrame(columns=["prediction_days", "metric", "value"])
        self.percentile_metrics = pd.DataFrame(columns=["prediction_days", "metric", "percentile","ndvi_value", "value"])

    def gather_metric(self, day, metric, value):
        # Store single metric with the corresponding day
        new_row = pd.DataFrame({"prediction_days": [day], "metric": [metric], "value": [value]})
        self.single_metrics = pd.concat([self.single_metrics, new_row], ignore_index=True)

    def gather_percentile_metrics(self, day, metric, metric_values, percentile_ranges, boundary_values):
        # Store percentile-based metrics with the corresponding day and boundary
        for i, (metric_value, percentile_range) in enumerate(zip(metric_values, percentile_ranges)):
            new_row = pd.DataFrame({
                "prediction_days": [day],
                "metric": [metric],
                "percentile":[percentile_range.item()],
                "ndvi_value": [boundary_values[i].item()],
                "value": [metric_value]
            })
            self.percentile_metrics = pd.concat([self.percentile_metrics, new_row], ignore_index=True)



def pipeline_compute_metrics():
# Example usage:
    metrics = MetricCollection()
    logger = init_logging()

    basepath = model_config.output_dir + '/dime/days_{}/features_90/images/output_data'


    for days in [10, 15, 30]:
        path = basepath.format(days)
        sample_image = np.load(os.path.join(path,"pred_data.npy"))
        mask = np.load(os.path.join(path,"mask.npy"))
        y_true = np.load(os.path.join(path,"true_data.npy"))
        img_path = os.path.join(path, "../")

        y_bias_corr = compute_coefficients(sample_image, y_true)

        ### Compute MSE spatially-wise and scatterplot
        loss = MSELoss(reduction="none")
        mse = compute_image_loss_plot(y_bias_corr, y_true, mask_mse, mask, False, img_path, cmap, False)
        metrics.gather_metric(days, 'rmse', np.sqrt(mse))

        ### compute mean NSE, SSIM
        img1 = torch.from_numpy(y_bias_corr).unsqueeze(1)
        img2 = torch.from_numpy(y_true).unsqueeze(1)
        ssim_value = ssim(img1, img2)
        nse_value = nse_error(img1, img2)

        print("SSIM for :", round(ssim_value, 4))
        print("NSE:", round(nse_value,4))

        metrics.gather_metric(days, 'ssim', ssim_value)
        metrics.gather_metric(days, 'nse', nse_value)

        ### Compute metrics by percentiles

        for metric in ['rmse', 'nse',"bias"]:
            metric_values, percentile_ranges, boundary_values = calculate_performance_metrics(y_true, 
                                                                                              y_bias_corr, 
                                                                                              metric)
            metrics.gather_percentile_metrics(days, 
                                              metric, 
                                              metric_values, 
                                              percentile_ranges, 
                                              boundary_values)
            # Print results
            # for i, (value, percentile, boundary) in enumerate(zip(metric_values, percentile_ranges, boundary_values)):
            #     print(f"For percentile range {percentile:.1f} to {percentile+0.1:.1f} (value <= {boundary:.2f}), the {metric.upper()} is {value:.4f}")

            # Plot the results
            # plot_performance_vs_percentile(metric_values, percentile_ranges, boundary_values, metric, cmap)



    # At the end, you can access the stored data
    print("Single metrics:\n", metrics.single_metrics)
    print("Percentile metrics:\n", metrics.percentile_metrics)
    print(metrics.single_metrics.sort_values("metric").to_string(index=False))


    for metric in ['rmse', 'nse',"bias"]:
        plot_performance_vs_percentile_multi_days(metrics.percentile_metrics, 
                                                  metric=metric, 
                                                  prediction_days_list=[10, 15, 30],
                                                  cmap=cmap,
                                                  path=img_path)
