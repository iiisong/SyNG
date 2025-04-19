import plotly.graph_objects as go

import numpy as np
import matplotlib.pyplot as plt
from matplotlib import gridspec
import sdv
from sdv.single_table import GaussianCopulaSynthesizer
from sdv.metadata import SingleTableMetadata
import pandas as pd

def plot_drift(feature_name, train_data, test_data, drift_points=[]):
    # Create the figure
    fig = go.Figure()

    # Add train data line
    fig.add_trace(go.Scatter(
        x=list(range(len(train_data[feature_name]))),
        y=train_data[feature_name],
        mode='lines+markers',
        name='Training Data',
        line=dict(color='blue'),
        marker=dict(symbol='circle'),
        showlegend=True,
    ))
    
    # add test_data line
    fig.add_trace(go.Scatter(
        x=list(range(len(train_data), len(train_data) + len(test_data[feature_name]))),
        y=test_data[feature_name],
        mode='lines+markers',
        name='Testing Data',
        line=dict(color='green'),
        marker=dict(symbol='circle'),
        showlegend=True,
    ))
    
    # Add drift points
    fig.add_trace(go.Scatter(
        x=[i + len(train_data) for i in drift_points],
        y=[test_data[feature_name][i] for i in drift_points],
        mode='markers',
        name='Drift Detected',
        marker=dict(color='red', size=10),
        showlegend=True,
    ))

    # Update layout
    fig.update_layout(
        title="Data with Drift Detection",
        xaxis_title="Sample Index",
        yaxis_title=feature_name,
        legend=dict(x=0, y=1),
    )

    # Show the plot
    fig.show()
    
# Auxiliary function to plot the data
def plot_data(stream, dist_a, dist_b, drifts=None):
   fig = plt.figure(figsize=(7,3), tight_layout=True)
   gs = gridspec.GridSpec(1, 2, width_ratios=[3, 1])
   ax1, ax2 = plt.subplot(gs[0]), plt.subplot(gs[1])
   ax1.grid()
   ax1.plot(stream, label='Stream')
   ax2.grid(axis='y')
   ax2.hist(dist_a, label=r'$dist_a$')
   ax2.hist(dist_b, label=r'$dist_b$')
#    ax2.hist(dist_c, label=r'$dist_c$')
   if drifts is not None:
       for drift_detected in drifts:
           ax1.axvline(drift_detected, color='red')
   plt.show()
  
def generate_synthetic_from_df(df, n_samples=500):
    # Define metadata for SDV
    metadata = SingleTableMetadata()
    metadata.detect_from_dataframe(data=df)
    
    # Fit SDV model
    model = GaussianCopulaSynthesizer(metadata)
    model.fit(df)
    
    # Sample new data
    synthetic_df = model.sample(n_samples)

    return synthetic_df