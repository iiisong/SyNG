import plotly.graph_objects as go

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
        x=[t + len(train_data) for t in drift_points],
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