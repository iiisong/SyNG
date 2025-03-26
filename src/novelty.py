import pandas as pd

# ephemeral
def ephemeral(data, feature_name, start, duration, func, on_row=False):
    data = data.copy()
    end = start + duration
    if on_row:
        data.loc[start : end, feature_name] = data.loc[start : end].apply(func, axis=1)
    else:
        data.loc[start : end, feature_name] = data.loc[start : end, feature_name].apply(func)
    return data

# persistant
def persistant(data, feature_name, location, func, after=True, on_row=False):
    data = data.copy()
    if after:
        if on_row:
            data.loc[location:, feature_name] = data.loc[location:].apply(func, axis=1)
        else:
            data.loc[location:, feature_name] = data.loc[location:, feature_name].apply(func)
        
    else:
        if on_row:
            data.loc[:location, feature_name] = data.loc[:location].apply(func, axis=1)
        else:
            data.loc[:location, feature_name] = data.loc[:location, feature_name].apply(func)
    
    return data

# periodic
def periodic(data, feature_name, location, duration, period, func, after=True, on_row=False):
    data = data.copy()
    p = location
    
    if after:
        while p < len(data):
            if on_row:
                data.loc[p : p + duration, feature_name] = data.loc[p : p + duration].apply(func, axis=1)
            else:
                data.loc[p : p + duration, feature_name] = data.loc[p : p + duration, feature_name].apply(func)
            p += period
        
    else:
        while p - duration >= 0:
            if on_row:
                data.loc[p - duration: p, feature_name] = data.loc[p - duration: p].apply(func, axis=1)
            else:
                data.loc[p - duration: p, feature_name] = data.loc[p - duration: p, feature_name].apply(func)
            p -= period
    
    return data

# degenerate is a random func