import pandas as pd
from functools import reduce

from river import preprocessing, linear_model, optim, compose

####################################################### FUNCTIONS #######################################################

def generateRegressionDataSet(intercept=[1,0], domain=range(0,500), noise=[0 for x in range(0,500)], columns=['x', 'y']) -> pd.DataFrame:
    '''
    Generate a regression dataset with a linear function and noise.
    
    Parameters:
    intercept : list
        The coefficients of the linear function.
    domain : range
        The range of x values.
    noise : list
        The noise to be added to the y values.
    columns : list
        The names of the columns in the resulting DataFrame.
        
    Returns:
    pd.DataFrame
    '''
    data = []
    for x, single_noise in zip(domain, noise):
        data.append([x, x * intercept[0] + intercept[1]+ single_noise])
    return pd.DataFrame(data, columns=columns)

def mergeDataSetsSudden(dataset1, dataset2) -> pd.DataFrame:
    '''
    Concat two datasets into one.
    
    Parameters:
    dataset1 : pd.DataFrame
        The first dataset.
    dataset2 : pd.DataFrame
        The second dataset.
        
    Returns:
    pd.DataFrame
    '''
    return pd.concat([dataset1, dataset2], ignore_index=True)

def generateRegressionDataSetWithSuddenDrift(intercept, domain=[range(0,500), range(500,1000)], noise=[[0 for x in range(0,500)], [0 for x in range(0,500)]], columns=['x', 'y']):
    '''
    Generate a regression dataset with a sudden drift.
    
    Parameters:
    intercept : list
        The coefficients of the linear function.
    domain : list
        The ranges of x values.
    noise : list
        The noise to be added to the y values.
    columns : list
        The names of the columns in the resulting DataFrame.
        
    Returns:
    pd.DataFrame
    '''
    
    datasets = []
    for d in zip(domain, noise, intercept):
        dataset = generateRegressionDataSet(intercept=d[2], domain=d[0], noise=d[1], columns=columns)
        datasets.append(dataset)
    return reduce(mergeDataSetsSudden, datasets)

def normalizeData(dataset):
    '''
    Normalize the dataset using z-score normalization.
    
    Parameters:
    dataset : pd.DataFrame
        The dataset to be normalized.
        
    Returns:
    pd.DataFrame
    '''
    return (dataset-dataset.mean())/dataset.std()

def make_model_synth(alpha):
    '''
    Create a model for synthetic data.
    
    Parameters:
    alpha : float
        The quantile level for the model.
        
    Returns:
    river.compose.Pipeline
    '''
    scale = preprocessing.StandardScaler()

    learn = linear_model.LinearRegression(
        intercept_lr=0,
        optimizer=optim.Adam(),
        loss=optim.losses.Quantile(alpha=alpha)
    )

    model = scale | learn

    return model

def make_model_real(alpha):
    '''
    Create a model for real data.
    
    Parameters:
    alpha : float
        The quantile level for the model.
    
    Returns:
    river.compose.Pipeline
    '''
    learn = linear_model.LinearRegression(
        intercept_lr=0.1,
        intercept_init=0.01,
        optimizer=optim.Adam(),
        loss=optim.losses.Quantile(alpha=alpha)
    )

    model = compose.Pipeline(
        ('ordinal_date', compose.FuncTransformer(get_ordinal_date)),
        ('scale', preprocessing.StandardScaler()),
        ('lin_reg', learn)
    )

    return model

def get_ordinal_date(x):
    '''
    Convert the date to ordinal format.
    '''
    return {'ordinal_date': x['Time'].toordinal()}

