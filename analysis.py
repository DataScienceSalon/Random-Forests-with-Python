def describe(df, var):
    '''
    This function expands the pandas describe method by adding 
    missing values information.
    '''    
    import numpy as np
    d = var.astype(str).describe().to_frame().T
    d['observations'] = df.shape[0]
    d['missing'] = var.isnull().sum()
    d['pct_missing'] = d['missing'] * 100 / d['observations']
    if np.issubdtype(var.dtype, np.datetime64):
        d['start'] = min(var).strftime('%Y-%m-%d')
        d['end'] = max(var).strftime('%Y-%m-%d')
        cols = ['observations', 'count', 'missing', 'pct_missing', 'unique', 'start', 'end', 'top', 'freq'] 
        d = d[cols]
    else:
        cols = ['observations', 'count', 'missing', 'pct_missing', 'unique', 'top', 'freq']
        d = d[cols]
    return(d)