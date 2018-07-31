def describe(df, var):
    '''
    This function expands the pandas describe method by adding 
    missing values information.
    '''    
    d = var.astype(str).describe().to_frame().T
    d['observations'] = df.shape[0]
    d['missing'] = var.isnull().sum()
    d['pct_missing'] = d['missing'] * 100 / d['observations']
    cols = ['observations', 'count', 'missing', 'pct_missing', 'unique', 'top', 'freq']
    d = d[cols]
    return(d)