def print_df(df):
    # This function pretty prints a pandas dataframe
    import tabulate
    print(tabulate.tabulate(df, headers='keys', tablefmt='psql'))

def freq_dist(counts, title):
    import seaborn as sns
    import matplotlib.pyplot as plt
    sns.set(style="whitegrid", font_scale=2)
    fig, ax = plt.subplots()
    fd_plot = sns.distplot(counts, bins=40, ax=ax, kde=False,
    color='steelblue').set_title(title)    
    return(fd_plot)

def bar_plot(df, xval, yval, title):
    import seaborn as sns
    import matplotlib.pyplot as plt
    sns.set(style="whitegrid", font_scale=2)
    fig, ax = plt.subplots()
    bp = sns.barplot(x=xval, y=yval, data=df, ax=ax, 
        color='steelblue').set_title(title)    
    return(bp)

def histogram(values, title):
    import seaborn as sns
    import matplotlib.pyplot as plt
    sns.set(style="whitegrid", font_scale=2)
    fig, ax = plt.subplots()
    hist = sns.distplot(values,bins=40, ax=ax, kde=False,
    color='steelblue').set_title(title)    
    return(hist)

