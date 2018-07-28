#%%
# ============================================================================ #
# Pretty print dataframes to output                                            #
# ============================================================================ #
import tabulate
from tabulate import tabulate
def print_df(df):
    print(tabulate(df, headers='keys', tablefmt='psql'))

