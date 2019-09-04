import numpy as np
import scipy.sparse as sp
import math

"""
create sparse coo matrix of specified size and density,
filled with gaussian values of given mean and std
diag=False sets the diagonal to zero
"""
def sparse_weights(n_rows,
                   n_columns='same',
                   weight_mean=0,
                   weight_std=1,
                   density=0.1,
                   equal_row_density=True,
                   mask_only=False,
                   diag=False,
                   clip=False,
                   clip_min=-math.inf,
                   clip_max= math.inf
                  ):
    if n_columns=='same':
        n_columns = n_rows

    if equal_row_density:
        # Each neuron should have the same number of dendrites
        # Which means each row of w has the same number of entries
        n_per_row = int(density * n_columns)
        n_total = n_per_row * n_rows
    
        column_index = np.zeros((n_rows,n_per_row))
        if diag:
            choices = n_columns
        for irow in range(n_rows):
            if not(diag):
                if irow<n_columns:
                    choices = np.concatenate((np.arange(irow),np.arange(irow+1,n_columns)))
                else:
                    choices = np.arange(n_columns)
            column_index[irow,:] = np.random.choice(choices, n_per_row, replace=False).flatten()
        column_index = column_index.flatten()
        row_index = np.meshgrid(range(n_per_row),range(n_rows))[1].flatten()
    else:
        w = sp.random(n_rows, n_columns, density)
        n_total = w.size
        row_index = w.row
        column_index = w.col
        column_range = range(n_columns)
        if not(diag):
            d = row_index - column_index
            dindex = np.where(d==0)
            if len(dindex[0])>0:
                for index in np.nditer(dindex):
                    columns_filled = column_index[np.where(row_index==row_index[index])]
                    choices = np.setdiff1d(column_range,columns_filled)
                    if choices.size>0:
                        column_index[index] = np.random.choice(np.setdiff1d(column_range,columns_filled))
                    else:
                        print('Unwanted self connection in sparse_weights()')
        
    if mask_only:
        weights = np.ones(n_total)
    else:
        weights = np.random.normal(weight_mean,weight_std,n_total)                
    
    if clip:
        np.clip(weights,clip_min,clip_max,weights)

    w = sp.coo_matrix((weights,(row_index, column_index)), shape=(n_rows,n_columns))

    return w
