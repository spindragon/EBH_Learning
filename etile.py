import numpy as np

def etile(data):
    ndims = data.ndim
    if (ndims < 3) or (ndims > 4):
        print('etile: data must be 3 or 4 dimensional')
        return -1
    if ndims == 4:
        [a,b,c,d] = data.shape
        out = np.empty([a*c,b*d])
        for ia in np.arange(a):
            yoffset = ia * c
            for ib in np.arange(b):
                xoffset = ib * d
                out[yoffset:yoffset+c,xoffset:xoffset+d] = np.squeeze(data[ia,ib,:,:])
    else:
        [nim,c,d] = data.shape
        zrows = np.arange(1,1+np.floor(np.sqrt(nim)))
        a = np.int(zrows[np.max(np.argwhere(np.mod(nim,zrows)==0))])
        b = np.int(nim / a)
        out = np.empty([a*c,b*d])
        for ia in np.arange(a):
            yoffset = ia * c
            for ib in np.arange(b):
                xoffset = ib * d
                out[yoffset:yoffset+c,xoffset:xoffset+d] = data[ia*b+ib,:,:]
    return out

