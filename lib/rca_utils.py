import torch
import torch.nn as nn
import numpy as np
import matplotlib.pyplot as plt


class neighbor_connection:   
    def __init__(self, left, right, top, bottom):
        self.xy, self.hn = self.getXYoffset(left, right, top, bottom)
            
           
    def getXYoffset(self, left, right, top, bottom):
        xy = []    
        for x in range(-left, right + 1):
            for y in range(-top, bottom + 1):
                xy.append([x,y])
        
        hn = max([left, right, top, bottom])
        
        return xy, hn
    
    def reflect(self):        
        self.xy = [[-x,-y] for x,y in self.xy]
        
    def plot_neighbor(self):
        xy = self.xy
        hn = self.hn
       
        M = np.zeros((2*hn + 1, 2*hn + 1), dtype=np.uint8)   
        for x, y in xy:
            x += hn
            y += hn
            M[y,x]=255
        M[hn, hn] = 128
        
        plt.imshow(M, cmap='gray')
        plt.title('%d neighbors' % len(xy))
        plt.show()


def makeOffsetConv(xy, hn):
    """Create a 2D convolution that does a separate offset for
    each element in xy"""
    m = nn.Conv2d(1,len(xy),2*hn+1, padding=hn, bias=False)
    m.weight.data.fill_(0)
    for ind, xyo in enumerate(xy):
        m.weight.data[ind,0,hn+xyo[1],hn+xyo[0]] = 1     # weight size [len(xy), 1, 2*hn+1, 2*hn+1]
    return m


def isConnected(d_radar, d_lidar, cfilter):
    
    cshape = d_radar.shape
    if len(cshape)==2:
        nshape = (1,1,cshape[0],cshape[1])
    elif len(cshape)==4:        
        nshape = cshape
    else:
        assert(False)   
    
    d_radar = d_radar.reshape( nshape )
    offsets = cfilter( d_lidar )            
    connection = -torch.ones_like(offsets)

    rel_error = 0.05
    abs_error = 1
    msk_overlap = (d_radar > 0) & (offsets > 0) 

    connection[  msk_overlap & ( torch.abs(d_radar - offsets) < abs_error ) & ( torch.abs(d_radar - offsets)/offsets < rel_error  ) ] = 1
    connection[  msk_overlap & ( (torch.abs(d_radar - offsets) >= abs_error) | ( torch.abs(d_radar - offsets)/offsets >= rel_error ) ) ] = 0
        
    return connection


def depth_to_connect(d_radar, d_lidar, neighbor, device):
    
    xy, hn = neighbor.xy, neighbor.hn  
    cfilter = makeOffsetConv(xy, hn).to(device)    
    connected = isConnected(d_radar, d_lidar, cfilter)  
    
    return connected


def cal_nb_depth(d_radar, neighbor, device):
    '''
    Get depth in the neighboring region
    
    input:
        d_radar: h x w or n x 1 x h x w
    output:
        nb_depth: 1 x n_nb x h x w
    '''    
    
    if len(d_radar.shape) == 2:
        d_radar = d_radar[None, None, ...]
    
    xy, hn = neighbor.xy, neighbor.hn  
    cfilter = makeOffsetConv(xy, hn).to(device)
    with torch.no_grad():
        nb_depth = cfilter( d_radar )
    
    return nb_depth



def otherHalf(connection, xy):
    """Return other half of connections for each pixel
       Can concatenate this with makeConnections output to get all connections for each pixel, see allConnections()
    """
    assert(len(xy)==connection.shape[1]) #should be one xy offset per connection
    #other = -torch.ones_like(connection)  #if want to say unknown connection to neighbors, need to make sure padding is -1 (rather than zero-padding) in isConnected
    other = torch.zeros_like(connection)
    for ind, xyo in enumerate(xy):
        if xyo[0]==0:
            if xyo[1]==0:
                other[:,ind,:,:] = connection[:,ind,:,:]  #This one is never called as we don't do pixels to each other
            elif xyo[1]<0:
                other[:,ind,:xyo[1],:] = connection[:,ind,-xyo[1]:,:]
            else:
                other[:,ind,xyo[1]:,:] = connection[:,ind,:-xyo[1],:]
        elif xyo[0]<0:
            if xyo[1]==0:
                other[:,ind,:,:xyo[0]] = connection[:,ind,:,-xyo[0]:]
            elif xyo[1]<0:
                other[:,ind,:xyo[1],:xyo[0]] = connection[:,ind,-xyo[1]:,-xyo[0]:]
            else:
                other[:,ind,xyo[1]:,:xyo[0]] = connection[:,ind,:-xyo[1],-xyo[0]:]
        else:
            if xyo[1]==0:
                other[:,ind,:,xyo[0]:] = connection[:,ind,:,:-xyo[0]]
            elif xyo[1]<0:
                other[:,ind,:xyo[1],xyo[0]:] = connection[:,ind,-xyo[1]:,:-xyo[0]]
            else:
                other[:,ind,xyo[1]:,xyo[0]:] = connection[:,ind,:-xyo[1],:-xyo[0]]
    return other

