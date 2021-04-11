import torch
import torch.nn as nn
from math import floor
import numpy as np
import matplotlib.pyplot as plt



class neighbor_connection:   
    def __init__(self, left, right, top, bottom):
        self.xy, self.hn = self.getXYoffset(left, right, top, bottom)
            
           
    def getXYoffset(self, left, right, top, bottom):
        """Find offsets for each pixel connection, to the center pixel
        """
        xy = []    
        for x in range(-left, right + 1):
            for y in range(-top, bottom + 1):
                xy.append([x,y])
        
        hn = max([left, right, top, bottom])
        
        return xy, hn
    
    def reflect(self):        
        self.xy = [[-x,-y] for x,y in self.xy]
    
    
    def plot_neighbor(self):
        '''
        visualize connectivity kernel (neighbors)
        xy: list of [x_offset, y_offset] of neighbors
        '''
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
    """Convolve an offset filter from makeOffsetConv and convert into pixel-pair connections
    seglabels: should be integer labels, with 0 being background
    This has a 1 if connected, 0 if not, and -1 if undefined"""
    cshape = d_radar.shape
    if len(cshape)==2:
        #if feeding in single segment label image:
        nshape = (1,1,cshape[0],cshape[1])
    elif len(cshape)==4:        
        #This is the usual 4-dimensional representation in a CNN:
        nshape = cshape
    else:
        assert(False)   
    
    d_radar = d_radar.reshape( nshape )
    offsets = cfilter( d_lidar )              #This does zero-padding (implicity makes boundary at size of image.  
    #Could alternatively add 1 before conv and then subtract 1 after to do -1 padding, but not sure if we want this.  Also would need to update otherHalf())
    # connection = -torch.ones(offsets.shape)  
    connection = -torch.ones_like(offsets)    #default are -1 for unknown
        
    # # use relative error or absolute error      
    # thres_error = 1
    # msk_overlap = (d_radar > 0) & (offsets > 0) 
    
    # connection[ ( torch.abs(d_radar - offsets) < thres_error  ) & msk_overlap ] = 1
    # connection[ ( torch.abs(d_radar - offsets) >= thres_error ) & msk_overlap ] = 0
    
    
    rel_error = 0.05
    abs_error = 1
    msk_overlap = (d_radar > 0) & (offsets > 0) 
    
    # note when offsets == 0 
    connection[  msk_overlap & ( torch.abs(d_radar - offsets) < abs_error ) & ( torch.abs(d_radar - offsets)/offsets < rel_error  ) ] = 1
    connection[  msk_overlap & ( (torch.abs(d_radar - offsets) >= abs_error) | ( torch.abs(d_radar - offsets)/offsets >= rel_error ) ) ] = 0
        
    return connection


def depth_to_connect(d_radar, d_lidar, neighbor, device):
    '''
    Compute connectivity map
    input:
        d_lidar, d_radar: h x w; 0 means unknown (or nb x 1 x h x w)
    output:
        connect_map: n_channel x h x w; connect 1, not connect 0, unknown -1  (or nb x nc x h x w)
    ''' 
    
    xy, hn = neighbor.xy, neighbor.hn  
    cfilter = makeOffsetConv(xy, hn).to(device)    
    connected = isConnected(d_radar, d_lidar, cfilter)  
    
    return connected


def cal_nb_depth(d_radar, neighbor, device):
    '''
    get depth in the neighboring region
    
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




if __name__ == '__main__':
    
    
    nb = neighbor_connection(*(3,3,24,8))
    
    
    nb.plot_neighbor()
    
    print(nb.xy)
    
    nb.reflect()
    
    print(nb.xy)
    


