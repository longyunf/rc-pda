'''
Pyramid network 

'''
import torch
import torch.nn as nn
import torch.nn.functional as F



__all__ = ['PyramidCNN', 'pnet']

class Block(nn.Module):

    def __init__(self, nChannels, doRes=False, doBN=False, doELU=False):
        super(Block, self).__init__()

        if doBN:
            self.bn1 = nn.BatchNorm2d(nChannels)
            self.bn2 = nn.BatchNorm2d(nChannels)
        else:
            self.bn1 = []
            self.bn2 = []
        self.conv1 = nn.Conv2d(nChannels, nChannels, kernel_size=3, padding=1, bias=True)
        self.conv2 = nn.Conv2d(nChannels, nChannels, kernel_size=3, padding=1, bias=True)
        if doELU:
            self.relu = nn.ELU()
        else:
            self.relu = nn.ReLU()
        self.doRes = doRes

    def forward(self, x):

        out = x

        if self.bn1:
            out = self.bn1(out)
        out = self.relu(out)
        out = self.conv1(out)

        if self.bn2:
            out = self.bn2(out)
        out = self.relu(out)
        out = self.conv2(out)

        if self.doRes:
            out += x

        return out

def NetBlock(nPerBlock, nChannels, doRes, doBN, doELU):
        layers = []
        for _ in range(nPerBlock):
            layers.append(Block(nChannels, doRes, doBN, doELU))
        return nn.Sequential(*layers)


def PredictBlock(nChannels, outChannels, doBN, doELU):
        layers = []

        if doBN:
            layers.append( nn.BatchNorm2d(nChannels) )
        if doELU:
            layers.append( nn.ELU() )
        else:
            layers.append( nn.ReLU() )
        layers.append( nn.Conv2d(nChannels, outChannels, kernel_size=3, padding=1, bias=True) )
        return nn.Sequential(*layers)

def PredictBox( nChannels, doBN, doELU):
    layers = []

    if doBN:
        layers.append( nn.BatchNorm2d(nChannels) )
    if doELU:
        layers.append( nn.ELU() )
    else:
        layers.append( nn.ReLU() )
    layers.append( nn.Conv2d(nChannels, 4, kernel_size=3, padding=1, bias=True) )
    layers.append( nn.ReLU() )  #Ensures always predicts value >= 0
    return nn.Sequential(*layers)

def PredictPixels( nChannels, doBN, doELU):
    layers = []

    if doBN:
        layers.append( nn.BatchNorm2d(nChannels) )
    if doELU:
        layers.append( nn.ELU() )
    else:
        layers.append( nn.ReLU() )
    layers.append( nn.Conv2d(nChannels, 2, kernel_size=3, padding=1, bias=True) )
    return nn.Sequential(*layers)

class PyramidCNN(nn.Module):
    def __init__(self, nLevels, nPred, nPerBlock, nChannels, inChannels, outChannels, doRes, doBN, doELU, predPix, predBoxes):
        super(PyramidCNN, self).__init__()
        assert( nLevels > 1 )
        #assert( nPred > 0 )
        self.nLevels = nLevels
        self.nPred = nPred
        self.predBoxes = predBoxes
        self.predPix = predPix
        if inChannels!=nChannels:
            self.inConv = nn.Conv2d(inChannels, nChannels, kernel_size=3, padding=1, bias=True)
        else:
            self.inConv = []
        self.blocksUp = nn.ModuleList()
        self.blocksDown = nn.ModuleList()
        self.blocksPred = nn.ModuleList()
        if self.predBoxes:
            self.predictBox = PredictBox( nChannels, doBN, doELU )
        if self.predPix:
            self.predictPix = PredictPixels( nChannels, doBN, doELU)

        for _ in range(nLevels-1):
            self.blocksUp.append( NetBlock( nPerBlock, nChannels, doRes, doBN, doELU) )
        for _ in range(nLevels):
            self.blocksDown.append( NetBlock( nPerBlock, nChannels, False, doBN, doELU) )  #no res-blocks going down
        for _ in range(nPred):
            self.blocksPred.append( PredictBlock( nChannels, outChannels, doBN, doELU) )

    def _addLevel(self, x, level):
        #recursively add levels until level = nLevels-1
        #first downsample:
        x = F.max_pool2d(x, 2, stride=2)
        
        #top level only have blocksDown, no blocksUp:
        if level < self.nLevels-1:
            x = self.blocksUp[level](x)
            y, out = self._addLevel( x, level + 1)
            x = x + y
        x = self.blocksDown[level](x)
        #then upsample:
        w = F.interpolate( x, scale_factor=2, mode='nearest')

        #Prediction outputs:
        if level < self.nPred:  #Then output at this level
            z = self.blocksPred[level](x)
            out = [z] + out  #prepend and always create a list
        else:
            out = []
        
        return w, out

            
    def forward(self, x):

        if self.inConv:
            x = self.inConv( x )
        
        level = 0
        x = self.blocksUp[level](x)
        y, out = self._addLevel( x, level + 1)
        z = x + y
        w = self.blocksDown[level](z)
        if level < self.nPred:  #Then output at this level
            z = self.blocksPred[level](w)
            out = [z] + out  #prepend and always create a list

        if self.predBoxes:
            out.append( self.predictBox(w) )   #predict boxes -- final element in output list

        if self.predPix:
            out.append( self.predictPix(w) )   #predict leaves -- final element in output list

        return out



def pnet(**kwargs):
    model = PyramidCNN( nLevels     = kwargs['nLevels'],
                        nPred        = kwargs['nPred'],
                        nPerBlock   = kwargs['nPerBlock'], 
                        nChannels    = kwargs['nChannels'], 
                        inChannels  = kwargs['inChannels'],
                        outChannels = kwargs['outChannels'],
                        doRes       = kwargs['doRes'],
                        doBN        = kwargs['doBN'],
                        doELU       = kwargs['doELU'],
                        predPix    = kwargs['predPix'],
                        predBoxes     = kwargs['predBoxes'] )
    return model
