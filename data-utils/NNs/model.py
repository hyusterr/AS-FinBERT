import torch
import torch.nn as nn
import torch.nn.functional as F


class CNN_model( nn.module ):
    
    def __init__( self, kernel=10, classes=2, embed_size=300 ):

        super( CNN, self ).__init__()
        
        self.embed = nn.embedding( 

