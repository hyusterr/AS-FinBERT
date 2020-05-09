#!bin/python
# -*- coding: UTF-8 -*-

import numpy as np
import torch
import argparse
import pickle
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from tqdm import tqdm
from pytorchtools import EarlyStopping
from args import process_command

# hyperparemeters
arguments = process_command()
epochs        = arguments.epoch
batch_size    = arguments.batch
learning_rate = arguments.lr
weight_decay  = arguments.wd
use_gpu = torch.cuda.is_available()
training_set_path = arguments.tr

# early stopping
patience = 20
early_stopping = EarlyStopping(patience=patience, verbose=True)

class Task3Dataset( Dataset ):

    def __init__( self, X, y ):

        self.X = X
        self.y = y

    def __len__( self ):
        
        return len( self.X )

    def __getitem__( self, idx ):

        return self.X[idx], self.y[idx]


class TextCNN( nn.Module ):

    def __init__( self, embedding=300, channel=10, classes=12, static=True ):

        super( TextCNN, self ).__init__()
        # in_channels, out_channels, kernel_size, stride, padding, bias
        self.embed = nn.Embedding( 100000, embedding )
        self.conv3 = nn.Conv2d( 1, channel, (3, embedding), padding=(1, 0) )
        self.conv4 = nn.Conv2d( 1, channel, (4, embedding), padding=(1, 0) )
        self.conv5 = nn.Conv2d( 1, channel, (5, embedding), padding=(1, 0) )
        self.fc1 = nn.Linear( 3 * channel, classes )
        
        self.static = static
        self.loss = nn.CrossEntropyLoss()
        # kernel_size, stride

    def loss_func( self ):
        return self.loss

    def main_task( self, x ):

        x = x.unsqueeze(1)

        if not self.static:
            x = self.embed( x )

        x3 = F.relu( self.conv3( x ) ).squeeze( 3 )
        x4 = F.relu( self.conv4( x ) ).squeeze( 3 )
        x5 = F.relu( self.conv5( x ) ).squeeze( 3 )
        
        x3 = F.max_pool1d( x3, x3.size( 2 ) ).squeeze( 2 )
        x4 = F.max_pool1d( x4, x4.size( 2 ) ).squeeze( 2 )
        x5 = F.max_pool1d( x5, x5.size( 2 ) ).squeeze( 2 )

        x = torch.cat( ( x3, x4, x5 ), 1 )

        score = self.fc1( x )

        return score

    def forward( self, x ):
        
        y_pred = self.main_task( x )
        
        return y_pred


if __name__ == '__main__':
    
    use_gpu = torch.cuda.is_available()

    # read preprocessed data
    print( 'preparing data...' )
    with open( training_set_path, 'rb' ) as f:
        train_X, train_y, idx_to_label_dict = pickle.load( f )
        f.close()

    with open( '128-task3-test.pickle', 'rb' ) as f:
        test_X, test_y = pickle.load( f )
        f.close()

    with open( '128-task3-val.pickle', 'rb' ) as f:
        val_X, val_y = pickle.load( f )
        f.close()

    print( train_X.dtype )
    print( test_X[:5] )
    train_X, train_y = torch.Tensor( train_X ), torch.Tensor( train_y )
    test_X, test_y   = torch.Tensor( test_X ), torch.Tensor( test_y )
    val_X, val_y     = torch.Tensor( val_X ), torch.Tensor( val_y ) 
    
    train_dataset = Task3Dataset( train_X, train_y )
    train_loader  = DataLoader( train_dataset, batch_size=batch_size, shuffle=True )
  
    val_dataset = Task3Dataset( val_X, val_y )
    val_loader  = DataLoader( val_dataset, batch_size=batch_size, shuffle=True )
    
    test_dataset = Task3Dataset( test_X, test_y )
    test_loader  = DataLoader( test_dataset, batch_size=batch_size, shuffle=True )

    model = TextCNN()
    critirion = nn.CrossEntropyLoss()
    optimizer = optim.Adam( model.parameters(), lr=learning_rate )

    if use_gpu:
        model.cuda()

    train_epoch_loss = []
    train_epoch_acc  = []
    val_epoch_loss   = []
    val_epoch_acc    = []

    print( 'start training...' )

    for epoch in tqdm( range( epochs ) ):

        print( epoch + 1, 'starting...' )

        model.train()

        train_loss = []
        train_acc  = []

        for idx, ( X_, y_ ) in enumerate( train_loader ):

            if use_gpu:
                X_ = X_.cuda()
                y_ = y_.cuda()

            optimizer.zero_grad()
            output = model( X_ )
            loss   = critirion( output, y_.long() )

            loss.backward()
            optimizer.step()

            predict = torch.max( output, 1 )[1]
            acc = np.mean( ( y_ == predict ).cpu().numpy() )
            
            train_acc.append( acc )
            train_loss.append( loss.item() )

        train_epoch_loss.append( (epoch + 1, np.mean(train_loss)) )
        train_epoch_acc.append( (epoch + 1, np.mean(train_acc)) )

        print("Epoch: {}, train Loss: {:.4f}, train Acc: {:.4f}".format(epoch + 1, np.mean(train_loss), np.mean(train_acc)))

        
        model.eval()
        with torch.no_grad():
            valid_loss = []
            valid_acc  = [] 
            
            for idx, ( X_, y_ ) in enumerate( val_loader ):
            
                if use_gpu:
                    X_ = X_.cuda()
                    y_ = y_.cuda()
                
                output = model( X_ )
                loss = critirion( output, y_.long() )
                predict = torch.max( output, 1 )[1]
                    
                acc = np.mean( ( y_ == predict ).cpu().numpy() )
                valid_loss.append( loss.item() )
                valid_acc.append( acc )
                    
            val_epoch_acc.append( ( epoch + 1, np.mean( np.mean( valid_acc ) ) ) )
            val_epoch_loss.append( ( epoch + 1, np.mean( np.mean( valid_loss ) ) ) )
            print("Epoch: {}, valid Loss: {:.4f}, valid Acc: {:.4f}".format(epoch + 1, np.mean(valid_loss), np.mean(valid_acc)))

            early_stopping( np.mean( valid_loss ), model )
                    
            if early_stopping.early_stop:
                print("Early stopping")
                break

    with open( 'history' + training_set_path, 'wb' ) as f:
        pickle.dump( ( (val_epoch_loss, train_epoch_loss), ( val_epoch_acc, train_epoch_loss ) ), f )
        f.close()


    test_pred = []
    test_ans  = []

    model.eval()
    with torch.no_grad():
        
        test_acc = []   
        
        for idx, ( X_, y_ ) in enumerate( test_loader ):
            
            if use_gpu:
                X_ = X_.cuda()
                y_ = y_.cuda()
                
            output = model( X_ )
            loss = critirion( output, y_.long() )
            predict = torch.max( output, 1 )[1]

            acc = np.mean( ( y_ == predict ).cpu().numpy() )
            test_acc.append( acc )

            test_pred += list( predict )
            test_ans  += list( y_ )
                    
        print("Epoch: {}, test acc: {:.4f}".format(epoch + 1, np.mean(test_acc)))


    test_ans  = [ int( i ) for i in test_ans  ]
    test_pred = [ int( i ) for i in test_pred ]


    with open( training_set_path.split('.')[0] + '-test-CNN.pickle', 'wb' ) as f:
        pickle.dump( ( test_ans, test_pred ), f, protocol=4 )
        f.close()
   
    from sklearn.metrics import classification_report
    print( classification_report( test_ans, test_pred, digits=4 ) )
