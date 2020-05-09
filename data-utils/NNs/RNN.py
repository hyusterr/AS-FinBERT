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
from pytorchtools import EarlyStopping
from tqdm import tqdm
from args import process_command

# parameters
# hyperparemeters
arguments = process_command()
epochs        = arguments.epoch
batch_size    = arguments.batch
learning_rate = arguments.lr
weight_decay  = arguments.wd
use_gpu = torch.cuda.is_available()
training_set_path = arguments.tr
patience = 20
hidden_dim = 256
input_dim  = 300
n_layers = 2
output_dim = 12
early_stopping = EarlyStopping(patience=patience, verbose=True)


class TextRNN(nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim, n_layers, drop_prob=0.0):
        super(TextRNN, self).__init__()
        self.hidden_dim = hidden_dim
        self.n_layers = n_layers

        self.lstm = nn.LSTM( input_dim, hidden_dim, n_layers, batch_first=True, dropout=drop_prob )
        self.gru = nn.GRU(input_dim, hidden_dim, n_layers, batch_first=True, dropout=drop_prob ) # , bidirectional=True)
        self.fc = nn.Linear(hidden_dim, output_dim)
        self.relu = nn.ReLU()
        
    def forward(self, x):
        out, h = self.lstm(x, None)  
        out = self.fc( self.relu(out[:,-1] ) )
        return out, h
                                                                                                            
    def init_hidden(self, batch_size):
        weight = next(self.parameters()).data
        hidden = weight.new(self.n_layers, batch_size, self.hidden_dim).zero_().cuda()
        return hidden


class Task3Dataset( Dataset ):

    def __init__( self, X, y ):

        self.X = X
        self.y = y

    def __len__( self ):
        
        return len( self.X )

    def __getitem__( self, idx ):

        return self.X[idx], self.y[idx]


if __name__ == '__main__':
    
    use_gpu = torch.cuda.is_available()

    # read preprocessed data
    print( 'preparing data...' )
    with open( training_set_path, 'rb' ) as f:
        train_X, train_y, idx_to_label_dict = pickle.load( f )
        f.close()

    with open( 'task3-test.pickle', 'rb' ) as f:
        test_X, test_y = pickle.load( f )
        f.close()

    with open( 'task3-val.pickle', 'rb' ) as f:
        val_X, val_y = pickle.load( f )
        f.close()

    train_X, train_y = torch.Tensor( train_X ), torch.Tensor( train_y )
    test_X, test_y   = torch.Tensor( test_X ), torch.Tensor( test_y )
    val_X, val_y     = torch.Tensor( val_X ), torch.Tensor( val_y ) 
    
    train_dataset = Task3Dataset( train_X, train_y )
    train_loader  = DataLoader( train_dataset, batch_size=batch_size, shuffle=True )
  
    val_dataset = Task3Dataset( val_X, val_y )
    val_loader  = DataLoader( val_dataset, batch_size=batch_size, shuffle=True )
    
    test_dataset = Task3Dataset( test_X, test_y )
    test_loader  = DataLoader( test_dataset, batch_size=batch_size, shuffle=True )

    model = TextRNN( input_dim, hidden_dim, output_dim, n_layers )
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
            output, h = model( X_ )
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
                
                output, h = model( X_ )
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

    model.eval()

    test_ans  = []
    test_pred = []

    with torch.no_grad():
        
        test_acc = []   
        
        for idx, ( X_, y_ ) in enumerate( test_loader ):
            
            if use_gpu:
                X_ = X_.cuda()
                y_ = y_.cuda()
                
            output, h = model( X_ )
            loss = critirion( output, y_.long() )
            predict = torch.max( output, 1 )[1]
            
            test_pred += list(predict)
            test_ans  += list(y_)
                    
            acc = np.mean( ( y_ == predict ).cpu().numpy() )
            test_acc.append( acc )
                    
        print("Epoch: {}, test acc: {:.4f}".format(epoch + 1, np.mean(test_acc)))

    
    test_ans  = [ int( i ) for i in test_ans  ]
    test_pred = [ int( i ) for i in test_pred ]


    with open( training_set_path.split('.')[0] + '-test-RNN.pickle', 'wb' ) as f:
        pickle.dump( ( test_ans, test_pred ), f, protocol=4 )
        f.close()
   
    from sklearn.metrics import classification_report
    print( classification_report( test_ans, test_pred, digits=4 ) )
