import random, os
import numpy as np
import math
import time
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F

def seed_everything(seed: int):   
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = True

seed_everything(42)

class CreditsRNN_LAST_HIDDEN(nn.Module):
    def __init__(self, features, embedding_projections, rnn_units=128, top_classifier_units=32, dropout=0.1, spatial_dropout=0.05):
        super(CreditsRNN_LAST_HIDDEN, self).__init__()
        
        self._credits_cat_embeddings = nn.ModuleList([self._create_embedding_projection(*embedding_projections[feature]) 
                                                      for feature in features])
        self._spatial_dropout = nn.Dropout2d(spatial_dropout)
        self._credits_cat_embeddings_dim = sum([embedding_projections[x][1] for x in features])
       
        self._gru = nn.GRU(input_size=self._credits_cat_embeddings_dim, hidden_size=rnn_units, batch_first=True, bidirectional=True)

        self._hidden_size = rnn_units
        self._flatten = nn.Flatten()
            
        pooling_result_dimension = self._hidden_size * 2 

        self._fc = nn.Linear(self._hidden_size * 2, 256)
            
        self._top_classifier = nn.Linear(in_features=pooling_result_dimension*2 + pooling_result_dimension, out_features=top_classifier_units)
            
        self._intermediate_activation = nn.ReLU()
        self._dropout = nn.Dropout(p=dropout)
        self._head = nn.Linear(in_features=top_classifier_units, out_features=1)
            
    
    def forward(self, features, lenght):
        
        batch_size = features[0].shape[0]
        
        embeddings = [embedding(features[i]) for i, embedding in enumerate(self._credits_cat_embeddings)]
        concated_embeddings = torch.cat(embeddings, dim=-1)
        
        concated_embeddings = concated_embeddings.permute(0, 2, 1).unsqueeze(3)
        
        dropout_embeddings = self._spatial_dropout(concated_embeddings)
        dropout_embeddings = dropout_embeddings.squeeze(3).permute(2, 0, 1)

        packed_embedded = nn.utils.rnn.pack_padded_sequence(dropout_embeddings, lenght, enforce_sorted=True)

        states, last_hidden = self._gru(packed_embedded)            
        states, _ = nn.utils.rnn.pad_packed_sequence(states)

        last_hidden = torch.tanh(self._fc(torch.cat((last_hidden[-2,:,:], last_hidden[-1,:,:]), dim = 1)))
            
        rnn_max_pool = states.max(dim=0)[0]
        rnn_avg_pool = states.sum(dim=0) / states.shape[1] 

        combined_input = torch.cat([rnn_max_pool, rnn_avg_pool, last_hidden], dim=-1)

        classification_hidden = self._top_classifier(combined_input)
        classification_hidden = self._dropout(classification_hidden)

        activation = self._intermediate_activation(classification_hidden)
        raw_output = self._head(activation)

        return raw_output
    
    @classmethod
    def _create_embedding_projection(cls, cardinality, embed_size, add_missing=True, padding_idx=0):
        add_missing = 1 if add_missing else 0
        return nn.Embedding(num_embeddings=cardinality+add_missing, embedding_dim=embed_size, padding_idx=padding_idx)


class CreditsRNN_SIMPLE(nn.Module):
    def __init__(self, features, embedding_projections, rnn_units=128, top_classifier_units=32, dropout=0.1, spatial_dropout=0.05):
        super(CreditsRNN_SIMPLE, self).__init__()
        
        self._credits_cat_embeddings = nn.ModuleList([self._create_embedding_projection(*embedding_projections[feature]) 
                                                          for feature in features])
        self._spatial_dropout = nn.Dropout2d(spatial_dropout)
        self._credits_cat_embeddings_dim = sum([embedding_projections[x][1] for x in features])
        
        self._gru = nn.GRU(input_size=self._credits_cat_embeddings_dim,
                                 hidden_size=rnn_units, batch_first=True, bidirectional=True)
        self._hidden_size = rnn_units
        pooling_result_dimension = self._hidden_size * 2

        self._top_classifier = nn.Linear(in_features=2*pooling_result_dimension, out_features=top_classifier_units)
            
        self._intermediate_activation = nn.ReLU()
        self._dropout = nn.Dropout(p=dropout)
        self._head = nn.Linear(in_features=top_classifier_units, out_features=1)
            
    
    def forward(self, features):
        batch_size = features[0].shape[0]
        embeddings = [embedding(features[i]) for i, embedding in enumerate(self._credits_cat_embeddings)]
        concated_embeddings = torch.cat(embeddings, dim=-1)
        
        concated_embeddings = concated_embeddings.permute(0, 2, 1).unsqueeze(3)
        
        dropout_embeddings = self._spatial_dropout(concated_embeddings)
        dropout_embeddings = dropout_embeddings.squeeze(3).permute(0, 2, 1)

        states, _ = self._gru(dropout_embeddings)
        rnn_max_pool = states.max(dim=1)[0]
        rnn_avg_pool = states.sum(dim=1) / states.shape[1] 
        combined_input = torch.cat([rnn_max_pool, rnn_avg_pool], dim=-1)

        classification_hidden = self._top_classifier(combined_input)
        classification_hidden = self._dropout(classification_hidden)

        activation = self._intermediate_activation(classification_hidden)
        raw_output = self._head(activation)

        return raw_output
    
    @classmethod
    def _create_embedding_projection(cls, cardinality, embed_size, add_missing=True, padding_idx=0):
        add_missing = 1 if add_missing else 0
        return nn.Embedding(num_embeddings=cardinality+add_missing, embedding_dim=embed_size, padding_idx=padding_idx)


class CreditsRNN_ATTENTION(nn.Module):
    def __init__(self, features, embedding_projections, rnn_units=128, top_classifier_units=32, dropout=0.1, spatial_dropout=0.05):
        super(CreditsRNN_ATTENTION, self).__init__()
        
        
        self._credits_cat_embeddings = nn.ModuleList([self._create_embedding_projection(*embedding_projections[feature]) 
                                                          for feature in features])
        self._spatial_dropout = nn.Dropout2d(spatial_dropout)
        self._credits_cat_embeddings_dim = sum([embedding_projections[x][1] for x in features])
                
        self._gru = nn.GRU(input_size=self._credits_cat_embeddings_dim,
                           hidden_size=rnn_units, batch_first=True, bidirectional=True)

        self._hidden_size = rnn_units
        self._pool = nn.AdaptiveAvgPool1d(10)
        self._flatten = nn.Flatten()
            
        pooling_result_dimension = self._hidden_size * 2 

        self._fc = nn.Linear(self._hidden_size * 2, 256)
        self._attn = nn.Linear((self._hidden_size * 2) + 256, 256)

        self._top_classifier = nn.Linear(in_features=pooling_result_dimension*4 + pooling_result_dimension, out_features=top_classifier_units)
            
        self._intermediate_activation = nn.ReLU()
        self._dropout = nn.Dropout(p=dropout)
        self._head = nn.Linear(in_features=top_classifier_units, out_features=1)
            
    
    def forward(self, features, lenght):
        
        batch_size = features[0].shape[0]
        
        embeddings = [embedding(features[i]) for i, embedding in enumerate(self._credits_cat_embeddings)]
        concated_embeddings = torch.cat(embeddings, dim=-1)
        
        concated_embeddings = concated_embeddings.permute(0, 2, 1).unsqueeze(3)
        
        dropout_embeddings = self._spatial_dropout(concated_embeddings)
        dropout_embeddings = dropout_embeddings.squeeze(3).permute(2, 0, 1)

        packed_embedded = nn.utils.rnn.pack_padded_sequence(dropout_embeddings, lenght, enforce_sorted=True)

        states, last_hidden = self._gru(packed_embedded)
            
        states, _ = nn.utils.rnn.pad_packed_sequence(states)

        src_len = states.shape[0]
        last_hidden = torch.tanh(self._fc(torch.cat((last_hidden[-2,:,:], last_hidden[-1,:,:]), dim = 1)))

        hidden = last_hidden.unsqueeze(1).repeat(1, src_len, 1)
        encoder_outputs = states.permute(1, 0, 2)
        energy = torch.tanh(self._attn(torch.cat((hidden, encoder_outputs), dim = 2)))
        energy = energy.permute(1, 0, 2)

        rnn_max_pool = states.max(dim=0)[0]
        rnn_avg_pool = states.sum(dim=0) / states.shape[1] 

        energy_max_pool = energy.max(dim=0)[0]
        energy_avg_pool = energy.sum(dim=0) / energy.shape[1] 

        combined_input = torch.cat([rnn_max_pool, rnn_avg_pool, energy_max_pool, energy_avg_pool, last_hidden], dim=-1)

        classification_hidden = self._top_classifier(combined_input)
        classification_hidden = self._dropout(classification_hidden)

        activation = self._intermediate_activation(classification_hidden)
        raw_output = self._head(activation)

        return raw_output
    
    @classmethod
    def _create_embedding_projection(cls, cardinality, embed_size, add_missing=True, padding_idx=0):
        add_missing = 1 if add_missing else 0
        return nn.Embedding(num_embeddings=cardinality+add_missing, embedding_dim=embed_size, padding_idx=padding_idx)    
    

class CreditsRNN_LAST_HIDDEN_SHUFFLE(nn.Module):
    def __init__(self, features, embedding_projections, rnn_units=128, top_classifier_units=32, dropout=0.1, spatial_dropout=0.05):
        super(CreditsRNN_LAST_HIDDEN_SHUFFLE, self).__init__()
        
        self._credits_cat_embeddings = nn.ModuleList([self._create_embedding_projection(*embedding_projections[feature]) 
                                                          for feature in features])
        self._spatial_dropout = nn.Dropout2d(spatial_dropout)
        self._credits_cat_embeddings_dim = sum([embedding_projections[x][1] for x in features])

        
        self._gru = nn.GRU(input_size=self._credits_cat_embeddings_dim,
                                 hidden_size=rnn_units, batch_first=True, bidirectional=True)

        self._hidden_size = rnn_units
            
        pooling_result_dimension = self._hidden_size * 2 

        self._fc = nn.Linear(self._hidden_size * 2, 256)

        self._top_classifier = nn.Linear(in_features=pooling_result_dimension*2 + self._hidden_size, out_features=top_classifier_units) #!!! Здесь отличие shuffle и не shuffle
                       
        self._intermediate_activation = nn.ReLU()
        self._dropout = nn.Dropout(p=dropout)
        self._head = nn.Linear(in_features=top_classifier_units, out_features=1)
            
    
    def forward(self, features, lenght):
        
        batch_size = features[0].shape[0]
        
        embeddings = [embedding(features[i]) for i, embedding in enumerate(self._credits_cat_embeddings)]
        concated_embeddings = torch.cat(embeddings, dim=-1)
        
        concated_embeddings = concated_embeddings.permute(0, 2, 1).unsqueeze(3)
        
        dropout_embeddings = self._spatial_dropout(concated_embeddings)
        dropout_embeddings = dropout_embeddings.squeeze(3).permute(2, 0, 1)

        packed_embedded = nn.utils.rnn.pack_padded_sequence(dropout_embeddings, lenght, enforce_sorted=True)

        states, last_hidden = self._gru(packed_embedded)
            
        states, _ = nn.utils.rnn.pad_packed_sequence(states)

        #src_len = states.shape[0]
        last_hidden = torch.tanh(self._fc(torch.cat((last_hidden[-2,:,:], last_hidden[-1,:,:]), dim = 1)))
     
        rnn_max_pool = states.max(dim=0)[0]
        rnn_avg_pool = states.sum(dim=0) / states.shape[1] 

        combined_input = torch.cat([rnn_max_pool, rnn_avg_pool, last_hidden], dim=-1)

        classification_hidden = self._top_classifier(combined_input)
        classification_hidden = self._dropout(classification_hidden)

        activation = self._intermediate_activation(classification_hidden)
        raw_output = self._head(activation)
        
        return raw_output
    
    @classmethod
    def _create_embedding_projection(cls, cardinality, embed_size, add_missing=True, padding_idx=0):
        add_missing = 1 if add_missing else 0
        return nn.Embedding(num_embeddings=cardinality+add_missing, embedding_dim=embed_size, padding_idx=padding_idx)    
    

    
#CONV
class CreditsCONV(nn.Module):
    def __init__(self,
                 features,
                 embedding_projections, 
                 hid_dim,
                 n_layers,
                 cls_units,
                 top_classifier_units, 
                 kernel_size, 
                 dropout, 
                 device,
                 max_length = 60):
        super().__init__()
        
        assert kernel_size % 2 == 1, "Kernel size must be odd!"
        
        self.device = device
        
        self._scale = torch.sqrt(torch.FloatTensor([0.5])).to(device)
        
        self._credits_cat_embeddings = nn.ModuleList([self._create_embedding_projection(*embedding_projections[feature]) 
                                                          for feature in features])
        
        emb_dim = sum([embedding_projections[x][1] for x in features])
        
        self._pos_embedding = nn.Embedding(max_length, emb_dim)
        
        self._emb2hid = nn.Linear(emb_dim, hid_dim)
        self._hid2emb = nn.Linear(hid_dim, emb_dim)
        
        self._convs = nn.ModuleList([nn.Conv1d(in_channels = hid_dim, 
                                              out_channels = 2 * hid_dim, 
                                              kernel_size = kernel_size, 
                                              padding = (kernel_size - 1) // 2)
                                    for _ in range(n_layers)])
        
        self._dropout = nn.Dropout(dropout)

        self._pool = nn.AdaptiveAvgPool1d(cls_units)
        self._flatten = nn.Flatten()
        self._top_classifier = nn.Linear(in_features=emb_dim * cls_units, out_features=top_classifier_units)
        self._intermediate_activation = nn.ReLU()
        self._head = nn.Linear(in_features=top_classifier_units, out_features=1)
        
    def forward(self, features):
        
        #src = [batch size, src len]
        
        embeddings = [embedding(features[i]) for i, embedding in enumerate(self._credits_cat_embeddings)]
        concated_embeddings = torch.cat(embeddings, dim=-1)
        
        batch_size = concated_embeddings.shape[0]
        src_len = concated_embeddings.shape[1]
        
        #create position tensor
        pos = torch.arange(0, src_len).unsqueeze(0).repeat(batch_size, 1).to(self.device)
        
        #pos = [0, 1, 2, 3, ..., src len - 1]
        
        #pos = [batch size, src len]
        
        #embed tokens and positions
        
        pos_embedded = self._pos_embedding(pos)
        
        #pos_embedded = [batch size, src len, emb dim]
        
        #combine embeddings by elementwise summing
        embedded = self._dropout(concated_embeddings + pos_embedded)
        
        #embedded = [batch size, src len, emb dim]
        
        #pass embedded through linear layer to convert from emb dim to hid dim
        conv_input = self._emb2hid(embedded)
        
        #conv_input = [batch size, src len, hid dim]
        
        #permute for convolutional layer
        conv_input = conv_input.permute(0, 2, 1) 
        
        #conv_input = [batch size, hid dim, src len]
        
        #begin convolutional blocks...
        
        for i, conv in enumerate(self._convs):
        
            #pass through convolutional layer
            conved = conv(self._dropout(conv_input))

            #conved = [batch size, 2 * hid dim, src len]

            #pass through GLU activation function
            conved = F.glu(conved, dim = 1)

            #conved = [batch size, hid dim, src len]
            
            #apply residual connection
            conved = (conved + conv_input) * self._scale

            #conved = [batch size, hid dim, src len]
            
            #set conv_input to conved for next loop iteration
            conv_input = conved
        
        #...end convolutional blocks
        
        #permute and convert back to emb dim
        conved = self._hid2emb(conved.permute(0, 2, 1))

        #conved = [batch size, src len, emb dim]
        conved = conved.permute(0, 2, 1)
        conved = self._pool(conved)
        conved = self._flatten(conved)
        conved = self._top_classifier(conved)
        conved = self._intermediate_activation(conved)
        conved = self._head(conved)
        
        return conved 

    @classmethod
    def _create_embedding_projection(cls, cardinality, embed_size, add_missing=True, padding_idx=0):
        add_missing = 1 if add_missing else 0
        return nn.Embedding(num_embeddings=cardinality+add_missing, embedding_dim=embed_size, padding_idx=padding_idx)

    
#Трансформер
class CreditsTransformer(nn.Module):
    def __init__(self,
                 features,
                 embedding_projections, 
                 n_layers, 
                 n_heads,
                 top_classifier_units,
                 pf_dim,
                 dropout, 
                 device,
                 max_length = 60):
        super().__init__()
        seed_everything(42)

        self.device = device
        self._credits_cat_embeddings = nn.ModuleList([self._create_embedding_projection(*embedding_projections[feature]) 
                                                          for feature in features])
        
        hid_dim = sum([embedding_projections[x][1] for x in features])
        
        self.pos_embedding = nn.Embedding(max_length, hid_dim)
        self.ids_embedding = nn.Embedding(5, 3)
        
        self.layers = nn.ModuleList([EncoderLayer(hid_dim, 
                                                  n_heads, 
                                                  pf_dim,
                                                  dropout, 
                                                  device) 
                                     for _ in range(n_layers)])
        
        self._dropout = nn.Dropout(dropout)
        
        self._scale = torch.sqrt(torch.FloatTensor([hid_dim])).to(device)

        self._flatten = nn.Flatten()
        self.classifire1 = nn.Linear(in_features=hid_dim * 2 + 3, out_features=top_classifier_units)
        self._intermediate_activation = nn.ReLU()
        self.classifire2 = nn.Linear(in_features=top_classifier_units, out_features=1)
        
    def forward(self, features, mask=None, ids=None):
        
        embeddings = [embedding(features[i]) for i, embedding in enumerate(self._credits_cat_embeddings)]
        concated_embeddings = torch.cat(embeddings, dim=-1)
        
        #concated_embeddings = [batch size, src len, hid_dim]
        
        batch_size = concated_embeddings.shape[0]
        seq_len = concated_embeddings.shape[1]
                       
        pos = torch.arange(0, seq_len).unsqueeze(0).repeat(batch_size, 1).to(self.device)
        
        #pos = [batch size, seq_len]

        
        seq = self._dropout((concated_embeddings * self._scale) + self.pos_embedding(pos))
        #seq = self._dropout((concated_embeddings * self._scale))
        
        
        #src = [batch size, src len, hid dim]
        
        for layer in self.layers:
            seq = layer(seq, mask)
        
        ids = torch.IntTensor(ids//875000).to(self.device)
        
        ids = self.ids_embedding(ids)
               
        max_pool = seq.max(dim=1)[0]
        avg_pool = seq.sum(dim=1) / seq.shape[1] 
        
        combined_input = torch.cat([ids, max_pool, avg_pool], dim=-1)

        classification_hidden = self.classifire1(combined_input)
        activation = self._intermediate_activation(classification_hidden)
        raw_output = self.classifire2(activation)
            
        return raw_output

    @classmethod
    def _create_embedding_projection(cls, cardinality, embed_size, add_missing=True, padding_idx=0):
        add_missing = 1 if add_missing else 0
        return nn.Embedding(num_embeddings=cardinality+add_missing, embedding_dim=embed_size, padding_idx=padding_idx)

    
class EncoderLayer(nn.Module):
    def __init__(self, 
                 hid_dim, 
                 n_heads, 
                 pf_dim,  
                 dropout, 
                 device):
        super().__init__()
        
        self.self_attn_layer_norm = nn.LayerNorm(hid_dim)
        self.ff_layer_norm = nn.LayerNorm(hid_dim)
        self.self_attention = MultiHeadAttentionLayer(hid_dim, n_heads, dropout, device)
        self.positionwise_feedforward = PositionwiseFeedforwardLayer(hid_dim, 
                                                                     pf_dim, 
                                                                     dropout)
        self.dropout = nn.Dropout(dropout)
        
    def forward(self, src, src_mask):
        
        #src = [batch size, src len, hid dim]
        #src_mask = [batch size, 1, 1, src len] 
                
        #self attention
        _src, _ = self.self_attention(src, src, src, src_mask)
        
        #dropout, residual connection and layer norm
        src = self.self_attn_layer_norm(src + self.dropout(_src))
        
        #src = [batch size, src len, hid dim]
        
        #positionwise feedforward
        _src = self.positionwise_feedforward(src)
        
        #dropout, residual and layer norm
        src = self.ff_layer_norm(src + self.dropout(_src))
        
        #src = [batch size, src len, hid dim]
        
        return src   
    
    
class MultiHeadAttentionLayer(nn.Module):
    def __init__(self, hid_dim, n_heads, dropout, device):
        super().__init__()
        
        assert hid_dim % n_heads == 0
        
        self.hid_dim = hid_dim
        self.n_heads = n_heads
        self.head_dim = hid_dim // n_heads
        
        self.fc_q = nn.Linear(hid_dim, hid_dim)
        self.fc_k = nn.Linear(hid_dim, hid_dim)
        self.fc_v = nn.Linear(hid_dim, hid_dim)
        
        self.fc_o = nn.Linear(hid_dim, hid_dim)
        
        self.dropout = nn.Dropout(dropout)
        
        self.scale = torch.sqrt(torch.FloatTensor([self.head_dim])).to(device)
        
    def forward(self, query, key, value, mask = None):
        
        batch_size = query.shape[0]
        
        #query = [batch size, query len, hid dim]
        #key = [batch size, key len, hid dim]
        #value = [batch size, value len, hid dim]
                
        Q = self.fc_q(query)
        K = self.fc_k(key)
        V = self.fc_v(value)
        
        #Q = [batch size, query len, hid dim]
        #K = [batch size, key len, hid dim]
        #V = [batch size, value len, hid dim]
                
        Q = Q.view(batch_size, -1, self.n_heads, self.head_dim).permute(0, 2, 1, 3)
        K = K.view(batch_size, -1, self.n_heads, self.head_dim).permute(0, 2, 1, 3)
        V = V.view(batch_size, -1, self.n_heads, self.head_dim).permute(0, 2, 1, 3)
        
        #Q = [batch size, n heads, query len, head dim]
        #K = [batch size, n heads, key len, head dim]
        #V = [batch size, n heads, value len, head dim]
                
        energy = torch.matmul(Q, K.permute(0, 1, 3, 2)) / self.scale
        
        #energy = [batch size, n heads, query len, key len]
                
        if mask is not None:
            energy = energy.masked_fill(mask == 0, -1e10)
        
        attention = torch.softmax(energy, dim = -1)
                
        #attention = [batch size, n heads, query len, key len]
                
        x = torch.matmul(self.dropout(attention), V)
        
        #x = [batch size, n heads, query len, head dim]
        
        x = x.permute(0, 2, 1, 3).contiguous()
        
        #x = [batch size, query len, n heads, head dim]
        
        x = x.view(batch_size, -1, self.hid_dim)
        
        #x = [batch size, query len, hid dim]
        
        x = self.fc_o(x)
        
        #x = [batch size, query len, hid dim]
        
        return x, attention

    
class PositionwiseFeedforwardLayer(nn.Module):
    def __init__(self, hid_dim, pf_dim, dropout):
        super().__init__()
        
        self.fc_1 = nn.Linear(hid_dim, pf_dim)
        self.fc_2 = nn.Linear(pf_dim, hid_dim)
        
        self.dropout = nn.Dropout(dropout)
        
    def forward(self, x):
        
        #x = [batch size, seq len, hid dim]
        
        x = self.dropout(torch.relu(self.fc_1(x)))
        
        #x = [batch size, seq len, pf dim]
        
        x = self.fc_2(x)
        
        #x = [batch size, seq len, hid dim]
        
        return x  
