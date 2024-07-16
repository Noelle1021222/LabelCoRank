import torch
import torch.nn as nn
import torch.nn.functional as F
import shutil
import numpy as np
import math
import torch
import transformers
from transformers import BertTokenizer
from params import MODEL_NAME_OR_PATH, HIDDEN_LAYER_SIZE
import params

class Atten(nn.Module):
    def __init__(self, size):
        super(Atten, self).__init__()
        # self.transQ = nn.Linear(params.label_ebd,size)
        self.transQ = Trans(params.label_ebd, size)

    def forward(self, features, ebd,attention_mask):
        attn = self.transQ(ebd)
        # attn = ebd
        batch_size = features.size(0)
        seq_len = features.size(1)
        # features1=self.transK(features)
        # features2=self.transV(features)

        extended_attention_mask = attention_mask.unsqueeze(1)
        extended_attention_mask = extended_attention_mask.to(dtype=next(self.parameters()).dtype)
        extended_attention_mask = (1.0 - extended_attention_mask) * -10000.0

        attnscore = torch.bmm(attn, features.transpose(1, 2))
        attnscore=attnscore+extended_attention_mask
        attnscore = F.softmax(attnscore, dim=-1)
        atten_features = torch.bmm(attnscore, features)
        atten_features = atten_features.view(batch_size, -1)

        return atten_features

class multiAtten(nn.Module):
    def __init__(self, headNum,headSize):
        super(multiAtten, self).__init__()
        
        self.head_num = headNum
        self.head_size = headSize
        self.all_head_size = headNum * headSize
        
        self.query = nn.Linear(params.label_ebd, self.all_head_size)
        self.key = nn.Linear(HIDDEN_LAYER_SIZE, self.all_head_size)
        self.value = nn.Linear(HIDDEN_LAYER_SIZE, self.all_head_size)
        
        self.dropout = nn.Dropout(0.3)

    def transpose_for_scores(self, x):
        new_x_shape = x.size()[:-1] + (self.head_num, self.head_size)
        x = x.view(*new_x_shape)
        return x.permute(0, 2, 1, 3)
        
    def forward(self, features, ebd,attention_mask):
        batch_size = features.size(0)
        extended_attention_mask = attention_mask.unsqueeze(1).unsqueeze(2) # batch, 1, 1, seq_len
        extended_attention_mask = extended_attention_mask.to(dtype=next(self.parameters()).dtype) # fp16 compatibility
        extended_attention_mask = (1.0 - extended_attention_mask) * -10000.0
        
        mixed_query_layer = self.query(ebd)
        mixed_key_layer = self.key(features)
        mixed_value_layer = self.value(features)

        query_layer = self.transpose_for_scores(mixed_query_layer)
        key_layer = self.transpose_for_scores(mixed_key_layer)
        value_layer = self.transpose_for_scores(mixed_value_layer)
        
        attention_scores = torch.matmul(query_layer, key_layer.transpose(-1, -2))
        attention_scores = attention_scores / math.sqrt(self.head_size)
        attention_scores = attention_scores + extended_attention_mask
        attention_probs = nn.Softmax(dim=-1)(attention_scores)
        attention_probs = self.dropout(attention_probs)
        
        context_layer = torch.matmul(attention_probs, value_layer)
        context_layer = context_layer.permute(0, 2, 1, 3).contiguous()
        
        new_context_layer_shape = context_layer.size()[:-2] + (self.all_head_size,)
        atten_features = context_layer.view(*new_context_layer_shape)
        atten_features = atten_features.view(batch_size, -1)

        return atten_features

class BERTClass(torch.nn.Module):
    def __init__(self, meshNum, allTag):
        super(BERTClass, self).__init__()

        self.allTag = allTag
        self.device = torch.device("cuda")
        self.bert =transformers.AutoModel.from_pretrained(MODEL_NAME_OR_PATH, return_dict=True,
                                                           output_hidden_states=True)

        self.matrix_label = np.load('../data/wjtFiles/matrix_labels_' + params.matrix_Index + '_top100.npy',
                                    allow_pickle=True)

        self.drop1 = torch.nn.Dropout(0.3)

        self.linear = torch.nn.Linear(HIDDEN_LAYER_SIZE , meshNum)

        nn.init.xavier_normal_(self.linear.weight)

        self.atten_bert = Atten(HIDDEN_LAYER_SIZE)
        self.atten_bert_multi = multiAtten(params.headNum,params.headSize)

        self.labelEmbedding = nn.Embedding(meshNum, params.label_ebd)

        self.labelEmbedding_POS = nn.Embedding(params.num_atten, params.label_ebd)

        self.final_feature = nn.Linear(params.num_atten*HIDDEN_LAYER_SIZE, meshNum)
        nn.init.xavier_normal_(self.final_feature.weight)

        self.transebd = Trans(params.label_ebd,HIDDEN_LAYER_SIZE)

        self.cornet1=CorNet(meshNum)
        
    def get_label_features(self,outputs,journal):
        batchsize = outputs.size(0)
        
        statics = torch.arange(0, params.num_atten, 1)
        
        correlation = torch.zeros((batchsize, params.num_atten), device="cuda", dtype=torch.int)
        predict = torch.sigmoid(outputs).cpu().detach()
        row_indices = []
        for i in predict:
            _, t = torch.topk(i, params.num_atten)
            if (len(t)<1):
                _, indices = torch.topk(i, 5)
                row_indices.append(indices)
            else:
                row_indices.append(t)

        for i in range(batchsize):
            data = [self.matrix_label[j] for j in row_indices[i]]
            lenth = len(row_indices[i])
            cat_tensor = row_indices[i]
            flat_data = [item for sublist in data for item in sublist ]
            result_dict = {}
            for key, value in flat_data:
                result_dict[key] = result_dict.get(key, 0) + value
            sorted_result = sorted(result_dict.items(), key=lambda x: x[1], reverse=True)
            sorted_keys = [item[0] for item in sorted_result if item[0] not in cat_tensor][:params.num_atten-lenth]
            cat_tensor=torch.cat((cat_tensor,torch.tensor(sorted_keys)), dim=0)
            cat_tensor.to("cuda")
            cat_tensor = torch.sort(cat_tensor.to("cuda"))[0]
            while(len(cat_tensor)<params.num_atten):
                cat_tensor=torch.cat((cat_tensor,cat_tensor),dim=0)[: params.num_atten]
            correlation[i]=cat_tensor

        label_embedding=self.labelEmbedding(correlation)
        label_embedding_POS=self.labelEmbedding_POS(torch.tile(statics,(batchsize,1)).to("cuda"))
        label_embedding_all=label_embedding+label_embedding_POS
        label_embedding_final=label_embedding_all[:,:params.statics,]
        label_features=label_embedding_all[:,params.statics:,]
        label_features = self.transebd(label_features)
        label_features = label_features.view(batchsize, -1)
        return label_embedding_final,label_features

    def forward(self, ids, mask, token_type_ids, journal=0):
        bert_output = self.bert(ids, attention_mask=mask, token_type_ids=token_type_ids)
        hidden_states = bert_output['hidden_states']
        last_hidden_state = bert_output['last_hidden_state']
        pooler_output = bert_output['pooler_output']
        drop_last_hidden = self.drop1(last_hidden_state)
        cls_pre = self.linear(pooler_output)
        atten_ebd, lb_features = self.get_label_features(cls_pre, journal)
        attention_mask = ids != 0
        atten_bert_features = self.atten_bert_multi(drop_last_hidden, atten_ebd,attention_mask)
        cat_features = torch.cat((atten_bert_features, lb_features), dim=1)
        final_output = self.final_feature(cat_features)
        return cls_pre, final_output
