# -*- coding: utf-8 -*-
# # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # #model_train
MAX_LEN = 512
TRAIN_BATCH_SIZE = 16
VALID_BATCH_SIZE = 16
TEST_BATCH_SIZE = 16

MODEL_NAME_OR_PATH = '../utils/preTrainedModels/roberta-base'

HIDDEN_LAYER_SIZE = 768
# HIDDEN_LAYER_SIZE = 1024
num_epochs = 50
learning_rate = 5e-06
warmup_epochs = 2
weight_decay = 5e-10
do_eval = True
do_save = True
printEV=100


headNum=12
headSize=64

label_ebd=768
trans_hidden=512
# num_base=25
num_atten=20
statics=20




matrix_Index="AAPD"
index='AAPD'#
modelName_train='' # -----------------------保存模型名

trainDataPath='../data/'+index+'data_train1.csv'
valDataPath='../data/'+index+'data_val1.csv'

tagIndex_train= 'AAPD_2'
tagListPath='../data/wjtFiles/tag2022List'+tagIndex_train+'.txt'

# # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # #model_train


# # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # #testOnly
n='0'
i='0'
j='AAPD'
threshold=0.3
modelName='' #
K=30 #
tagIndex_testOnly='AAPD_2'
tagListPath_testOnly='../data/wjtFiles/tag2022List'+tagIndex_testOnly+'.txt'

# matrix_Index_test="Mesh"
# matrix_label_test="matrix_labels_"+matrix_Index_test+"_topK.npy"
# matrix_jrl_test="matrix_journals_"+matrix_Index_test+"_topK.npy"


idx='AAPD'
testDataPath='../data/'+idx +'data_test1.csv'
#testDataPath='../data/'+n+'a/'+n+i+j+'testData.csv'
targetResultPath='../results/'+n+'a/'+n+i+j+'targetResult'+str(threshold)+"TOP"+str(K)+'m'+modelName+'.txt'
predictResultPath='../results/'+n+'a/'+n+i+j+'predictResult'+str(threshold)+"TOP"+str(K)+'m'+modelName+'.txt'
topKResultPath='../results/'+n+'a/'+n+i+j+'topKResult'+str(threshold)+"TOP"+str(K)+'m'+modelName+'.txt'

extendTags='../results/'+n+'a/'+n+i+j+'extendTags'+str(threshold)+"TOP"+str(K)+'m'+modelName+'.txt'
pre1Tags='../results/'+n+'a/'+n+i+j+'pre1Tags'+str(threshold)+"TOP"+str(K)+'m'+modelName+'.txt'

testPmidWritePath='../results/'+n+'a/'+n+i+j+'pmid'+str(threshold)+"TOP"+str(K)+'m'+modelName+'.txt'
