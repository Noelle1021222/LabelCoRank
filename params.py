
# # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # resultShow.py
n_rs='0'
i_rs='0'
j_rs='AAPD'
threshold_resultShow=0.3
modelN=''
TOPK=30
tagIndex_resultShow='AAPD_2'

tagListPath_resultShow='data/wjtFiles/tag2022List'+tagIndex_resultShow+'.txt'

tPath='./results/'+n_rs+'a/'+n_rs+i_rs+j_rs+'targetResult'+str(threshold_resultShow)+"TOP"+str(TOPK)+'m'+modelN+'.txt' #读
pPath='./results/'+n_rs+'a/'+n_rs+i_rs+j_rs+'predictResult'+str(threshold_resultShow)+"TOP"+str(TOPK)+'m'+modelN+'.txt'#读
rPath='./results/'+n_rs+'a/tagShow/'+n_rs+i_rs+j_rs+'tagShow'+str(threshold_resultShow)+"TOP"+str(TOPK)+'m'+modelN+'.txt'#写

topKPath='./results/'+n_rs+'a/'+n_rs+i_rs+j_rs+'topKResult'+str(threshold_resultShow)+"TOP"+str(TOPK)+'m'+modelN+'.txt'#读
topKShowPath='./results/'+n_rs+'a/topKResult/'+n_rs+i_rs+j_rs+'topKResultTag'+str(threshold_resultShow)+"TOP"+str(TOPK)+'m'+modelN+'.txt'#写
tShowPath='./results/'+n_rs+'a/topKResult/'+n_rs+i_rs+j_rs+'targetResultTag'+str(threshold_resultShow)+"TOP"+str(TOPK)+'m'+modelN+'.txt'#写
# # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # resultShow.py


# # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # score.py
n_score='0'
i_score='0'
j_score='AAPD'
threshold_score=0.3
modelN_score=''
TOPK_score=30
tagIndex_score='AAPD_2'

tagListPath_score='data/wjtFiles/tag2022List'+tagIndex_score+'.txt'

tPath_score='./results/'+n_score+'a/'+n_score+i_score+j_score+'targetResult'+str(threshold_score)+"TOP"+str(TOPK_score)+'m'+modelN_score+'.txt'#读
pPath_score='./results/'+n_score+'a/'+n_score+i_score+j_score+'predictResult'+str(threshold_score)+"TOP"+str(TOPK_score)+'m'+modelN_score+'.txt'#读

topKShowPath_score='./results/'+n_score+'a/'+'topKResult/'+n_score+i_score+j_score+'topKResultTag'+str(threshold_score)+"TOP"+str(TOPK_score)+'m'+modelN_score+'.txt'#读
tShowPath_score='./results/'+n_score+'a/'+'topKResult/'+n_score+i_score+j_score+'targetResultTag'+str(threshold_score)+"TOP"+str(TOPK_score)+'m'+modelN_score+'.txt'#读


# # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # score.py


# # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # #createTestJson
o='7'
p='3'
k='5'
threshold_cte=0.3
TOPK_cte=30
modelN_cte=''
t2i=''

prePath='./results/'+o+'a/'+o+p+k+'predictResult'+str(threshold_cte)+"TOP"+str(TOPK_cte)+'m'+modelN_cte+'.txt'

jsonPath='./results/'+o+'a/testJson/'+o+p+k+'test'+str(threshold_cte)+"TOP"+str(TOPK_cte)+'m'+modelN_cte+'.json'
t2idPath='data/wjtFiles/tag2id'+t2i+'.txt'
pmidPath='./results/'+o+'a/'+o+p+k+'pmid'+t2i+str(threshold_cte)+"TOP"+str(TOPK_cte)+'m'+modelN_cte+'.txt'

# # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # #createTestJson
