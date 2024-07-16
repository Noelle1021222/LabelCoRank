import numpy as np
import params
import os
targetResult = open(params.tPath, encoding='utf-8', mode='r')
predictResult= open(params.pPath, encoding='utf-8', mode='r')
resultTag=open(params.rPath, encoding='utf-8', mode='w')

topKResult=open(params.topKPath, encoding='utf-8', mode='r')
targetShow=open(params.tShowPath, encoding='utf-8', mode='w')
topKShow=open(params.topKShowPath, encoding='utf-8', mode='w')


target=targetResult.readline()
predict=predictResult.readline()
topK=topKResult.readline()

allTag=list()
with open(params.tagListPath_resultShow, encoding='utf-8', mode='r') as tagData:
    for line in tagData:
        allTag.append(line)
lenth=len(allTag)

while target:
    target=np.array(eval(target))
    predict=np.array(eval(predict))
    topK = np.array(eval(topK))

    for i in target:
        tag=allTag[i].rstrip('\n')
        resultTag.writelines(" $"+tag)
        targetShow.writelines(tag + "$")
    resultTag.writelines("\n")
    targetShow.writelines("\n")

    for j in predict:
        tag=allTag[j].rstrip('\n')
        resultTag.writelines(" $"+tag)
    resultTag.writelines("\n")

    for k in topK:
        tag=allTag[k].rstrip('\n')
        topKShow.writelines(tag+"$")
    topKShow.writelines("\n")

    target=targetResult.readline()
    predict = predictResult.readline()
    topK = topKResult.readline()

targetResult.close()
predictResult.close()
resultTag.close()
topKResult.close()
targetShow.close()
topKShow.close()

with open(params.tShowPath) as fp:
    l=[list(filter(None,line.strip().split('$'))) for line in fp]
    labels = np.asarray(l, dtype=object)
    np.save(os.path.splitext(params.tShowPath)[0], labels)

with open(params.topKShowPath) as fp:
    labels = np.asarray( [list(filter(None,line.strip().split('$'))) for line in fp], dtype=object)
    np.save(os.path.splitext(params.topKShowPath)[0], labels)

print('resultShow标签数据读取路径' + params.tagListPath_resultShow)
print('resultShow标签数量 '+str(lenth))

print('resultShow‘targetResult’读取路径' + params.tPath)
print('resultShow‘predictResult’读取路径' + params.pPath)
print('resultShow‘resultTag’写入路径' + params.rPath)

print('topKShow‘topKResult’读取路径' + params.topKPath)
print('topKShow‘targetResultTag’写入路径' + params.tShowPath)
print('topKShow‘topKResultTag’写入路径' + params.topKShowPath)

