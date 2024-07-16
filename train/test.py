import torch
import numpy as np
import params
from load_data import data_loader_onlytest
import model1,model2,model3,model4,model5,model6

targetResult = open(params.targetResultPath, encoding='utf-8', mode='w')
predictResult = open(params.predictResultPath, encoding='utf-8', mode='w')
topKresult = open(params.topKResultPath, encoding='utf-8', mode='w')

val_targets = []
val_outputs = []

if __name__ == '__main__':

    if torch.cuda.is_available():
        device = torch.device("cuda")
        print(f'There are {torch.cuda.device_count()} GPU(s) available, '
              f'We will use the GPU: {torch.cuda.get_device_name(0)}.')
    else:
        print('No GPU available, using the CPU instead.')
        device = torch.device("cpu")


    best_model ='../save_model/' + params.modelName + '.pt'
    
    allTag = list()
    with open(params.tagListPath_testOnly, encoding='utf-8', mode='r') as tagData:
        for line in tagData:
            allTag.append(line)
    meshNum=len(allTag)


    model = model6.BERTClass(meshNum,allTag)
    checkpoint = torch.load(best_model)

    model.load_state_dict(checkpoint)
    model.to(device)

    test_loader = data_loader_onlytest(params.testDataPath, meshNum)

    with torch.no_grad():
        model.eval()
        for batch_idx, data in enumerate(test_loader, 0):
            if batch_idx % 100 == 0:
                print(f'BATCH: {batch_idx}')
            ids = data['ids'].to(device, dtype=torch.long)
            mask = data['mask'].to(device, dtype=torch.long)
            token_type_ids = data['token_type_ids'].to(device, dtype=torch.long)
            targets = data['targets'].to(device, dtype=torch.float)
            _,outputs = model(ids, mask, token_type_ids)

            target = targets.cpu().detach().numpy()
            predict = torch.sigmoid(outputs).cpu().detach().numpy()
            for t in target:
                t = t.nonzero()
                tList = t[0].tolist()
                targetResult.writelines(str(tList) + '\n')

            for r in predict:
                r2 = (np.array(r) >= params.threshold).astype(int)
                r2 = r2.nonzero()
                pList = r2[0].tolist()
                predictResult.writelines(str(pList) + '\n')
                scores,indices=torch.topk(torch.from_numpy(r),params.K)
                topKresult.writelines(str(indices.tolist())+'\n')


        print('Test End')
        print('testonly标签数据路径：' + params.tagListPath_testOnly)
        print('testonly标签数量：'+str(meshNum))
        print('testonly测试数据：' + params.testDataPath)
        print('testonly测试使用阈值：' + str(params.threshold))
        print('testonly测试使用模型名：' + params.modelName)

        print('testonly目标结果保存：' + params.targetResultPath)
        print('testonly预测结果保存：' + params.predictResultPath)
        print('testonly的TopK结果保存：' + params.topKResultPath)

        del data, batch_idx, targets, outputs
