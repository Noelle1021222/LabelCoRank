from utils import method
import warnings
warnings.filterwarnings('ignore')
import params
import numpy as np
import os
from sklearn.preprocessing import MultiLabelBinarizer
from sklearn.metrics import accuracy_score, f1_score, classification_report, precision_score, recall_score
from utils.evaluation import get_p_1, get_p_3, get_p_5, get_n_1, get_n_3, get_n_5
import csv

def socre_norm():
    targetResult = open(params.tPath_score, encoding='utf-8', mode='r')
    predictResult = open(params.pPath_score, encoding='utf-8', mode='r')

    allTag = list()
    with open(params.tagListPath_score, encoding='utf-8', mode='r') as tagData:
        for line in tagData:
            allTag.append(line)
    lenth = len(allTag)
    print('score标签数据路径' + params.tagListPath_score)
    print('score标签数量' + str(lenth))
    print('score‘targetResult’读取路径' + params.tPath_score)
    print('score‘predictResult’读取路径' + params.pPath_score)


    target = targetResult.readline()
    predict = predictResult.readline()

    tgt = []
    pre = []

    while target:
        target = eval(target)
        predict = eval(predict)

        listTag1 = [0] * lenth

        for j in target:
            listTag1[j] = 1
        tgt.append(listTag1)

        listTag1 = [0] * lenth
        for j in predict:
            listTag1[j] = 1
        pre.append(listTag1)
        target = targetResult.readline()
        predict = predictResult.readline()

    tgt = np.array(tgt)
    pre = np.array(pre)

    EBR = method.example_recall(tgt, pre)
    EBP = method.example_precision(tgt, pre)
    EBF = method.example_f1(tgt, pre)

    acc = method.example_accuracy(tgt, pre)
    # acc=accuracy_score(tgt, pre)
    mif = f1_score(tgt, pre, average='micro')
    mip = precision_score(tgt, pre, average='micro')
    mir = recall_score(tgt, pre, average='micro')

    maf = f1_score(tgt, pre, average='macro')
    maps = precision_score(tgt, pre, average='macro')
    mar = recall_score(tgt, pre, average='macro')
    Hamming_Loss = method.Hamming_Loss(tgt, pre)

    acc = "%.2f%%" % (round(float(acc) * 100, 6))
    MiF = "%.2f%%" % (round(float(mif) * 100, 6))
    MiR = "%.2f%%" % (round(float(mir) * 100, 6))
    MiP = "%.2f%%" % (round(float(mip) * 100, 6))
    MaF = "%.2f%%" % (round(float(maf) * 100, 6))
    MaR = "%.2f%%" % (round(float(mar) * 100, 6))
    MaP = "%.2f%%" % (round(float(maps) * 100, 6))
    EBF = "%.2f%%" % (round(float(EBF) * 100, 6))
    EBR = "%.2f%%" % (round(float(EBR) * 100, 6))
    EBP = "%.2f%%" % (round(float(EBP) * 100, 6))
    Hamming_Loss= round(float(Hamming_Loss), 6)

    return (acc,MiF,MiR,MiP,MaF,MaR,MaP,EBF,EBR,EBP,Hamming_Loss)

def evl(results, targets, a=0.55, b=1.5):
    res, targets = np.load(results, allow_pickle=True), np.load(targets, allow_pickle=True)

    mlb = MultiLabelBinarizer(sparse_output=True)
    targets = mlb.fit_transform(targets)

    p1 =round(float(get_p_1(res, targets, mlb)), 5)
    p2 =round(float(get_p_3(res, targets, mlb)), 5)
    p3 = round(float(get_p_5(res, targets, mlb)) , 5)
    n1 = round(float(get_n_1(res, targets, mlb)) , 5)
    n2 = round(float(get_n_3(res, targets, mlb)) , 5)
    n3 =round(float(get_n_5(res, targets, mlb)) , 5)
    return p1, p2, p3, n1, n2, n3



def printEvl(p1,p2,p3,n1,n2,n3):
    print('Precision@1,3,5:',p1, p2, p3)
    print('nDCG@1,3,5:', n1,n2, n3)

def printNormal(acc,MiF,MiR,MiP,MaF,MaR,MaP,EBF,EBR,EBP,Hamming_Loss):
    print(f"ACC = {acc}")
    print(f"MiF = {MiF}")
    print(f"MiR = {MiR}")
    print(f"MiP = {MiP}")
    print(f"MaF = {MaF}")
    print(f"MaR = {MaR}")
    print(f"MaP = {MaP}")
    print(f"EBF = {EBF}")
    print(f"EBR = {EBR}")
    print(f"EBP = {EBP}")
    print(f"Hamming_Loss = {Hamming_Loss}")

def writeCsv(acc,MiF,MiR,MiP,MaF,MaR,MaP,EBF,EBR,EBP,Hamming_Loss,p1,p2,p3,n1,n2,n3,
             csv_file_path = "experiment_results.csv",writeNewLine=False,writeHead=False):
    new_results = [
        {"Model": params.modelN_score, "P@1": p1, "P@3": p2, "P@5": p3, "nDCG@1": n1, "nDCG@3": n2, "nDCG@5": n3,
         "ACC": acc, "MIF": MiF, "MIP": MiR, "MIR": MiP,"MAF": MaF,"MAP": MaR, "MAR": MaP, "EBF": EBF, "EBP": EBR, "EBR": EBP,
         "HammingLoss": Hamming_Loss},
    ]

    with open(csv_file_path, mode='a', newline='') as csvfile:
        fieldnames = ["Model", "P@1", "P@3", "P@5","nDCG@1", "nDCG@3", "nDCG@5",
                      "ACC", "MIF", "MIP", "MIR", "MAF", "MAP", "MAR","EBF", "EBP", "EBR",
                      "HammingLoss"]

        writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
        if csvfile.tell() == 0:
            writer.writeheader()
        if (writeNewLine):
            writer.writerow({})
        if (writeHead):
            writer.writeheader()
        for result in new_results:
            writer.writerow(result)


if __name__ == '__main__':

    results = os.path.splitext(params.topKShowPath_score)[0] + '.npy'
    targets = os.path.splitext(params.tShowPath_score)[0] + '.npy'
    writeSocre_norm = False
    # writeSocre_norm = True
    acc=MiF=MiR=MiP=MaF= MaR= MaP= EBF= EBR= EBP= Hamming_Loss ="None"
    if(writeSocre_norm):
        acc, MiF, MiR, MiP, MaF, MaR, MaP, EBF, EBR, EBP, Hamming_Loss = socre_norm()
    printNormal(acc, MiF, MiR, MiP, MaF, MaR, MaP, EBF, EBR, EBP, Hamming_Loss)
    p1,p2,p3,n1,n2,n3=evl(results,targets)
    printEvl(p1,p2,p3,n1,n2,n3)

    csv_file_path = "experiment_results.csv"
    writeNewLine = False
    writeHead = False
    writeCsv(acc,MiF,MiR,MiP,MaF,MaR,MaP,EBF,EBR,EBP,Hamming_Loss,p1,p2,p3,n1,n2,n3,csv_file_path,writeNewLine,writeHead)

    print('score‘topKResultTag’读取路径',results)
    print('score‘targetResultTag’读取路径',targets)
    print('score‘实验保存’读取路径', csv_file_path)


