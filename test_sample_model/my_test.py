from sklearn import metrics
import torch
from torch import nn
from torch import optim
from torch.nn import functional as F
import numpy as np
from graph_cmap_loader import *
import scipy.io as i
from my_args import *
import dgl
import xlwt
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score,average_precision_score
from sklearn.metrics import confusion_matrix

device = torch.device('cuda')


def create_variable(tensor):
    # Do cuda() before wrapping with variable
    if torch.cuda.is_available():
        return Variable(tensor.cuda())
    else:
        return Variable(tensor)


def test(model, device, loader):
    model.eval()
    total_preds = torch.Tensor()
    total_labels = torch.Tensor()
    total_preds_score=torch.Tensor()
    output_score_txt=[]
    output_txt=[]
    print('Make test for {} samples...'.format(len(loader.dataset)))
    with torch.no_grad():
        for batch_idx, (p1, p2, G1, dmap1, G2, dmap2, y) in enumerate(loader):
            print('p1:',p1)
            print('p2:',p2)
            print(y)
            
            pad_dmap1 = pad_dmap(dmap1)
            pad_dmap2 = pad_dmap(dmap2)
            predict_score_tensor= model(dgl.batch(G1), pad_dmap1, dgl.batch(G2), pad_dmap2)#代表输出的是正样本的概率分数
            #print('predict_score:', predict_score)
            #将tensor变量转化为numpy类型
            predict_score_numpy= predict_score_tensor.cpu().numpy()
            #将numpy类型转化为list类型
            predict_score_list=predict_score_numpy.tolist()
            
            predict_label_tensor = torch.round(predict_score_tensor.squeeze(1))#阈值为0.5
            #print('predict_label:', predict_label)
            predict_label_numpy = predict_label_tensor.cpu().numpy()
            #将numpy类型转化为list类型
            predict_label_list=predict_label_numpy.tolist()
            
            file = xlwt.Workbook('encoding = utf-8') 
            sheet1 = file.add_sheet('sheet1',cell_overwrite_ok=True)
             
            # 先填标题 
            # sheet1.write(a,b,c) 函数中参数a、b、c分别对应行数、列数、单元格内容
            sheet1.write(0, 0, "index") # 第1行第1列
            sheet1.write(0, 1, "receptor") # 第1行第2列
            sheet1.write(0, 2, "peptide") # 第1行第3列
            sheet1.write(0, 3, "label") # 第1行第4列
            sheet1.write(0, 4, "predict_score") # 第1行第5列
            sheet1.write(0, 5, "predict_label") # 第1行第6列
 
            # 循环填入数据
            for i in range(len(predict_score_list)):
                #print(predict_score[i])
                sheet1.write(i + 1, 0, i) # 第1列index
                sheet1.write(i + 1, 1, str(p1[i])) # 第2列receptor    
                sheet1.write(i + 1, 2, str(p2[i]))  # 第3列peptide
                sheet1.write(i + 1, 3, str(y[i])) # 第4列真实标签    
                sheet1.write(i + 1, 4, str(predict_score_list[i])) # 第5列预测分数    
                sheet1.write(i + 1, 5, str(predict_label_list[i]))  # 第6列预测标签
 
            # 保存Excel到.py源文件同级目录
            file.save('./results/independent test set testing/'+str(batch_idx)+'.xls')

            total_labels = torch.cat((total_labels.cpu(), y.float().cpu()), 0)
            total_preds_score = torch.cat((total_preds_score.cpu(), predict_score_tensor.cpu()), 0)
            total_preds = torch.cat((total_preds.cpu(), predict_label_tensor.cpu()), 0)

    #return 0
    return total_labels.cpu().numpy().flatten(), total_preds.cpu().numpy().flatten(), total_preds_score.cpu().numpy().flatten()

   

def predicting():
    attention_model = testArgs['model']#from my_args import *
    #attention_model = TheAttention_modelClass(*args, **kwargs)
    checkpoint = torch.load('./save_model_pkl/TPepPro_receptor-peptide(train 19187 pairs)_GAT.pkl')#训练好的模型参数
    attention_model.load_state_dict({k.replace('module.', ''): v for k, v in checkpoint.items()}, strict=False)
    #attention_model.state_dict(torch.load('/home/zhongle/TAGPPI-main/model_pkl/GATepoch4.pkl'))
    #attention_model.eval()
    print('------------------开始预测------------------------')
    #test(attention_model, device, test_loader)
    total_labels, total_preds, total_preds_score = test(attention_model, device, test_loader)#from graph_cmap_loader_test import *
    test_acc = accuracy_score(total_labels, total_preds)
    test_prec = precision_score(total_labels, total_preds)
    test_recall = recall_score(total_labels, total_preds)
    test_f1 = f1_score(total_labels, total_preds)
    test_auc = roc_auc_score(total_labels, total_preds_score)
    con_matrix = confusion_matrix(total_labels, total_preds)
    test_spec = con_matrix[0][0] / (con_matrix[0][0] + con_matrix[0][1])
    test_auprc = average_precision_score(total_labels, total_preds_score)
    test_sen = con_matrix[1, 1] / (con_matrix[1, 1] + con_matrix[1, 0])
    print("acc: ", test_acc, " ; prec: ", test_prec, " ; recall: ", test_recall, " ; f1: ", test_f1, " ; auc: ", test_auc, " ; spec:", test_spec, " ; auprc: ", test_auprc, " ; sensitivity: ", test_sen)
    with open(rst_file, 'a+') as fp:
        fp.write('acc=' + str(test_acc) + '\tprec=' + str(test_prec) + '\trecall=' + str(test_recall) + '\tf1=' + str(test_f1) + '\tauc=' + str(test_auc) + '\tspec=' + str(test_spec) +  '\tauprc=' + str(test_auprc) + '\tsensitivity=' + str(test_sen) + '\n')
    print('------------------预测完成------------------------')

