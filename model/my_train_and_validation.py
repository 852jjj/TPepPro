from sklearn import metrics
import torch
from torch import nn
from torch import optim
from torch.nn import functional as F
import numpy as np
from graph_cmap_loader import *
from TAGlayer import *
from my_args import *
import dgl
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score,average_precision_score
from sklearn.metrics import confusion_matrix
from sklearn.model_selection import StratifiedKFold
import numpy as np
import xlwt

device = torch.device('cuda')


def create_variable(tensor):
    # Do cuda() before wrapping with variable
    if torch.cuda.is_available():
        return Variable(tensor.cuda())
    else:
        return Variable(tensor)




def validation(model, device, loader):
    model.eval()
    total_preds = torch.Tensor()
    total_labels = torch.Tensor()
    total_preds_score=torch.Tensor()
    print('Make validation for {} samples...'.format(len(loader.dataset)))
    with torch.no_grad():
        for batch_idx, (p1, p2, G1, dmap1, G2, dmap2, y) in enumerate(loader):
            #print(y)
            pad_dmap1 = pad_dmap(dmap1)
            pad_dmap2 = pad_dmap(dmap2)
            output_score= model(dgl.batch(G1), pad_dmap1, dgl.batch(G2), pad_dmap2)#代表输出的是正样本的概率分数
            #print(output_score)
            output = torch.round(output_score.squeeze(1))#阈值为0.5
            #print(output)#加的
            
            total_labels = torch.cat((total_labels.cpu(), y.float().cpu()), 0)
            total_preds_score = torch.cat((total_preds_score.cpu(), output_score.cpu()), 0)
            total_preds = torch.cat((total_preds.cpu(), output.cpu()), 0)
            

    return total_labels.cpu().numpy().flatten(), total_preds.cpu().numpy().flatten(),total_preds_score.cpu().numpy().flatten()

def test(model, device, loader,k):
    model.eval()
    total_preds = torch.Tensor()
    total_labels = torch.Tensor()
    total_preds_score=torch.Tensor()
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
            sheet1.write(0, 0, "序号") # 第1行第1列
            sheet1.write(0, 1, "receptor") # 第1行第2列
            sheet1.write(0, 2, "peptide") # 第1行第3列
            sheet1.write(0, 3, "label") # 第1行第4列
            sheet1.write(0, 4, "predict_score") # 第1行第5列
            sheet1.write(0, 5, "predict_label") # 第1行第6列
 
            # 循环填入数据
            for i in range(len(predict_score_list)):
                #print(predict_score[i])
                sheet1.write(i + 1, 0, i) # 第1列序号
                sheet1.write(i + 1, 1, str(p1[i])) # 第2列receptor    
                sheet1.write(i + 1, 2, str(p2[i]))  # 第3列peptide
                sheet1.write(i + 1, 3, str(y[i])) # 第4列真实标签    
                sheet1.write(i + 1, 4, str(predict_score_list[i])) # 第5列预测分数    
                sheet1.write(i + 1, 5, str(predict_label_list[i]))  # 第6列预测标签
 
            # 保存Excel到.py源文件同级目录
            file.save('./results/train_5fold_cross_validation/'+'fold'+str(k)+'/'+str(batch_idx)+'.xls')
            

            
            total_labels = torch.cat((total_labels.cpu(), y.float().cpu()), 0)
            total_preds_score = torch.cat((total_preds_score.cpu(), predict_score_tensor.cpu()), 0)
            total_preds = torch.cat((total_preds.cpu(), predict_label_tensor.cpu()), 0)
            

    return total_labels.cpu().numpy().flatten(), total_preds.cpu().numpy().flatten(),total_preds_score.cpu().numpy().flatten()


def train(trainArgs):
    

    all_protein1 = []
    all_protein2 = []
    all_Y = []
    with open('data/actions/xxx.actions.tsv', 'r') as f:
        all_lines = f.readlines()

        for line in all_lines:
            row = line.rstrip().split('\t')
            all_protein1.append(row[0])
            all_protein2.append(row[1])
            all_Y.append(float(row[2]))
    k=0      
    Skf = StratifiedKFold(n_splits=5, shuffle=True,random_state=42)
    for split, ( train_index, test_index) in enumerate( Skf.split(all_Y,all_Y)):
        k=k+1
        print('第',k,'折:')
        '''print(train_index)
        print(test_index)'''
        #train_valid_dataset
        train_valid_protein1_cv = np.array(all_protein1)[train_index]
        train_valid_protein2_cv = np.array(all_protein2)[train_index]
        train_valid_Y_cv = np.array(all_Y)[train_index]
        
        '''print(train_valid_protein2_cv)
        print(train_valid_protein1_cv)
        print(train_valid_Y_cv)'''

        #test_dataset
        test_protein1_cv = np.array(all_protein1)[test_index]
        test_protein2_cv = np.array(all_protein2)[test_index]
        test_Y_cv = np.array(all_Y)[test_index]
       
        '''print(test_protein2_cv)
        print(test_protein1_cv)
        print(test_Y_cv)'''

        train_size = train_valid_protein2_cv.shape[0]
        print("训练集和验证集的蛋白质对为:",train_size)
        valid_size =  int(train_size*0.05) 
        print("验证集的蛋白质对为:",valid_size)

        train_protein2_cv = np.concatenate((train_valid_protein2_cv[:int(train_size/2-valid_size/2)], train_valid_protein2_cv[int(train_size/2+valid_size/2):]), axis=0)
        train_protein1_cv = np.concatenate((train_valid_protein1_cv[:int(train_size/2-valid_size/2)], train_valid_protein1_cv[int(train_size/2+valid_size/2):]), axis=0)
        train_Y_cv = np.concatenate((train_valid_Y_cv[:int(train_size/2-valid_size/2)], train_valid_Y_cv[int(train_size/2+valid_size/2):]), axis=0)
        print( "训练集的蛋白质对为：",train_protein2_cv.shape[0])
        '''print(train_protein2_cv)
        print(train_protein2_cv[0])'''
        '''print(train_protein1_cv)
        print(train_Y_cv)'''

        valid_protein2_cv = train_valid_protein2_cv[int(train_size/2-valid_size/2):int(train_size/2+valid_size/2)]
        valid_protein1_cv = train_valid_protein1_cv[int(train_size/2-valid_size/2):int(train_size/2+valid_size/2)]
        valid_Y_cv = train_valid_Y_cv[int(train_size/2-valid_size/2):int(train_size/2+valid_size/2)]
        print("验证集的蛋白质对为：",valid_protein2_cv.shape[0])
        

        train_ds = MyDataset(list1=train_protein1_cv, list2=train_protein2_cv, list3=train_Y_cv)
        valid_ds = MyDataset(list1=valid_protein1_cv, list2=valid_protein2_cv, list3=valid_Y_cv)
        test_ds = MyDataset(list1=test_protein1_cv, list2=test_protein2_cv, list3=test_Y_cv)

        train_loader = DataLoader(train_ds, batch_size = batchsize, shuffle=True,drop_last = False,collate_fn=collate)
        test_loader = DataLoader(test_ds, batch_size = batchsize, shuffle=True,drop_last = False,collate_fn=collate)
        validation_loader = DataLoader(valid_ds, batch_size = batchsize, shuffle=True,drop_last = False,collate_fn=collate)


    
    
        train_losses = []
        train_accs = []
        max_acc=0
        trainArgs['model'] = GATPPI(modelArgs).cuda()
        
        trainArgs['optimizer'] = torch.optim.Adam(trainArgs['model'].parameters(),lr=trainArgs['lr'])
        
        optimizer = trainArgs['optimizer']
        trainArgs['lr_scheduler'] = torch.optim.lr_scheduler.StepLR(optimizer, step_size = 20, gamma = 0.1, last_epoch=-1)
        
        criterion = torch.nn.BCELoss()
        attention_model = trainArgs['model']
        #trainArgs['optimizer'] = torch.optim.Adam(trainArgs['model'].parameters(),lr=trainArgs['lr'])
        
        for i in range(trainArgs['epochs']):
            print("Running EPOCH", i + 1)
            total_loss = 0
            #print(total_loss)
            n_batches = 0
            correct = 0
            #train_loader = trainArgs['train_loader']
            
          
            for batch_idx, (p1, p2, G1, dmap1, G2, dmap2, y) in enumerate(train_loader):
                y_pred = attention_model(dgl.batch(G1), pad_dmap(dmap1), dgl.batch(G2), pad_dmap(dmap2))
                correct += torch.eq(torch.round(y_pred.type(torch.DoubleTensor).squeeze(1)),
                                y.type(torch.DoubleTensor)).data.sum()
                loss = criterion(y_pred.type(torch.DoubleTensor).squeeze(1), y.type(torch.DoubleTensor))
                total_loss += loss.data
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
                n_batches += 1
            #print(total_loss)
            #print(n_batches)
            avg_loss = total_loss / n_batches
            acc = correct.numpy() / (len(train_loader.dataset))

            train_losses.append(avg_loss)
            train_accs.append(acc)
            print("train avg_loss is", avg_loss)
            print("train ACC = ", acc)
        
            # validation
            total_labels, total_preds,total_preds_score = validation(attention_model, device, validation_loader)
            validation_acc = accuracy_score(total_labels, total_preds)
            validation_prec = precision_score(total_labels, total_preds)
            validation_recall = recall_score(total_labels, total_preds)
            validation_f1 = f1_score(total_labels, total_preds)
            validation_auc = roc_auc_score(total_labels, total_preds_score)
            validation_auprc=average_precision_score(total_labels, total_preds_score)
        
            con_matrix = confusion_matrix(total_labels, total_preds)
            validation_sen= con_matrix[1,1]/(con_matrix[1,1]+con_matrix[1,0])
            validation_spec = con_matrix[0][0] / (con_matrix[0][0] + con_matrix[0][1])
            validation_mcc = (con_matrix[0][0]*con_matrix[1][1]-con_matrix[0][1]*con_matrix[1][0])/(((con_matrix[1][1]+con_matrix[0][1])*(con_matrix[1][1]+con_matrix[1][0])*(con_matrix[0][0]+con_matrix[0][1])*(con_matrix[0][0]+con_matrix[1][0]))**0.5)
            print("acc: ", validation_acc, " ; prec: ", validation_prec, " ; recall: ", validation_recall, " ; f1: ", validation_f1, " ; auc: ",validation_auc, " ; spec:", validation_spec, " ; mcc: ", validation_mcc," ; auprc: ", validation_auprc," ; sensitivity: ", validation_sen)
            with open(rst_file, 'a+') as fp:
             fp.write('epoch:' + str(i + 1) + '\ttrainacc=' + str(acc) + '\ttrainloss=' + str(avg_loss.item()) + '\tacc=' + str(validation_acc) + '\tprec=' + str(validation_prec) + '\tf1=' + str(validation_f1) + '\tauc=' + str(validation_auc) + '\tspec=' + str(validation_spec) + '\tmcc=' + str(validation_mcc) +'\tauprc='+str(validation_auprc)+'\tsensitivity=' + str(validation_sen)+'\n')
            if  validation_acc>max_acc:
                max_acc=validation_acc
                print("save model")
                torch.save(attention_model.state_dict(), pkl_path + '.pkl')
        #test
        attention_model = trainArgs['model']#from my_args import *
        #attention_model = TheAttention_modelClass(*args, **kwargs)
        checkpoint = torch.load('./model_pkl/GAT.pkl')#更改这里
        attention_model.load_state_dict({k.replace('module.', ''): v for k, v in checkpoint.items()}, strict=False)
        #attention_model.state_dict(torch.load('/home/zhongle/TAGPPI-main/model_pkl/GATepoch4.pkl'))
        #attention_model.eval()
        total_labels, total_preds, total_preds_score = test(attention_model, device, test_loader,k)#from graph_cmap_loader_test import *
        test_acc = accuracy_score(total_labels, total_preds)
        test_prec = precision_score(total_labels, total_preds)
        test_recall = recall_score(total_labels, total_preds)
        test_f1 = f1_score(total_labels, total_preds)
        test_auc = roc_auc_score(total_labels, total_preds_score)
        con_matrix = confusion_matrix(total_labels, total_preds)
        test_spec = con_matrix[0][0] / (con_matrix[0][0] + con_matrix[0][1])
        test_auprc=average_precision_score(total_labels, total_preds_score)
        test_sen= con_matrix[1,1]/(con_matrix[1,1]+con_matrix[1,0])
        test_mcc = (con_matrix[0][0]*con_matrix[1][1]-con_matrix[0][1]*con_matrix[1][0])/(((con_matrix[1][1]+con_matrix[0][1])*(con_matrix[1][1]+con_matrix[1][0])*(con_matrix[0][0]+con_matrix[0][1])*(con_matrix[0][0]+con_matrix[1][0]))**0.5)
        print("acc: ",test_acc," ; prec: ",test_prec," ; recall: ",test_recall," ; f1: ",test_f1," ; auc: ",test_auc," ; spec:",test_spec," ; mcc: ",test_mcc," ; auprc: ",test_auprc," ; sensitivity: ",test_sen)
        with open(rst_file, 'a+') as fp:
            fp.write('acc=' + str(test_acc) + '\tprec=' + str(test_prec) + '\trecall=' + str(test_recall)+ '\tf1=' + str(test_f1) + '\tauc=' + str(test_auc) + '\tspec='+str(test_spec)+ '\tmcc='+str(test_mcc)+'\tauprc='+str(test_auprc)+'\tsensitivity=' + str(test_sen)+'\n')