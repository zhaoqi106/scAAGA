# coding：<encoding name> ： # coding: utf-8
import pickle
import warnings
warnings.filterwarnings('ignore')
import numpy as np
import matplotlib.pyplot as plt
# from tqdm import tqdm
import tqdm
# from keras.models import load_model



def save_func(file_name,save_path):
    f = open(save_path, 'wb')
    pickle.dump(file_name, f)
    f.close()

def open_file(save_path):
    f = open(save_path, 'rb')
    open_name= pickle.load(f)
    f.close()
    return open_name

def save_text( file_name,save_path):
    file = open(save_path,'a')
    for i in range(len(file_name)):
        s = str(file_name[i]).replace('[','').replace(']','')#去除[],这两行按数据不同，可以选择
        s = s.replace("'",'').replace(',','') +'\n'   #去除单引号，逗号，每行末尾追加换行符
        file.write(s)
    file.close()
    print("保存成功") 

def F1_score(layer_output):
    avg_list=[]
    var_list=[]
    f1_list=[]
    # avg_max_index=[]
    # var_min_index=[]
    for i in range(len(layer_output[1])):
        avg=np.average(layer_output[:,i])
        var=np.var(layer_output[:,i])
        # if avg_list[i]>0.8:
        #     avg_max_index.append(i)
        # if var_list[i]<0.015:
        #     var_min_index.append(i)
        # gene_index=np.intersect1d(avg_max_index,var_min_index)
        f1 = 2*avg*(1-var)/(avg+(1-var))

        avg_list.append(avg)
        var_list.append(var)
        f1_list.append(f1)
    # f1_list.sort()

    return f1_list
    # return avg_list


# attention值绝对值后求f1score
def abs_f1score(layer_output):
    avg_list=[]
    # var_list=[]
    std_list=[]
    f1_list=[]
    # avg_max_index=[]
    # var_min_index=[]
    for i in range(len(layer_output[1])):
        avg=np.average(list(map(abs,layer_output[:,i])))
        std=np.std(list(map(abs,layer_output[:,i])))
        # var=np.var(layer_output[:,i])
        # if avg_list[i]>0.8:
        #     avg_max_index.append(i)
        # if var_list[i]<0.015:
        #     var_min_index.append(i)
        # gene_index=np.intersect1d(avg_max_index,var_min_index)

    
        f1 = 2*avg*(1-std)/(avg+(1-std))

        avg_list.append(avg)
        std_list.append(std)
        # var_list.append(var)
        f1_list.append(f1)
    # f1_list.sort()

    return f1_list
def loss_plot(autoencoder,save_path):
    plt.plot(autoencoder.history["loss"],'-o',label='Training_loss')
    plt.plot(autoencoder.history["val_loss"],'-o',label='Validation_loss')
    plt.legend(loc='upper right')
    plt.title("Loss of DPI Model")
    # plt.gcf().set_size_inches(5, 5)
    f = plt.gcf()
    f.savefig(save_path,dpi=300)
    f.clear()
    

def expend_inputmtx_outputmtx(temp_mtx, zero_rate=0.5, copytimes=3):
    temp_mtx = temp_mtx.copy()
    temp_input_mtx = np.zeros((temp_mtx.shape[0]*copytimes, temp_mtx.shape[1])).astype(np.float16)
    temp_output_mtx = np.zeros((temp_mtx.shape[0]*copytimes, temp_mtx.shape[1])).astype(np.float16)

    temp_new_cellindex = 0
    for ci in tqdm(range(copytimes)):
        for cell_index in range(temp_mtx.shape[0]):
            # temp_features_index = np.random.choice(temp_mtx.shape[1], size=temp_mtx.shape[1], replace=False, p=None)
            temp_cell = np.array(temp_mtx[cell_index]).astype(np.float16)#减少内存占用
            temp_output_mtx[temp_new_cellindex] = temp_cell
            temp_features_index = np.random.choice(temp_mtx.shape[1], size=int(temp_mtx.shape[1]*zero_rate), replace=False, p=None)  
            temp_cell[temp_features_index] = 0
            temp_input_mtx[temp_new_cellindex] = temp_cell
            temp_new_cellindex += 1
    del temp_mtx
    return temp_input_mtx, temp_output_mtx

def low_expend_inputmtx_outputmtx(temp_mtx, zero_rate=0.1, copytimes=3):
    temp_mtx = temp_mtx.copy()
    temp_input_mtx = np.zeros((temp_mtx.shape[0]*copytimes, temp_mtx.shape[1])).astype(np.float16)
    temp_output_mtx = np.zeros((temp_mtx.shape[0]*copytimes, temp_mtx.shape[1])).astype(np.float16)

    temp_new_cellindex = 0
    for ci in tqdm(range(copytimes)):
        for cell_index in range(temp_mtx.shape[0]):
            # temp_features_index = np.random.choice(temp_mtx.shape[1], size=temp_mtx.shape[1], replace=False, p=None)
            temp_cell = np.array(temp_mtx[cell_index]).astype(np.float16)#减少内存占用
            temp_output_mtx[temp_new_cellindex] = temp_cell
            nonzero_index = np.nonzero(temp_cell)
            zero_temp_cell= temp_cell[nonzero_index]
            test_index=[]
            # from j in temp_cell:
            #     if
            lower_q=np.quantile(zero_temp_cell,0.05,interpolation='lower') 
            # print(lower_q)
            for i in range(temp_mtx.shape[1]):
                if 0<temp_cell[i]<=lower_q:# 基因表达量大于0，小于细胞表达的5%
                    test_index.append(i)
            temp_features_index = np.random.choice(len(test_index), size=int((len(test_index))*zero_rate), replace=False, p=None)

            # temp_features_index = np.random.choice(temp_mtx.shape[1], size=int(temp_mtx.shape[1]*zero_rate), replace=False, p=None)  
            temp_cell[temp_features_index] = 0

            temp_input_mtx[temp_new_cellindex] = temp_cell
            temp_new_cellindex += 1
    del temp_mtx
    return temp_input_mtx, temp_output_mtx

def say():
    print("3")