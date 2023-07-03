# coding：<encoding name> ： # coding: utf-8
from funct import loss_plot,open_file,save_func,save_text,F1_score
import scanpy as sc
import scanpy as sc
import numpy as np
import tensorflow as tf
# from tensorflow import keras
# import keras
from keras.callbacks import EarlyStopping
from keras.layers import Input, Dense,Activation
from keras.models import Model
from keras import optimizers
from keras.callbacks import  ModelCheckpoint
import matplotlib.pyplot as plt
import scipy.stats
from sklearn.metrics.cluster import adjusted_mutual_info_score,normalized_mutual_info_score
from sklearn.metrics import adjusted_rand_score
from sklearn.cluster import KMeans
from tqdm import tqdm


# #true
# filepath = "COVID19_Healthy.h5ad"   # n_obs × n_vars = 97039 × 24737
# scobj = sc.read_h5ad(filepath)
# scobj.var_names_make_unique
# sc.pp.filter_cells(scobj, min_genes=200)
# sc.pp.filter_genes(scobj, min_cells=3)
# scobj.var['mt'] = scobj.var_names.str.startswith('MT-')  # annotate the group of mitochondrial genes as 'mt'
# sc.pp.calculate_qc_metrics(scobj, qc_vars=['mt'], percent_top=None, log1p=False, inplace=True)
# sc.pl.violin(scobj, ['n_genes_by_counts', 'total_counts', 'pct_counts_mt'], jitter=0.4, multi_panel=True)
# sc.pp.normalize_total(scobj,target_sum=1e4)
# sc.pp.log1p(scobj)
# feature_count= 3000
# # feature_count= 1000
# sc.pp.highly_variable_genes(
#     scobj,
#     n_top_genes=feature_count,
#     flavor="seurat_v3",
#     subset=False
# )
# # sc.pl.highly_variable_genes(scobj)
# scobj=scobj[:,scobj.var["highly_variable"]]
# sample_cells = scobj.X.copy()     # all


def expend_inputmtx_outputmtx(temp_mtx, zero_rate=0.5, copytimes=20):
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

#test
scobj=sc.read_h5ad("PBMC_COVID19_Healthy_Annotated.h5ad")
sample_size = 25000
sample_cells_index = np.random.choice(scobj.shape[0], size=sample_size, replace=False, p=None)

notsampled_cells_index = np.intersect1d(range(scobj.shape[0]), sample_cells_index)
notsampled_cells_index = np.delete(range(scobj.shape[0]),notsampled_cells_index)

sample_cells = scobj.X[sample_cells_index].copy()
notsampled_cells = scobj.X[notsampled_cells_index].copy()

# select_index=open_file('pkl/5000MH8919332sample_cells_index.pkl')
# select_notsample_index=open_file('pkl/3000MH8919332not_sampled_cells_index.pkl')
# sample_cells = scobj.X.copy()  
# sample_cells_index=select_index
# notsampled_cells_index=select_notsample_index
# sample_cells= scobj.X[select_index].copy()
# notsampled_cells = scobj.X[select_notsample_index].copy()

# expend_inputmtx, expend_outputmtx = expend_inputmtx_outputmtx(sample_cells, copytimes=10)
# new_expinput=open_file('pkl/5000MH8919332sample_expend_inputmtx.pkl')
# new_expoutput=open_file('pkl/5000MH8919332sample_expend_outputmtx.pkl')
# expend_inputmtx4 = expend_inputmtx
# expend_outputmtx4 = expend_outputmtx

feature_count=4000
encoding_dim_exp3 = 64

#sigmoid
input_exp3 = Input(shape=(feature_count,))
input_sigmoid_exp3= Dense(feature_count, activation='sigmoid',name="gene_sigmoid")(input_exp3)
input_merge_exp3=tf.keras.layers.Multiply()([input_exp3,input_sigmoid_exp3])
encoded_exp3 = Dense(512, activation='relu')(input_merge_exp3)
encoded_exp3 = Dense(128, activation='relu')(encoded_exp3)
latent_exp3 = Dense(encoding_dim_exp3, activation='linear', name="latent")(encoded_exp3)

decoded_exp3 = Dense(feature_count, activation='relu', name="decoded")(latent_exp3)

autoencoder_exp3 = Model(input_exp3, decoded_exp3)
model_checkpoint_callback = ModelCheckpoint(filepath = "model_sig/model/weights_{epoch:04d}-{loss:.4f}.h5", 
							monitor='loss', 
							verbose=2, 
							save_best_only=False, 
							save_weights_only=False, 
							mode='auto', 
							period=50
                                           )
es = EarlyStopping(monitor="val_loss", patience=16, verbose=2)
# es = EarlyStopping(monitor="loss", patience=3, verbose=2)
autoencoder_exp3.compile(optimizer=tf.optimizers.Nadam(learning_rate=0.0001, beta_1=0.9, beta_2=0.999, epsilon=1e-05, clipnorm=1), loss='msle')
temp_index = np.random.choice(sample_cells.shape[0], size=sample_cells.shape[0], replace=False, p=None)
estimator_exp3 = autoencoder_exp3.fit(x=[sample_cells[temp_index]], y=[sample_cells[temp_index]],
                epochs=200,
                batch_size=768,
                verbose=2,
                shuffle=True,
                callbacks=[es,model_checkpoint_callback],
                validation_data=([notsampled_cells], [notsampled_cells])
                )
# Model.save('my_moedl')
loss_plot(estimator_exp3,save_path="model_sig/exp3_loss.png")

autoencoder_exp3_latent = Model(inputs=autoencoder_exp3.inputs, outputs=autoencoder_exp3.get_layer("latent").output).predict([sample_cells],verbose=2)
autoencoder_exp3_decoded = Model(inputs=autoencoder_exp3.inputs, outputs=autoencoder_exp3.get_layer("decoded").output).predict([sample_cells],verbose=2)
autoencoder_exp3_attention = Model(inputs=autoencoder_exp3.inputs, outputs=autoencoder_exp3.get_layer("gene_sigmoid").output).predict([sample_cells],verbose=2)
# save_text(autoencoder_exp3_attention,"attention_tanh.txt")

plt.hist(sample_cells[:,54])
plt.hist(autoencoder_exp3_decoded[:,54], alpha=0.5)
f = plt.gcf()  #获取当前图像
f.savefig("model_sig/54.png")
f.clear()  #释放内存
s=scipy.stats.spearmanr(sample_cells[:,54], autoencoder_exp3_decoded[:,54])
print("54_spearmanr=",s)
plt.hist(sample_cells[:,129])
plt.hist(autoencoder_exp3_decoded[:,129], alpha=0.5)
f = plt.gcf()  #获取当前图像
f.savefig("model_sig/129.png")
f.clear()  #释放内存
s=scipy.stats.spearmanr(sample_cells[:,129], autoencoder_exp3_decoded[:,129])
print("129_spearmanr=",s)
scobj_autoencoder_exp3_raw_latent = sc.AnnData(autoencoder_exp3_latent, obs=scobj[sample_cells_index].obs)
# scobj_autoencoder_exp3_raw_latent = sc.AnnData(autoencoder_exp3_latent, obs=scobj.obs)

sc.pp.neighbors(scobj_autoencoder_exp3_raw_latent, n_neighbors=10)
sc.tl.leiden(scobj_autoencoder_exp3_raw_latent)
sc.tl.umap(scobj_autoencoder_exp3_raw_latent)
sc.pl.umap(scobj_autoencoder_exp3_raw_latent, color=["initial_clustering"],save=".ini_umap.pdf")

# scobj_merge_raw = scobj[sample_cells_index]
scobj_exp3 = scobj[sample_cells_index]
scobj_exp3.layers["raw_decode"] = autoencoder_exp3_decoded
scobj_exp3.obsm["X_umap"] = scobj_autoencoder_exp3_raw_latent.obsm["X_umap"]
# sc.pl.umap(scobj_exp3, color=["initial_clustering"])
sc.pl.umap(scobj_exp3, color=["ISG15"],save=".isg15_input.pdf")
sc.pl.umap(scobj_exp3, color=["ISG15"], layer="raw_decode",save=".isg15_output.pdf")
labels_full = scobj[sample_cells_index].obs["full_clustering"]
labels_ini = scobj[sample_cells_index].obs["initial_clustering"]
# labels_full = scobj.obs["full_clustering"]
# labels_ini = scobj.obs["initial_clustering"]
NMI=normalized_mutual_info_score(scobj_autoencoder_exp3_raw_latent.obs["leiden"], labels_ini)
print("NMI=",NMI)
kmeans_merge_raw = KMeans(n_clusters=15, random_state=0).fit(autoencoder_exp3_latent)
kmeans_nmi=normalized_mutual_info_score(kmeans_merge_raw.labels_, labels_ini)
print("kmeans_nmi=",kmeans_nmi)
ami=adjusted_mutual_info_score(scobj_autoencoder_exp3_raw_latent.obs["leiden"], labels_ini)
print("ami=",ami)
ari=adjusted_rand_score(scobj_autoencoder_exp3_raw_latent.obs["leiden"], labels_ini)
print("ari=",ari)

scobj_exp3_attention = sc.AnnData(autoencoder_exp3_attention, obs=scobj[sample_cells_index].obs,var=scobj[sample_cells_index].var)
scobj_exp3_attention.obs = scobj_exp3_attention.obs.reset_index()

ini_index=[] #每个细胞类群的index
ini_celltypes=np.array(scobj_exp3_attention.obs.initial_clustering.unique())
len=len(ini_celltypes)
# ini_celltypes=['B_cell','CD14','CD16','CD4','CD8','DCs','HSC','Lymph_prolif','MAIT',
                # 'NK_16hi','NK_56hi','Plasmablast','Platelets','Treg','gdT','pDC']
for i in range(len):
    ini_index.append(scobj_exp3_attention.obs[scobj_exp3_attention.obs['initial_clustering']==ini_celltypes[i]].index)

# 每个细胞群的f1score,并进行排序
each_cell=[] #每个细胞类群表达量

each_cell_f1score=[] #16个细胞类群，每种gene的f1score
gene_index=[] #每种gene的f1score排序
gene_index_unorder=[]
# ncell=0
for i in range(len):
    
    each_cell.append(scobj_exp3_attention.X[ini_index[i]])
    # ncell+=1
    each_cell_f1score.append(F1_score(each_cell[i]))
    b=sorted(enumerate(each_cell_f1score[i]), key=lambda x:x[1],reverse = True)  # x[1]是因为在enumerate(a)中，a数值在第1位
    c=sorted(enumerate(each_cell_f1score[i]), key=lambda x:x[1])  # 从低到高
    gene_index.append([x[0] for x in b])
    gene_index_unorder.append([x[0] for x in c])

 # 记录每个细胞群f1core排名靠前的gene name
gene_name=[]
for i in range(len):
  name=[]
  for j in gene_index[i][:50]:
    name.append(scobj_exp3_attention.var.index[j])
  gene_name.append(name)   
    
# 保存每个细胞群里top50的基因
for i in range(len):
  j=i+1
  # text_save("cell_%s_gene.txt" %j,gene_name[i])
  save_text(gene_name[i],"model_sig/cell_%s_gene.txt" %j)

  # 记录每个细胞群f1core排名靠后的gene name
gene_name2=[]
for i in range(len):
  name2=[]
  for j in gene_index_unorder[i][:50]:
    name2.append(scobj_exp3_attention.var.index[j])
  gene_name2.append(name)   
    
# 保存每个细胞群里top50的基因
for i in range(len):
  j=i+1
  # text_save("cell_%s_gene.txt" %j,gene_name[i])
  save_text(gene_name2[i],"model_sig/unorder_cell_%s_gene.txt" %j)
  




