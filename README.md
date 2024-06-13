# TPepPro
Prediction of peptide-protein interactions based on Transformer.
***
# Install Dependencies
Python ver. == 3.7  
For others, run the following command:   
```
pip install torch
pip install scikit-learn -i https://pypi.tuna.tsinghua.edu.cn/simple/
conda install pytorch torchvision torchaudio cudatoolkit=10.1 -c pytorch
conda install pytorch==1.5.1 torchvision==0.6.1 cudatoolkit=10.1 -c pytorch
pip install dgl_cu101-0.6.1-cp37-cp37m-manylinux1_x86_64.whl 

```
***
# Run
## 1.For the proteins and peptides in the input sample pairs, they need to be preprocessed first to generate their respective amino acid vector representations and contact map files. The specific steps are as follows:  
Generating amino acid vectors for proteins and peptides means that you need to store all the amino acid sequences of proteins and peptides in the sample pair into a **fasta** file, then write the path of this fasta file to the corresponding location in the **data pre-processing/generate_embeddings.py** code, and run ```python generate_embeddings. py``` can generate an **npz** file that stores the amino acid vectors of all proteins and peptides in the sample pair (the amino acid vector representation of each protein/peptide generated is stored in the **npy** file, and the vector of the generated protein/peptide is represented as **L*1024**, where **L** is the sequence length of the protein/peptide).

To generate contact map files for proteins and peptides, you need to first find the pdb files of the proteins and peptides in Alphafold, and then write them to TPepPro according to the corresponding **txt** file of the sequences of all proteins and peptides used and the **pdb** file of the proteins and peptides in all sample pairs
In the corresponding location of the **data pre-processing/generate_contact_map.py** code, you can run ```python generate_contact_map.py``` to generate contact map files for proteins and peptides, save them in **npz** format, and the shape of the contact map is **L*L** (**L** is the sequence length).

***
## 2.Model testing and training
We have provided model files trained using peptide-protein complexes from the Propedia database (http://bioinfo.dcc.ufmg.br/propedia2/index.php/download) as positive samples, along with an independent test set. 
To run model predictions, Please execute the following command：  
**cd test_indep_model**
```
python my_main_test.py
```
The input to this sample run is the receptor-peptide pair used to perform the test (the receptor-peptide pair here is selected from some of the sample pairs ofthe receptor-peptide dataset we used in the paper), which are saved in a TSV type file.  
Data in are organized with the following columns：**test_sample_model/data/actions/sample_cmap.actions.tsv**  
**receptor ID  peptide ID  label**   

The input of the test sample run also includes the preprocessed amino acid vector representations of these proteins/peptides and their corresponding contact map files, and then their corresponding amino acid vector representations are stored in the form of npy files in the **test_sample_model/data/sample_embeddings.npz** file, and their contact map files are stored in the format of npz files **test_sample_model/data/sample_cmap folder**.  

Take, for example, **1a1m_A 1a1m_C 1** in the sample pair used for testing.  
**The amino acid vector representation of the generated receptor 1a1m_A is in the form of: **  
[[ 0.15398422 -0.23978221 -0.01549047 ... -0.18346733 -0.09291625
  -0.01021743]  
 [ 0.4489373  -0.02665013 -0.17931005 ... -0.1891016   0.1553362
   0.321806  ]  
 [-0.02809518  0.05845888 -0.495727   ... -0.42856818  0.06753138
  -0.07047243]  
 ...  
 [ 0.3649687   0.0686414   0.17520183 ...  0.0492078   0.12580633
   0.12104917]  
 [ 0.05224043 -0.10220779  0.27868733 ... -0.07296755  0.20231174
   0.35552794]  
 [-0.2167282  -0.02923655  0.09628326 ... -0.01234592 -0.02105464
  -0.09572704]]    
**The amino acid vector of the generated receptor 1a1m_A is represented in the shape of: **(278, 1024)  
**The representation of the contact map file of the generated receptor 1a1m_A is：**  
[[0 1 1 ... 0 0 0]  
 [1 0 1 ... 0 0 0]  
 [1 1 0 ... 0 0 0]  
 ...  
 [0 0 0 ... 0 1 1]  
 [0 0 0 ... 1 0 1]  
 [0 0 0 ... 1 1 0]]   
**The shape of the generated receptor 1a1m_A contact map file is：** (278, 278)  
**The amino acid vector representation of the generated peptide 1a1m_C is in the form of：**  
[[ 0.18085133 -0.07571788 -0.19935569 ... -0.01112768 -0.06433804
   0.22170882]  
 [ 0.16378656  0.1083569  -0.11270649 ...  0.14606047 -0.1882304
  -0.00713845]  
 [ 0.06448993 -0.02783195 -0.03839097 ...  0.12223947 -0.06157787
   0.08955193]  
 ...  
 [ 0.2744665  -0.06482965 -0.09515045 ...  0.03673784 -0.08219997
  -0.13130511]  
 [ 0.26488724  0.00398306 -0.19564752 ... -0.02756201 -0.05639829
   0.10974102]  
 [ 0.06472415  0.0295378   0.04856678 ...  0.09662343 -0.1039608
  -0.12716724]]  
**The amino acid vector of the generated peptide 1a1m_C is represented in the shape of：** (9, 1024)  
**The contact map file of the generated peptide 1a1m_C is represented as：**  
[[0 1 1 0 0 0 0 0 0]  
 [1 0 1 1 0 0 0 0 0]  
 [1 1 0 1 1 0 0 0 0]  
 [0 1 1 0 1 1 0 0 0]  
 [0 0 1 1 0 1 1 0 0]  
 [0 0 0 1 1 0 1 1 0]  
 [0 0 0 0 1 1 0 1 1]  
 [0 0 0 0 0 1 1 0 1]  
 [0 0 0 0 0 0 1 1 0]]  
**The shape of the contact map file of the generated peptide 1a1m_C is：** (9, 9)  
The output of the test sample run is the prediction result of the test sample pair saved in the form of an xls file, which is in the following format:  
**序号    receptor	peptide	label	predict_score	predict_label**  
**序号：** The ordinal number used to represent the sample pair.  
**receptor：**receptor ID  
**peptide：**peptide ID  
**label：** Indicates the classification of sample pairs, if the label is 1, the sample pair will be marked as interactive, and if the label is 0, the sample pair will be marked as having no interaction.  
**predict_score：** Marking the possibility of interaction between sample pairs, here we set the threshold to 0.5, if the prediction score of the model for sample pairs is greater than or equal to 0.5, then predict that there is interaction between sample pairs, and vice versa, predict that there is no interaction between sample pairs.  
**predict_label：**indicates whether there is interaction in the prediction of the sample pair, if the predict label is 1, the sample pair is predicted to have interaction, if the predict label is 0, it is predicted that the sample pair has no interaction.  
***
To train your own data, after generating the required files, execute the following command to perform 5-fold cross-validation.  
**cd model**
```
python my_main.py
```
