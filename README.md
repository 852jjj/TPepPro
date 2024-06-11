# TPepPro

Prediction of peptide-protein interactions based on Transformer.

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

# Run

## 1.Generate amino acid embedding and contact map files, run the following code separately.

cd data pre-processing

```
python generate_embeddings.py
python generate_contact_map.py
```

## 2.Model testing and training

We have provided model files trained using peptide-protein complexes from the Propedia database (http://bioinfo.dcc.ufmg.br/propedia2/index.php/download) as positive samples, along with an independent test set. 
To run model predictions, Please execute the following command：  
cd test_indep_model

```
python my_main_test.py
```

To train your own data, after generating the required files, execute the following command to perform 5-fold cross-validation.  
cd model

```
python my_main.py
```