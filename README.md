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
### Instructions for Running the Model

#### 1. Preprocessing Proteins and Peptides

To preprocess the proteins and peptides in the input sample pairs, generate their respective amino acid vector representations and contact map files. Follow steps below:

##### Generating Amino Acid Vectors

1. Store all the amino acid sequences of proteins and peptides in the sample pair into a FASTA file.
2. Write the path of this FASTA file into the appropriate location in the `data_preprocessing/generate_embeddings.py` script.
3. Run the script with the command `python generate_embeddings.py` to generate an NPZ file that stores the amino acid vectors of all proteins and peptides in the sample pair. Each protein/peptide vector representation is stored in an NPY file and has the shape `L*1024`, where `L` is the sequence length of the protein/peptide.

##### Generating Contact Map Files

1. Obtain the PDB files of the proteins and peptides from AlphaFold.
2. Write these files into the appropriate locations in the `data_preprocessing/generate_contact_map.py` script, along with the corresponding sequence TXT files.
3. Run the script with the command `python generate_contact_map.py` to generate contact map files for the proteins and peptides in NPZ format. The shape of the contact map is `L*L` (where `L` is the sequence length).

#### 2. Model Training and Testing

We have provided model files trained using peptide-protein complexes from the [Propedia database](http://bioinfo.dcc.ufmg.br/propedia2/index.php/download) as positive samples, along with an independent test set. To run model predictions, execute the following command:

```sh
cd test_indep_model
python my_main_test.py
```

##### Input for the Sample Run

The input for this sample run includes the receptor-peptide pairs used for testing, saved in a TSV file (`test_sample_model/data/actions/sample_cmap.actions.tsv`) with the following columns:

- `receptor ID`
- `peptide ID`
- `label`

The preprocessed amino acid vector representations and corresponding contact map files are also required. These are stored as follows:

- Amino acid vector representations in NPZ format: `test_sample_model/data/sample_embeddings.npz`
- Contact map files in NPZ format: `test_sample_model/data/sample_cmap/`

For example, consider the testing sample pair `1a1m_A 1a1m_C 1`:

- The amino acid vector representation of the receptor `1a1m_A`:
  ```plaintext
  [[ 0.15398422 -0.23978221 -0.01549047 ... -0.18346733 -0.09291625 -0.01021743]
   [ 0.4489373  -0.02665013 -0.17931005 ... -0.1891016   0.1553362   0.321806  ]
   [-0.02809518  0.05845888 -0.495727   ... -0.42856818  0.06753138 -0.07047243]
   ...
   [ 0.3649687   0.0686414   0.17520183 ...  0.0492078   0.12580633  0.12104917]
   [ 0.05224043 -0.10220779  0.27868733 ... -0.07296755  0.20231174  0.35552794]
   [-0.2167282  -0.02923655  0.09628326 ... -0.01234592 -0.02105464 -0.09572704]]
  ```
  Shape: `(278, 1024)`

- The contact map file of the receptor `1a1m_A`:
  ```plaintext
  [[0 1 1 ... 0 0 0]
   [1 0 1 ... 0 0 0]
   [1 1 0 ... 0 0 0]
   ...
   [0 0 0 ... 0 1 1]
   [0 0 0 ... 1 0 1]
   [0 0 0 ... 1 1 0]]
  ```
  Shape: `(278, 278)`

- The amino acid vector representation of the peptide `1a1m_C`:
  ```plaintext
  [[ 0.18085133 -0.07571788 -0.19935569 ... -0.01112768 -0.06433804  0.22170882]
   [ 0.16378656  0.1083569  -0.11270649 ...  0.14606047 -0.1882304  -0.00713845]
   [ 0.06448993 -0.02783195 -0.03839097 ...  0.12223947 -0.06157787  0.08955193]
   ...
   [ 0.2744665  -0.06482965 -0.09515045 ...  0.03673784 -0.08219997 -0.13130511]
   [ 0.26488724  0.00398306 -0.19564752 ... -0.02756201 -0.05639829  0.10974102]
   [ 0.06472415  0.0295378   0.04856678 ...  0.09662343 -0.1039608  -0.12716724]]
  ```
  Shape: `(9, 1024)`

- The contact map file of the peptide `1a1m_C`:
  ```plaintext
  [[0 1 1 0 0 0 0 0 0]
   [1 0 1 1 0 0 0 0 0]
   [1 1 0 1 1 0 0 0 0]
   [0 1 1 0 1 1 0 0 0]
   [0 0 1 1 0 1 1 0 0]
   [0 0 0 1 1 0 1 1 0]
   [0 0 0 0 1 1 0 1 1]
   [0 0 0 0 0 1 1 0 1]
   [0 0 0 0 0 0 1 1 0]]
  ```
  Shape: `(9, 9)`

##### Output of the Test Sample Run

The prediction result of the test sample pair is saved in an XLS file with the following format:

- `序号`: Ordinal number representing the sample pair.
- `receptor`: Receptor ID.
- `peptide`: Peptide ID.
- `label`: Classification of sample pairs (1 for interactive, 0 for non-interactive).
- `predict_score`: Probability of interaction between sample pairs. A threshold of 0.5 is used; scores >= 0.5 indicate interaction.
- `predict_label`: Predicted interaction (1 for interaction, 0 for no interaction).

Example:

| 序号 | receptor | peptide | label | predict_score | predict_label |
|----|-----------|---------|------|---------------|---------------|
| 1  | 1a1m_A    | 1a1m_C  | 1    | 0.85          | 1             |

#### Training Your Own Data

After generating the required files, you can perform 5-fold cross-validation by executing the following command:

```sh
cd model
python my_main.py
```

---

