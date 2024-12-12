# MutDPAL:Prediction of Pathogenic Mutations in Human Membrane Proteins and Their Associated Diseases via Utilizing Pre-trained Bio-LLMs
## Introduction  
MutDPAL is a novel deep learning approach to predict the pathogenicity of missense mutations in membrane proteins and determine the disease categories of the pathogenic mutation.
![MutDPAL](./architecture.png)
## Requirements

* python = 3.8.19   
* cuda = 11.8    
* torch = 2.3.0  
* numpy = 1.24.3  
* scikit-learn = 1.3.2  
* tqdm = 4.66.4  

In case you want to use conda for your own installation please create a new MutDPAL environment.
We showed an example of creating an environment.
```sh
conda create -n MutDPAL python=3.8.19
conda activate MutDPAL
conda install pytorch==2.3.0 pytorch-cuda=11.8 -c pytorch -c nvidia
conda install scikit-learn=1.3.2 
conda install  numpy=1.24.3
```
Or you can use the provided [environment.yml](./environment.yml) to create all the required dependency packages.
```sh
conda env create -f environment.yml
```

It is also necessary to install two pre-trained models: [ESM-1v](https://huggingface.co/facebook/esm1v_t33_650M_UR90S_1), [BioBert](https://huggingface.co/dmis-lab/biobert-base-cased-v1.1/). We use the pre-trained weights from HuggingFace for prediction. and [TMbed](https://github.com/BernhoferM/TMbed). Please download them to your device and modify the corresponding paths to extraction features.  

## Quick start
1. Feature extraction using `protein_feature.py` and `BioBert` 
2. cd scripts
3. For pathogenic classification task  
```python
cd patho_classification/  
python main.py --mode train
# or
python main.py --mode test
```
4. For multi-label disease classification task  
 ```python
cd dis_classification/  
python main.py --mode train
# or
python main.py --mode test
```
> We provide examples in `./example`.

First, you must provide a csv file, containing the UniprotID, seq, ref, pos, mut, mut_seq, y_class, y, etc. of the protein.  You can use `protein_feature.py` to extract the semantic information and physicochemical and biochemical properties of the protein.  

Then, based on the sequence, use TMbed to obtain the protein's topological structure information and represent it in natural language format, as shown in the `transmembrane_description.txt` file.   

Finally, leverage BioBERT to generate the corresponding transmembrane environment features.   

> Then run:
```bash
cd scripts/
python main.py --mode test
```
And the disease classification prediction results will be saved in `../model_state/test_logs.txt`. 


## Contacts  
Any more questions, please do not hesitate to contact me: 20234227054@stu.suda.edu.cn
