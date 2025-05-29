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
conda install  tqdm=4.66.4  
```
Or you can use the provided [environment.yml](./environment.yml) to create all the required dependency packages.
```sh
conda env create -f environment.yml
```

It is also necessary to install two pre-trained models: [ESM-1v](https://huggingface.co/facebook/esm1v_t33_650M_UR90S_1), [BioBert](https://huggingface.co/dmis-lab/biobert-base-cased-v1.1/). We use the pre-trained weights from HuggingFace for prediction. and [TMbed](https://github.com/BernhoferM/TMbed). Please download them to your device and modify the corresponding paths to extraction features.  

## Quick start
1. Download  'data' file and add it to the current path:

   url： https://drive.google.com/drive/folders/1lTz0hUA8VZ_1h12rx3653XD_QZvnwQpv?usp=sharing

   
2. cd scripts
3. For pathogenic classification task  
```python
cd patho_classification/  
python main.py --mode train  # Model retraining
# or
python main.py --mode test  # Load the saved weights to evaluate the model on test data
```
For Pred-MutHTP dataset  
```python
python pred-muthtp.py --mode train  # Model retraining  
# or
python pred-muthtp.py --mode test  # Load the saved weights to evaluate the model on holdout-test data
```  
4. For multi-label disease classification task  
 ```python
cd dis_classification/  
python main.py --mode train
# or
python main.py --mode test
```


## Contacts  
Any more questions, please do not hesitate to contact me: 20234227054@stu.suda.edu.cn
