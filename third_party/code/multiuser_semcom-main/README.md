# Multi-user SemCom

## Environment

If you're using Docker you have to take care of the related speed degredation problem ([ref1](https://stackoverflow.com/questions/50464643/performance-issues-with-machine-learning-using-docker-and-flask), [ref2](https://medium.com/free-code-camp/how-a-badly-configured-tensorflow-in-docker-can-be-10x-slower-than-expected-3ac89f33d625)), so I'm using conda for now.

The code is run and tested with Python 3.10.14 and CUDA 11.8. 

The below is the record of command
```bash
# install miniconda
wget https://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-x86_64.sh
bash Miniconda3-latest-Linux-x86_64.sh

# create environment
# or just `conda create --name semcom_cuda118 --file spec-list.txt` if that's the newest version
conda create --name semcom_cuda118 python=3.10
conda activate semcom_cuda118

conda install pytorch torchvision torchaudio pytorch-cuda=11.8 -c pytorch -c nvidia
pip install transformers==4.26.1

# make sure cuda is actually available and useable
python
>>> import torch
>>> torch.cuda.is_available()
True
>>> torch.Tensor([1]).to('cuda') + 1    # try to load and do something
tensor([2.], device='cuda:0')
>>> exit()
```

The conda environment can also be created by `spec-file.txt` provided
```bash
bash Miniconda3-latest-Linux-x86_64.sh
conda create --name semcom_cuda118 --file spec-file.txt
conda activate semcom_cuda118

# Install transformers by `pip`
pip install transformers==4.26.1
```


To replicate the environment to another device:
```bash
conda activate semcom_cuda118
conda list --explicit > spec-file.txt

# send spec-list.txt and Miniconda3 bash file to another device
# on another device:
# assume miniconda installed
conda create --name semcom_cuda118 --file spec-file.txt
conda activate semcom_cuda118
gh auth login
```

(或是在舊環境 `conda env export environment.yml`，之後 `conda env create -f environment.yml`)

## Dataset & Task
### MOSEI and MOSI
For multimodal sentiment analysis (MSA).
Download the CMU-MOSI and CMU-MOSEI dataset from [Google Drive](https://drive.google.com/drive/folders/1IBwWNH0XjPnZWaAlP1U2tIJH6Rb3noMI?usp=sharing) and place the contents inside `./data/msadata` folder. Note that these are (pre-computed splits).

### AVE dataset
For audio-visual event (AVE). AVE dataset can be downloaded from [Google Drive](https://drive.google.com/open?id=1FjKwe79e0u96vdjIVwfRQ1V6SoDHe7kK) (zip file) and place the unzipped contents inside `./data/avedata` folder.

[Audio feature](https://drive.google.com/file/d/1F6p4BAOY-i0fDXUOhG7xHuw_fnO5exBS/view?usp=sharing) and [visual feature](https://drive.google.com/file/d/1hQwbhutA3fQturduRnHMyfRqdrRHgmC9/view?usp=sharing) (7.7GB) are also needed to be downloaded. Please put features into `./data/avedata` folder before running the code. 

The videos of AVE dataset are put in `./data/avedata/AVE_Dataset/AVE` folder.

## Run
Please treat src as module.

- Training
```bash
python -m src.train.train_UDeepSC --config /PATH/TO/CONFIG
# e.g. python -m src.train.train_UDeepSC --config ./src/train_config/train_config_msa.json
```

- Shapley value
```bash
python -m src.test.shap.shap_main -t TASK --ckpt_path /PATH/TO/CHECKPOINT --save_dir /PATH/TO/SAVE_DIR
# e.g.  python -m src.test.shap.shap_main -t ave --ckpt_path ./checkpoint/20251106/awgn_12_udeepsc_ave_symbols_24_ave_1 --save_dir ./tmp/20251105/
```

- Experiment
```bash
python -m src.test.experiment.regression --config /PATH/TO/CONFIG
# e.g. python -m src.test.experiment.regression --config ./src/eval_config/udeepsc_msa.json
```