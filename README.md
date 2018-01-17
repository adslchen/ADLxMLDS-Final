Deep Complex Network for Voice Separation
=====================
This repository is the final project for b02504036, r06943014, b02901031,
r06943020. 

Requirements
------------
- Python >= 3.5
- Pytorch == 0.2.0

Install requirements for music separation

```
pip install librosa, scipy
```

Download data sets for training.
```
python download.py
```
Unzip the download file
```
unzip mir-1k.zip
```

Training
------------

```
python train.py --local-data=[/path/to/mir-1k/train_data]
--model_dir=[/path/to/your/model/saving/directory]
```


Testing 
------------
```
python test.py --test_dir=[/path/to/mir-1k/test_data]
--output=[/path/to/output/saving/dir] --model_path=models/complex.pth
```


