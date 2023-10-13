# Preparation data and env
* Make sure your directory looks like:
```
JSDRV/
```
RumorEval-S, SemEval-8 contains raw data consists of claim, related posts, label

* Make Sure your env consitent to "cuda.yaml"

# Run code
Directly run training and eval  
* make sure you have dataset under `data/`.  
* download `roberta_base` and put this folder under the `JSDRV/` root directory

* run the `train.py` file using `JSDRV/` as the working directory:
  * `python train.py --outdir . --config_file R.ini`, or
  * `python train.py --outdir . --config_file S.ini`


