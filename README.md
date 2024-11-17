# AYA-Expanse LLM Finetuning
An implementation of fine-tuning AYA-Expanse model on English to German and Persian Translation

***For fine-tuning and infering from the model, you need to download AYA-Expanse model from https://huggingface.co/blog/ariG23498/cohere-aya-expanse***

## How to run
- Download the base model from the provided link
- Create a Python3 virtual env
- Activate the environment creatred in the previous step
- install the requirements in requirements.txt file
```
pip install -r requirements.txt
```
- For finetuning run the following:
```
python run_finetuning.py
```
- For running demo app run the following:
```
python run_app.py
```