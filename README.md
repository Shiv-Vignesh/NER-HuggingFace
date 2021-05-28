# HuggingFace NER Trainer 
Generic NER model trained on HuggingFace transformers library to predict `Address, Amount, Currency, Date, OrganizationName, PersonName` entities. 

## set up envirnoment 
---bash
pip install -r requirements.txt 

## prepare data for training

- Create a directory called data to place our training data. 
---bash 
mkdir data

- The training data is a .txt file with the following format : 
    - token_text token_tag
    - token_text token_tag
    - \n # each new line indicates new data sample
    - token_text token_tag 

- Check utils/enums.py for test split, batch size & other information. 

## Run the application
--bash 
python3 main.py

## Model 

- Model : BERT-base (refer bert_model.py)
    - full_finetuning is parameter which decides whether to re-train the entire model or the classifier module of BERT. 
    - Optimizer : AdamW (weight decay)

- Loading model weights 

    - By default, model weights are downloaded from HuggingFace library. 
    - Checkpointed model can be loaded from 'saved model' directory. 

    - Refer to utils/enums.py to change the download option. 

## Training reports & visualization. 

- The training logs are saved in 'report' directory, in train_logs.txt
    - Format {epoch} {loss} {accuracy} {F1 score} {precision} {Recall}

- Loss, Accuracy & F1 score graph are saved in 'graph plots' directory.