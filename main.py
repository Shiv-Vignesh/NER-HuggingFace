import pandas as pd
import torch
from torch.utils.data import TensorDataset, DataLoader, RandomSampler, SequentialSampler
import transformers

from utilis.sentence_getter import SentenceGetter
from utilis.bert_tokenizer import BertTokenizerClass
from utilis.Enums import Parameters

from Bert_Trainer.bert_model import BERT 
from Bert_Trainer.trainer import train

from sklearn.model_selection import train_test_split


def check_device():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    return device

def load_data(path):
    '''
    Load the dataset from the specified PATH & create tag_values & tag2idx.
    tag_values: list of unique entities present in the dataset
    tag2idx : Dict representing numerical value for each entity
    '''

    data = pd.read_csv(path, delimiter=" ", header= None)
    data.columns = ["Doc_name","block_no","token_no","token","category","xmin","ymin","xmax","ymax"]

    getter = SentenceGetter(data)

    sentences = [[word[0] for word in sentence] for sentence in getter.sentences]
    labels = [[s[1] for s in sentence] for sentence in getter.sentences]

    tag_values = list(set(data['category'].values))
    tag_values.append("PAD")
    tag2idx = {t: i for i,t in enumerate(tag_values)}

    return sentences, labels, tag2idx, tag_values

def tokenize_data(sentences, labels, tag2idx):
    '''
    Tokenize the sentences & labels
    '''

    bert_tokenizer = BertTokenizerClass(sentences, labels,tag2idx)
    return bert_tokenizer.input_ids, bert_tokenizer.tags

def prepare_dataloaders(input_ids, tags):
    '''
    Prepare train & validation dataloaders for training in torch supported format
    '''

    attention_masks = [[float(id!=0.0) for id in ids] for ids in input_ids]

    training_inputs, validation_inputs, training_tags, validation_tags = train_test_split(input_ids,
                                                                                            tags,
                                                                                            test_size=Parameters.TEST_SPLIT)


    training_masks, validation_masks,_,_ = train_test_split(attention_masks, 
                                                            input_ids,
                                                            test_size=Parameters.TEST_SPLIT)

    training_inputs = torch.tensor(training_inputs, dtype=torch.long)
    training_tags = torch.tensor(training_tags, dtype=torch.long)
    
    validation_inputs = torch.tensor(validation_inputs, dtype=torch.long)
    validation_tags = torch.tensor(validation_tags, dtype=torch.long)

    training_masks = torch.tensor(training_masks, dtype=torch.long)
    validation_masks = torch.tensor(validation_masks, dtype=torch.long)

    train_data = TensorDataset(training_inputs, training_masks,training_tags)
    train_sampler = RandomSampler(train_data)
    train_dataloader = DataLoader(train_data, sampler=train_sampler, batch_size = Parameters.bs)

    valid_data = TensorDataset(validation_inputs, validation_masks, validation_tags)
    valid_sampler = RandomSampler(valid_data)
    valid_dataloader = DataLoader(valid_data, sampler= valid_sampler, batch_size= Parameters.bs)

    return train_dataloader, valid_dataloader

def generate_model(tag2idx,full_finetuning):
    '''
    Create instance of BERT model & optimizer
    '''

    bert = BERT(tag2idx,full_finetuning)

    return bert.model, bert.optimizer




if __name__ == "__main__":
    path = "data/dataset.txt"

    device = check_device()

    sentences , labels, tag2idx, tag_values = load_data(path=path)
    input_ids, tags = tokenize_data(sentences, labels, tag2idx)
    train_loader, valid_loader = prepare_dataloaders(input_ids, tags)

    print(train_loader)

    model,optimizer = generate_model(tag2idx, Parameters.FULL_FINETUNING)

    train(model,optimizer,device,train_loader, valid_loader, tag_values)