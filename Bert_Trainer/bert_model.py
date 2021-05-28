import torch
import torch.nn as nn
import os
from utilis.Enums import Parameters
from transformers import BertForTokenClassification, AdamW

class BERT(nn.Module):

    '''
    Create an object of BERT model &  AdamW optimizer. 

    Parameters
    ----------------------
    tag2idx : dict containing the tags/entites & its numeric representation
    
    full_finetuning : Boolean 
    if True, fine tune weights of entire BERT model. 
    else, fine tune only the classifier module of BERT 

    '''

    def __init__(self, tag2idx, full_finetuning=True):
        super(BERT, self).__init__()

        if os.path.exists(os.path.join(Parameters.MODEL_DIR,"final_model.pt")) and not Parameters.DOWNLOAD_WEIGHTS:
          print("Loading Checkpointed Generic NER model")
          self.model = BertForTokenClassification.from_pretrained(
              Parameters.MODEL_DIR,
              num_labels = len(tag2idx),
              output_attentions = False,
              output_hidden_states = False
            )


        else:
          print("Using Downloaded BERT-pretrained weights")
          self.model = BertForTokenClassification.from_pretrained(
              'bert-base-cased',
              num_labels=len(tag2idx),
              output_attentions = False,
              output_hidden_states = False
          )

        self.optimizer = self.generate_optimizer(full_finetuning)
        
    
    def generate_optimizer(self,full_finetuning):
        if full_finetuning:
            param_optimizer = list(self.model.named_parameters())
            no_decay = ['bias', 'gamma', 'beta']
            optimizer_grouped_parameters = [
                {'params': [p for n, p in param_optimizer if not any(nd in n for nd in no_decay)],
                    'weight_decay_rate': 0.01},
                {'params': [p for n, p in param_optimizer if any(nd in n for nd in no_decay)],
                    'weight_decay_rate': 0.0}
                ]

        else:
            param_optimizer = list(self.model.classifier.named_parameters())
            optimizer_grouped_parameters = [{"params": [p for n, p in param_optimizer]}]

        return AdamW(optimizer_grouped_parameters, lr=3e-5, eps=1e-8)


    def save_checkpoint(save_path, model, valid_loss):
        if save_path == None:
          return

        state_dict = {'model_state_dict':self.model.state_dict(),
                      'valid_loss':valid_loss}

        torch.save_checkpoint(state_dict, save_path)
        print('Model saved to'+' '+str(save_path))






