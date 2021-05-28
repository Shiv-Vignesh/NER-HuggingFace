from utilis.Enums import Parameters
import os
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from seqeval.metrics import f1_score, accuracy_score, classification_report, precision_score, recall_score
import matplotlib.pyplot as plt
import pandas as pd

def identify_predicted_tags(predictions, true_labels, tag_values):
    '''
    identify the tags predicted by the model. 

    Parameters
    ---------------------
    predictions : List of predictions by the model. 
    true_labels : List of ground truth values.
    tag_values : dict to map numeric value to category 
    '''
    predicted_tags = []
    for p, l in zip(predictions, true_labels):
        temp = []

        for p_i,l_i in zip(p, l):
            if tag_values[l_i] != 'PAD':
              temp.append(tag_values[p_i])

        predicted_tags.append(temp)

    return predicted_tags


def identify_true_tags(true_labels, tag_values):

    '''
    identify true tags

    Parameters
    -------------------
    true_labels : List of numeric values. 
    tag_values : dict to map numeric value to its category

    '''

    actual_tags = []
    for l in true_labels:
        temp = []

        for l_i in l:
            if tag_values[l_i] != 'PAD':
              temp.append(tag_values[l_i])

        actual_tags.append(temp)

    return actual_tags

def save_report(iteration, report_type:str, predicted, ground_truth, loss):
  if not os.path.isdir(Parameters.TRAIN_REPORT_DIR):
    os.makedirs(Parameters.TRAIN_REPORT_DIR)

  if not os.path.isdir(Parameters.VALID_REPORT_DIR):
    os.makedirs(Parameters.VALID_REPORT_DIR)

  if report_type == "training":
    with open(Parameters.TRAIN_REPORT_DIR+"/train_logs.txt","a+") as f:
      acc = accuracy_score(predicted, ground_truth)
      f1 = f1_score(predicted, ground_truth)
      precision = precision_score(predicted, ground_truth)
      recall = recall_score(predicted, ground_truth)

      f.write(f"EPOCH_{iteration} {loss} {acc} {f1} {precision} {recall} \n")

    f.close()

  if report_type == "validation":
    with open(Parameters.VALID_REPORT_DIR+"/validation_logs.txt","a+") as f:
      acc = accuracy_score(predicted, ground_truth)
      f1 = f1_score(predicted, ground_truth)
      precision = precision_score(predicted, ground_truth)
      recall = recall_score(predicted, ground_truth)

      f.write(f"EPOCH_{iteration} {loss} {acc} {f1} {precision} {recall}")

    f.close()

def train(model, optimizer, device, train_dataloader, valid_dataloader, tag_values):
    '''
    method to train the BERT model on custom dataset, plot the loss & accuracy graphs & save model instance. 

    Parameters
    ------------------------------
    model : Instance of BERT or any other HuggingFace TokenClassification Model
    optimizer : Instance of optimizer used in training. 
    train_dataloader : Instance of torch Dataloader class for training inputs. 
                        Consists of batch inputs, batch input masks & batch outputs. 

    valid_dataloader : Same as train_dataloader but for validation/test dataset

    tag_values : dict to map numeric value to its category
    '''

    loss_values, validation_loss_values = [], []
    tr_f1, val_f1 = [],[]
    train_acc, val_acc = [], []

    total_steps = len(train_dataloader) * Parameters.EPOCHS

    model = model.to(device)

    for iteration in range(0, Parameters.EPOCHS):
        model.train()

        total_loss = 0

        train_predictions, train_true_labels = [], []

        for step, batch in enumerate(train_dataloader):
            batch = tuple(t.to(device) for t in batch)
            batch_input_ids, batch_input_mask, batch_labels = batch

            model.zero_grad()

            outputs = model(batch_input_ids, token_type_ids=None, attention_mask= batch_input_mask, labels= batch_labels)
            loss = outputs[0]

            if(Parameters.USE_L1):
                l1_criterion = nn.L1Loss(size_average=False)
                reg_loss = 0

                for param in model.parameters():
                  reg_loss += l1_criterion(param, target=torch.zeros_like(param))

                factor = 0.00005
                loss += factor * reg_loss

            train_logits = outputs[1].detach().cpu().numpy()
            train_ids = batch_labels.to('cpu').numpy()

            train_predictions.extend([list(p) for p in np.argmax(train_logits,axis=2)])
            train_true_labels.extend(train_ids)

            loss.backward()

            total_loss += loss.item()

            torch.nn.utils.clip_grad_norm_(parameters=model.parameters(), max_norm=Parameters.MAX_GRAD_NORM)

            optimizer.step()

        avg_train_loss = total_loss/len(train_dataloader)
        print("Average train loss: {}".format(avg_train_loss))
        loss_values.append(avg_train_loss)

        train_pred_tags = identify_predicted_tags(train_predictions, train_true_labels, tag_values)

        train_actual_tags = identify_true_tags(train_true_labels, tag_values)


        print("Training Classification Report: {}".format(classification_report(list(train_pred_tags), list(train_actual_tags),digits=5)))
        print()

        tr_f1.append(f1_score(train_pred_tags,train_actual_tags))
        train_acc.append(accuracy_score(train_pred_tags, train_actual_tags))

        save_report(iteration,
                    "training",
                    train_pred_tags,
                    train_actual_tags,
                    avg_train_loss
        )

        model.eval()

        eval_loss, eval_accuracy = 0, 0
        predictions, true_labels = [],[]

        for batch in valid_dataloader:
            batch = tuple(t.to(device) for t in batch)

            b_input_ids, b_input_mask, b_labels = batch

            with torch.no_grad():
                outputs = model(b_input_ids, token_type_ids=None, attention_mask = b_input_mask, labels = b_labels)
    
            logits = outputs[1].detach().cpu().numpy()
            label_ids = b_labels.to('cpu').numpy()

            eval_loss += outputs[0].mean().item()

            predictions.extend([list(p) for p in np.argmax(logits, axis=2)])
            true_labels.extend(label_ids)

        eval_loss = eval_loss/len(valid_dataloader)
        validation_loss_values.append(eval_loss)

        print("Validation loss: {}".format(eval_loss))

        pred_tags = identify_predicted_tags(predictions, true_labels, tag_values)

        valid_tags = identify_true_tags(true_labels, tag_values)



        print("----------------------------------VALIDATION METRICs-------------------------------------------------------")
        print("Validation Accuracy: {}".format(accuracy_score(pred_tags, valid_tags)))
  
        print("Validation Classification Report: {}".format(classification_report(list(pred_tags), list(valid_tags))))
        print()

        val_f1.append(f1_score(pred_tags, valid_tags))

        val_acc.append(accuracy_score(pred_tags, valid_tags))

        save_report(iteration,
                    "validation",
                    pred_tags,
                    valid_tags,
                    eval_loss
        )

    if not os.path.isdir(Parameters.PLOTS_DIR):
        os.makedirs(Parameters.PLOTS_DIR)

    if not os.path.isdir(Parameters.MODEL_DIR):
        os.makedirs(Parameters.MODEL_DIR)
  

    # torch.save(model.state_dict(),os.path.join(Parameters.MODEL_DIR,"final_model.pt"))
    model.save_pretrained(Parameters.MODEL_DIR)
    
    print("Training Loss Graph")
    plt.plot(loss_values, label="training loss")

    print("Validation Loss Graph")
    plt.plot(validation_loss_values, label="validation loss")
    plt.legend(loc="upper left")
    plt.savefig(os.path.join(Parameters.PLOTS_DIR,"loss_curve.png"))
    plt.show()

    print("Training Accuracy")
    plt.plot(train_acc, label="training accuracy")

    print("Validation Accuracy")
    plt.plot(val_acc, label="validation accuracy")
    plt.legend(loc="upper left")
    plt.savefig(os.path.join(Parameters.PLOTS_DIR,"accuracy_curve.png"))
    plt.show()

    print("Training F1 Score")
    plt.plot(tr_f1, label="training F1")

    print("Validation F1 Score")
    plt.plot(val_f1, label="validation F1")
    plt.legend(loc="upper left")
    plt.savefig(os.path.join(Parameters.PLOTS_DIR,"F1_curve.png"))
    plt.show()  
