import enum
import torch
import torch.nn as nn
import torchvision
import torch.optim as optim
from collections import OrderedDict
from data_utils import DataLoader
from model import CatOrDog
import os

# def calc_error(predictions, labels):
#     err_count = 0
#     for i in range(len(predictions)):
#         if predictions[i] != labels[i]:
#             err_count += 1
#     return err_count

def calc_error(predictions, labels):
    err_count = 0
    preds = []
    for pred in predictions:
        if pred >= 0.5:
            preds.append(1)
        else:
            preds.append(0)


    for i in range(len(preds)):
        if int(preds[i]) != int(labels[i]):
            err_count += 1

    return err_count

def validate(model, validation_data):
    val_err = 0
    val_loss = 0
    num_labels = 0
    loss_function = nn.BCEWithLogitsLoss()
    for batch, data in enumerate(validation_data):
        images, labels = data
        outputs = model(images)
        predictions = outputs.squeeze()
        loss = loss_function(outputs, labels.unsqueeze(1).float())

        val_err += calc_error(predictions, labels)
        val_loss += loss.item()
        num_labels += len(labels)

    return 1 - val_err/num_labels, val_loss/len(validation_data)

def train(model, training_data, validation_data, batch_size=256, epochs=12, learning_rate=0.0005):
    loss_function = nn.BCEWithLogitsLoss()
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)
    optimizer.zero_grad()
    training_err = [0.0 for i in range(epochs)]
    training_acc = [0.0 for i in range(epochs)]
    training_loss = [0.0 for i in range(epochs)]
    val_acc = [0.0 for i in range(epochs)]
    val_loss = [0.0 for i in range(epochs)]


    for epoch in range(epochs):
        accumulated_training_err = 0
        accumulated_training_loss = 0
        num_labels = 0
        for batch, data in enumerate(training_data):
            if batch % 10 == 0:  
                print("In batch: ", batch)
            # if batch > 5:
            #     break
            images, labels = data
            outputs = model(images)
            predictions = outputs.squeeze()
            loss = loss_function(outputs, labels.unsqueeze(1).float())
            loss.backward()
            optimizer.step()
            optimizer.zero_grad()
            accumulated_training_err += calc_error(predictions=predictions, labels=labels)
            accumulated_training_loss += loss.item()
            num_labels += len(labels)
            if batch % 10 == 0 and epoch > 0:
                torch.save(model.state_dict(), f"model_training_data_batch_{batch}_epoch{epoch}_batch_size_{batch_size}_lr_{learning_rate}_overnight")


        
        training_err[epoch] = accumulated_training_err / num_labels
        training_acc[epoch] = 1 - training_err[epoch]
        training_loss[epoch] = accumulated_training_loss / len(training_data)
        val_acc[epoch], val_loss[epoch] = validate(model, validation_data)
        torch.save(model.state_dict(), f"model_training_data_epoch_{epoch}_batch_size_{batch_size}_lr_{learning_rate}_overnight")
        print(f"Finished epoch {epoch}. ")
        print(f"Training accuracy: {training_acc[epoch]} | Training loss: {training_loss[epoch]} | Validation accuracy: {val_acc[epoch]} | Validation loss: {val_loss[epoch]}.")
    
        
if __name__ == "__main__":
    training_loader = DataLoader(batch_size=256)
    val_loader = DataLoader(batch_size=256, path=os.path.join(os.getcwd(), "validation"))
    training_loader.load()
    val_loader.load()
    model = CatOrDog()
    train(model, training_loader.data, val_loader.data)
