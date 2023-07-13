import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import optuna
import torchvision.datasets as datasets
from torch.utils.data import DataLoader
from torchvision import datasets, models, transforms
from efficientnet_pytorch import EfficientNet
from collections import Counter
from sklearn.utils.class_weight import compute_class_weight
from torch.utils.data import DataLoader, TensorDataset
from torch.nn import functional as F
from torchvision import transforms
from torchvision.transforms.functional import resize, to_tensor
from sklearn.metrics import f1_score

#custom functions
from Model_functions import *


# Define the device to be used for training
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

HGG_train = load_from_dir('/home/viktoriia.trokhova/Stacked_4/train/HGG_stack')
LGG_train = load_from_dir('/home/viktoriia.trokhova/Stacked_4/train/LGG_stack')
HGG_val = load_from_dir('/home/viktoriia.trokhova/Stacked_4/val/HGG_stack')
LGG_val = load_from_dir('/home/viktoriia.trokhova/LGG_stack')


X_train = np.array(HGG_train + LGG_train)
X_val = np.array(HGG_val + LGG_val)

# Creating the labels array
y_train = np.array([0] * len(HGG_train) + [1] * len(LGG_train))
y_val = np.array([0] * len(HGG_val) + [1] * len(LGG_val))

class_counts = Counter(y_train)

class_weights = compute_class_weight(class_weight='balanced', classes=np.unique(y_train), y=y_train)
print(class_weights)

class_weights_np = np.array(class_weights, dtype=np.float32)
class_weights_tensor = torch.from_numpy(class_weights_np)
class_weights_tensor = torch.from_numpy(class_weights_np)
if torch.cuda.is_available():
    class_weights_tensor = class_weights_tensor.to(device)

# Convert labels to categorical tensor
y_train_tensor = torch.tensor(y_train, dtype=torch.long)
y_val_tensor = torch.tensor(y_val, dtype=torch.long)

# Convert arrays to tensors
X_train_tensor = torch.tensor(X_train, dtype=torch.float32)
X_val_tensor = torch.tensor(X_val, dtype=torch.float32)

# Convert labels to one-hot encoded format
num_classes = len(np.unique(y_train))
y_train_categorical = torch.nn.functional.one_hot(y_train_tensor, num_classes=num_classes)
y_val_categorical = torch.nn.functional.one_hot(y_val_tensor, num_classes=num_classes)

# Create train and validation datasets
train_dataset = TensorDataset(X_train_tensor, y_train_categorical)
val_dataset = TensorDataset(X_val_tensor, y_val_categorical)

    
class Effnet(nn.Module):
    def __init__(self, pretrained=True, dense_0_units=None, dense_1_units=None, dropout=None):
        super().__init__()

        # Load the pretrained EfficientNet-B1 model
        efficientnet_b1 = EfficientNet.from_pretrained('efficientnet-b1')

        # Replace the first convolutional layer to handle images with shape (240, 240, 4)
        efficientnet_b1._conv_stem = nn.Conv2d(4, 32, kernel_size=3, stride=2, bias=False)
        
        # Reuse the other layers from the pretrained EfficientNet-B1 model
        self.features = efficientnet_b1.extract_features
        self.avgpool = nn.AdaptiveAvgPool2d(output_size=(1, 1))
        self.dropout = nn.Dropout(dropout)
        if dense_0_units is not None:
            dense_0_units = int(dense_0_units)
            self.fc1 = nn.Linear(in_features=1280, out_features=dense_0_units, bias=True)
        else:
            self.fc1 = None
        if dense_1_units is not None:
            dense_1_units = int(dense_1_units)
            self.fc2 = nn.Linear(in_features=dense_0_units, out_features=dense_1_units, bias=True)
            self.fc3 = nn.Linear(dense_1_units, 2)
        else:
            self.fc2 = None
            self.fc3 = nn.Linear(dense_0_units, 2)
        
        
    def forward(self, x):
        x = self.features(x)
        x = self.avgpool(x)
        x = self.dropout(x)
        x = x.view(x.size(0), -1)
        x = F.relu(self.fc1(x))
        if self.fc2 is not None:
            x = F.relu(self.fc2(x))
            x = self.fc3(x)
        else:
            x = self.fc3(x)   
        return x
        

    

class FocalLoss(nn.Module):
    def __init__(self, weight=None, gamma=2.0, alpha=0.25, reduction='mean'):
        super(FocalLoss, self).__init__()
        self.weight = weight
        self.gamma = gamma
        self.alpha = alpha
        self.reduction = reduction

    def forward(self, input, target):
        ce_loss = F.cross_entropy(input, target, reduction=self.reduction, weight=self.weight)
        pt = torch.exp(-ce_loss)
        focal_loss = (self.alpha * (1-pt)**self.gamma * ce_loss).mean()
        return focal_loss


def train_and_evaluate(param, model, trial):
    f1_scores = []
    accuracies = []
    EPOCHS = 5
    
    criterion = FocalLoss(weight=class_weights_tensor, gamma=2.0, alpha=0.25)
    optimizer = getattr(optim, param['optimizer'])(model.parameters(), lr=param['learning_rate'])
    train_loader = DataLoader(train_dataset, batch_size=param['batch_size'], shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=param['batch_size'], shuffle=False)

    for epoch_num in range(EPOCHS):
        torch.cuda.empty_cache()
        model.train()
        total_acc_train = 0
        total_loss_train = 0
        train_correct = 0
        train_loss = 0

        for batch_idx, (data, target) in enumerate(train_loader):
            data, target = data.permute(0, 3, 1, 2), target.float() # Permute dimensions
            #print(target.float)
            optimizer.zero_grad()
            output = model(data)
            loss = criterion(output, target)
            #print('loss:', loss)
            train_loss += loss.item()
            #print('output:', output)

            predictions = torch.argmax(output, dim=1).detach().cpu().numpy()
            #print('predictions:', predictions)

            target_numpy = target.detach().cpu().numpy()
            correct_predictions = np.sum(predictions == target_numpy.argmax(axis=1))
           
            #print('correct_predictions:', correct_predictions)

            batch_accuracy = correct_predictions / target_numpy.shape[0]
            #print("Number of correct predictions:", correct_predictions)
            #print("Accuracy of the batch:", batch_accuracy)
            train_correct += batch_accuracy
            loss.backward()
            optimizer.step()

        # Calculate epoch-level loss and accuracy
        epoch_loss = train_loss / len(train_loader)
        epoch_accuracy = train_correct / len(train_loader)

        print("Epoch Loss:", epoch_num, ': ', epoch_loss)
        print("Epoch Accuracy:", epoch_num, ': ', epoch_accuracy)
            
       
        model.eval()
        val_loss = 0
        val_correct = 0
        val_f1_score = 0  # Initialize val_f1_score
        val_labels = []
        y_preds = []
        
        with torch.no_grad():
            for data, target in val_loader:
                data, target = data.permute(0, 3, 1, 2), target.float() # Permute dimensions
                #data = data.float()
                output = model(data)
                val_loss += criterion(output, target)
                softmax = nn.Softmax(dim=1)
                output = softmax(output)
                predictions = torch.argmax(output, dim=1).detach().cpu().numpy()
                #print('predictions:', predictions)

                target_numpy = target.detach().cpu().numpy()
                correct_predictions = np.sum(predictions == target_numpy.argmax(axis=1))

                #print('correct_predictions:', correct_predictions)

                batch_accuracy = correct_predictions / target_numpy.shape[0]
                #print("Number of correct predictions:", correct_predictions)
                #print("Accuracy of the batch:", batch_accuracy)
                val_correct += batch_accuracy
                
                # Calculate F1 score
                f1 = f1_score(target_numpy.argmax(axis=1), predictions, average='macro')
                val_f1_score += f1

            # Calculate epoch-level loss, accuracy, and F1 score
            epoch_val_loss = val_loss / len(val_loader)
            epoch_val_accuracy = val_correct / len(val_loader)
            epoch_val_f1_score = val_f1_score / len(val_loader)
            print('val f1-score:',  epoch_num, ': ', epoch_val_f1_score)
            print('val accuracy:',  epoch_num, ': ', epoch_val_accuracy)
        
        f1_scores.append(epoch_val_f1_score)        
        print(f1_scores)
        trial.report(epoch_val_f1_score, epoch_num)
        if trial.should_prune():
            raise optuna.exceptions.TrialPruned()
    
    final_f1 = max(f1_scores)
    PATH = '/home/viktoriia.trokhova/model_weights/stack_4.pt'
    torch.save(model.state_dict(), PATH)

    return final_f1

#Define a set of hyperparameter values, build the model, train the model, and evaluate the accuracy
def objective(trial):

    params = {
        'learning_rate': trial.suggest_categorical("learning_rate", [0.0001, 0.001, 0.01, 0.1]),
        'optimizer': trial.suggest_categorical("optimizer", ["Adam", "SGD"]),
        'dense_0_units': trial.suggest_categorical("dense_0_units", [16, 32, 48, 64, 80, 96, 112, 128]),
        'dense_1_units': trial.suggest_categorical("dense_1_units", [None, 16, 32, 48, 64, 80, 96, 112, 128]),
        'batch_size': trial.suggest_categorical("batch_size", [16, 32, 64]),
        'drop_out': trial.suggest_float("dropout", 0.2, 0.8, step=0.1)
    }

    model = Effnet(pretrained=True, dense_0_units=params['dense_0_units'],  dense_1_units=params['dense_1_units'], dropout=params['drop_out'])

    max_f1 = train_and_evaluate(params, model, trial)

    return max_f1


EPOCHS = 50
    
study = optuna.create_study(direction="maximize", sampler=optuna.samplers.TPESampler(), pruner=optuna.pruners.HyperbandPruner(min_resource=1, max_resource=6, reduction_factor=5))
study.optimize(objective, n_trials=40)
pruned_trials = study.get_trials(deepcopy=False, states=[TrialState.PRUNED])
complete_trials = study.get_trials(deepcopy=False, states=[TrialState.COMPLETE])

print("Study statistics: ")
print("  Number of finished trials: ", len(study.trials))
print("  Number of pruned trials: ", len(pruned_trials))
print("  Number of complete trials: ", len(complete_trials))

print("Best trial:")
trial = study.best_trial
print("  Value: ", trial.value)

print("  Params: ")
for key, value in trial.params.items():
    print("    {}: {}".format(key, value))




learning_rate_best = 0.001
optimizer_best = 'SGD'
dense_0_units_best = 64
dense_1_units_best = 48
batch_size_best = 64
dropout_best = 0.4
    

model = Effnet(pretrained=False, dense_0_units=dense_0_units_best, dense_1_units=dense_1_units_best, dropout=dropout_best)
                                                             
                                                          

def train_and_evaluate(model, learning_rate_best, optimizer_best, dense_0_units_best, dense_1_units_best, 
                       batch_size_best):    
    EPOCHS = 50
    
    # Create optimizer
    optimizer = getattr(optim, optimizer_best)(model.parameters(), lr=learning_rate_best)
    criterion = FocalLoss(weight=class_weights_tensor, gamma=2.0, alpha=0.25)
    train_loader = DataLoader(train_dataset, batch_size=batch_size_best, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size_best, shuffle=False)

    # For tracking metrics over epochs
    history = {'loss': [], 'val_loss': [], 'accuracy': [], 'val_accuracy': [], 'f1_score': [], 'val_f1_score': []}
    
    # For early stopping
    best_val_f1 = 0
    best_epoch = 0
    patience = 5
    no_improve = 0


    for epoch_num in range(EPOCHS):
        torch.cuda.empty_cache()
        model.train()
        total_loss_train = 0
        train_correct = 0
        train_f1_score = 0
        train_loss = 0

        # Training loop
        for batch_idx, (data, target) in enumerate(train_loader):
            data, target = data.permute(0, 3, 1, 2), target.float() # Permute dimensions
            #print('target:',target.float)
            optimizer.zero_grad()
            output = model(data)
            loss = criterion(output, target)
            #print('loss:', loss)
            train_loss += loss.item()
            #print('output:', output)

            softmax = nn.Softmax(dim=1)
            output = softmax(output)
            #print('target:',target.float)
            #print('output:', output)
            
            predictions = torch.argmax(output, dim=1).detach().cpu().numpy()
            #print('predictions:', predictions)

            target_numpy = target.detach().cpu().numpy()
            #print('target:',target_numpy)
            correct_predictions = np.sum(predictions == target_numpy.argmax(axis=1))
           
            #print('correct_predictions:', correct_predictions)

            batch_accuracy = correct_predictions / target_numpy.shape[0]
            #print("Number of correct predictions:", correct_predictions)
            #print("Accuracy of the batch:", batch_accuracy)
            train_correct += batch_accuracy
            #print(batch_accuracy)

            f1 = f1_score(target_numpy.argmax(axis=1), predictions, average='macro')
            train_f1_score += f1
            #print(f1)
            #print(train_f1_score)
            
            loss.backward()
            optimizer.step()

        epoch_loss = train_loss / len(train_loader)
        epoch_accuracy = train_correct / len(train_loader)
        epoch_f1_score = train_f1_score / len(train_loader)

        history['loss'].append(epoch_loss)
        history['accuracy'].append(epoch_accuracy)
        history['f1_score'].append(epoch_f1_score)
        
        print(history)

        model.eval()
        total_loss_val = 0
        val_correct = 0
        val_f1_score = 0
        val_loss = 0
        val_labels = []
        y_preds = []
        
        with torch.no_grad():
            for data, target in val_loader:
                data, target = data.permute(0, 3, 1, 2), target.float() # Permute dimensions
                #data = data.float()
                output = model(data)
                val_loss += criterion(output, target)
                softmax = nn.Softmax(dim=1)
                output = softmax(output)
                predictions = torch.argmax(output, dim=1).detach().cpu().numpy()
                #print('predictions:', predictions)

                target_numpy = target.detach().cpu().numpy()
                correct_predictions = np.sum(predictions == target_numpy.argmax(axis=1))

                #print('correct_predictions:', correct_predictions)

                batch_accuracy = correct_predictions / target_numpy.shape[0]
                #print("Number of correct predictions:", correct_predictions)
                #print("Accuracy of the batch:", batch_accuracy)
                val_correct += batch_accuracy
                
                # Calculate F1 score
                f1 = f1_score(target_numpy.argmax(axis=1), predictions, average='macro')
                val_f1_score += f1

            
        epoch_val_loss = val_loss / len(val_loader)
        epoch_val_accuracy = val_correct / len(val_loader)
        epoch_val_f1_score = val_f1_score / len(val_loader)

        history['val_loss'].append(epoch_val_loss)
        history['val_accuracy'].append(epoch_val_accuracy)
        history['val_f1_score'].append(epoch_val_f1_score)

        print(history)
            
        if epoch_val_f1_score > best_val_f1:
            best_val_f1 = epoch_val_f1_score
            best_epoch = epoch_num
            no_improve = 0

            # Save best model
            PATH = '/home/viktoriia.trokhova/model_weights/model_best.pt'
            torch.save(model.state_dict(), PATH)

        else:
            no_improve += 1

        if no_improve > patience:
            print("Early stopping at epoch: ", epoch_num)
            break

    return history, best_val_f1

  
history, best_val_f1 = train_and_evaluate(model, learning_rate_best, optimizer_best, dense_0_units_best, dense_1_units_best, batch_size_best)

plot_acc_loss_f1(history, '/home/viktoriia.trokhova/plots/', 'resnet')
