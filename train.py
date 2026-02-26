import datetime
import torch
from CNN2 import Net , NetDropout , DeepNet , ResBlock
from rps_dataloader import  RPSDataLoaderAugmented , RPSDataLoaderGreenAugmented , RPS_Dataloader
from torch import optim
from sklearn.metrics import confusion_matrix, classification_report, accuracy_score, precision_recall_fscore_support
from torch.utils.data import Dataset, DataLoader, random_split
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

device = (torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu'))

def training_loop(n_epochs, optimizer, model, loss_fn, train_loader, val_loader = None):

    history = {
        'train_loss': [],
        'train_acc': [],
        'val_loss': [],
        'val_acc': []
    }

    for epoch in range(1, n_epochs + 1):
        loss_train = 0.0
        correct = 0
        total = 0
        for imgs, labels in train_loader:
            optimizer.zero_grad()
            imgs = imgs.to(device=device)
            labels = labels.to(device=device)
            outputs = model(imgs)
            loss = loss_fn(outputs, labels)
            loss.backward()
            optimizer.step()
            # Statistics
            loss_train += loss.item()
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()

        if epoch == 1 or epoch == n_epochs or epoch % 10 == 0:
            print('{} Epoch {}, Training loss {}'.format(
            datetime.datetime.now(), epoch,
            loss_train / len(train_loader)))

        if val_loader:
            val_loss, val_acc = validate(model, val_loader, loss_fn, device)
            
            # Save metrics
            history['train_loss'].append(loss_train / len(train_loader))
            history['train_acc'].append(100 * correct / total)
            history['val_loss'].append(val_loss)
            history['val_acc'].append(val_acc)

    return history

# def validate(model, train_loader, val_loader , loss_fn):
#     for name, loader in [("train", train_loader), ("val", val_loader)]:
#         correct = 0
#         total = 0
#         loss_total = 0.0
#         # We do not want gradients
#         # here, as we will not want to
#         # update the parameters.
#         with torch.no_grad():
#             for imgs, labels in loader:
#                 imgs = imgs.to(device=device)
#                 labels = labels.to(device=device)
#                 outputs = model(imgs)
#                 _, predicted = torch.max(outputs, dim=1)
#                 total += labels.shape[0]
#                 correct += int((predicted == labels).sum())
#         print("Accuracy {}: {:.2f}".format(name , correct / total))
def validate(model, val_loader, criterion, device):
    """Validate model"""
    model.eval()
    running_loss = 0.0
    correct = 0
    total = 0
    
    with torch.no_grad():
        for images, labels in val_loader:
            images, labels = images.to(device), labels.to(device)
            
            # Forward pass
            outputs = model(images)
            loss = criterion(outputs, labels)
            
            # Statistics
            running_loss += loss.item()
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
    
    epoch_loss = running_loss / len(val_loader)
    epoch_acc = 100 * correct / total
    print("Validation loss: {}, Accuracy: {}%".format(epoch_loss, epoch_acc))
    return epoch_loss, epoch_acc


def plot_training_curves(history, model_name, save_path=None):
    """Plot training and validation loss/accuracy curves"""
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    
    epochs = range(1, len(history['train_loss']) + 1)
    
    # Loss plot
    axes[0].plot(epochs, history['train_loss'], 'b-', label='Training Loss', linewidth=2)
    axes[0].plot(epochs, history['val_loss'], 'r-', label='Validation Loss', linewidth=2)
    axes[0].set_title(f'{model_name} - Loss', fontsize=14, fontweight='bold')
    axes[0].set_xlabel('Epoch', fontsize=12)
    axes[0].set_ylabel('Loss', fontsize=12)
    axes[0].legend(fontsize=11)
    axes[0].grid(True, alpha=0.3)
    
    # Accuracy plot
    axes[1].plot(epochs, history['train_acc'], 'b-', label='Training Accuracy', linewidth=2)
    axes[1].plot(epochs, history['val_acc'], 'r-', label='Validation Accuracy', linewidth=2)
    axes[1].set_title(f'{model_name} - Accuracy', fontsize=14, fontweight='bold')
    axes[1].set_xlabel('Epoch', fontsize=12)
    axes[1].set_ylabel('Accuracy (%)', fontsize=12)
    axes[1].legend(fontsize=11)
    axes[1].grid(True, alpha=0.3)
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
    else:
        plt.show()


def plot_confusion_matrix(cm, class_names, model_name, save_path=None):
    """Plot confusion matrix"""
    plt.figure(figsize=(8, 6))
    
    # Normalize confusion matrix
    cm_normalized = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
    
    sns.heatmap(cm_normalized, annot=True, fmt='.2%', cmap='Blues', 
                xticklabels=class_names, yticklabels=class_names,
                cbar_kws={'label': 'Percentage'})
    
    plt.title(f'{model_name} - Confusion Matrix', fontsize=14, fontweight='bold')
    plt.ylabel('True Label', fontsize=12)
    plt.xlabel('Predicted Label', fontsize=12)
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
    else:       
        plt.show()


def evaluate_model(model, test_loader, device, class_names):
    """
    Comprehensive model evaluation
    
    Returns:
        metrics: Dictionary containing all evaluation metrics
    """
    model.eval()
    all_preds = []
    all_labels = []
    
    with torch.no_grad():
        for images, labels in test_loader:
            images = images.to(device)
            outputs = model(images)
            print(f"Outputs: {outputs}")
            _, predicted = torch.max(outputs.data, 1)
            print(f"Predicted: {predicted}, True Labels: {labels}")
            all_preds.extend(predicted.cpu().numpy())
            all_labels.extend(labels.numpy())
    
    all_preds = np.array(all_preds)
    all_labels = np.array(all_labels)
    
    # Calculate metrics
    accuracy = accuracy_score(all_labels, all_preds)
    precision, recall, f1, _ = precision_recall_fscore_support(
        all_labels, all_preds, average='weighted'
    )
    
    # Per-class metrics
    precision_per_class, recall_per_class, f1_per_class, _ = precision_recall_fscore_support(
        all_labels, all_preds, average=None, labels=range(len(class_names))
    )
    
    metrics = {
        'accuracy': accuracy,
        'precision': precision,
        'recall': recall,
        'f1_score': f1,
        'per_class': {
            class_names[i]: {
                'precision': precision_per_class[i],
                'recall': recall_per_class[i],
                'f1_score': f1_per_class[i]
            }
            for i in range(len(class_names))
        },
        'confusion_matrix': confusion_matrix(all_labels, all_preds),
        'predictions': all_preds,
        'true_labels': all_labels
    }
    
    return metrics


def grid_search_with_K_fold(model_class, param_grid, train_dataset, k=5 , log_file=None, model_name=None):
    from sklearn.model_selection import ParameterGrid, KFold
    import numpy as np

    best_params = None
    best_score = 0.0

    # Create K-Fold cross-validator
    kf = KFold(n_splits=k, shuffle=True, random_state=42)

    # Iterate over all combinations of hyperparameters
    for params in ParameterGrid(param_grid):
        print(f"Testing parameters: {params}")
        fold_scores = []

        # Perform K-Fold cross-validation
        for train_index, val_index in kf.split(train_dataset):
            # Create data loaders for the current fold
            train_subset = torch.utils.data.Subset(train_dataset, train_index)
            val_subset = torch.utils.data.Subset(train_dataset, val_index)
            train_fold_loader = torch.utils.data.DataLoader(train_subset, batch_size=params['batch_size'], shuffle=True)
            val_fold_loader = torch.utils.data.DataLoader(val_subset, batch_size= params['batch_size'], shuffle=False)

            # Initialize the model with the current set of hyperparameters
            model = model_class().to(device)
            # Train the model on the current fold
            optimizer = params.get('optimizer')(model.parameters(), lr=params.get('learning_rate'))
            loss_fn = torch.nn.CrossEntropyLoss()
            training_loop(n_epochs=params['n_epoch'], optimizer=optimizer, model=model, loss_fn=loss_fn, train_loader=train_fold_loader)

            # Validate the model on the validation fold
            correct = 0
            total = 0
            with torch.no_grad():
                for imgs, labels in val_fold_loader:
                    imgs = imgs.to(device=device)
                    labels = labels.to(device=device)
                    outputs = model(imgs)
                    _, predicted = torch.max(outputs, dim=1)
                    total += labels.shape[0]
                    correct += int((predicted == labels).sum())
            fold_score = validate(model, val_fold_loader, loss_fn, device)[1]  # Get validation accuracy
            fold_scores.append(fold_score)

        avg_score = np.mean(fold_scores)
        print(f"Average score for parameters {params}: {avg_score}")
        if log_file:
        #salvataggio
            with open(log_file, "a") as f:
                f.write(f"DATE: {datetime.datetime.now()} | MODEL: {model_name}\n")
                f.write(f"PARAMS: {params}\n")
                f.write(f"AVG ACCUARCY (K-FOLD): {avg_score:.6f}\n")
                f.write("-" * 30 + "\n")

        if avg_score > best_score:
            best_score = avg_score
            best_params = params

    print(f"Best parameters: {best_params} with score: {best_score}")
    if log_file:
        #salvataggio
        with open(log_file, "a") as f:
            f.write(f"DATE: {datetime.datetime.now()} | MODEL: {model_name}\n")
            f.write(f"Best parameters: {best_params} with score: {best_score}\n")
    
    return best_params



def test_model_best_params(model_class, best_params, train_dataset, test_loader, log_file=None, model_name=None, dataset_name=None):
    # Initialize the model with the best hyperparameters
    model = model_class().to(device)
    optimizer = best_params.get('optimizer')(model.parameters(), lr=best_params.get('learning_rate'))
    loss_fn = torch.nn.CrossEntropyLoss()
    
    # Train the model on the entire training dataset
    train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=best_params['batch_size'], shuffle=True)
    history = training_loop(n_epochs=best_params['n_epoch'], optimizer=optimizer, model=model, loss_fn=loss_fn, train_loader=train_loader , val_loader=test_loader)

    # Validate the model on the validation set
    metrics = evaluate_model(model, test_loader, device, ['paper', 'rock', 'scissors'])
    plot_training_curves(history, model_name ,save_path=f'{dataset_name}_{model_name}_training_curves.png')
    plot_confusion_matrix(metrics['confusion_matrix'], ['paper', 'rock', 'scissors'], model_name , save_path=f'{dataset_name}_{model_name}_confusion_matrix.png')
    
    print(f"Test Metrics: {metrics}")
    if log_file:
        with open(log_file, "a") as f:
            f.write(f"DATE: {datetime.datetime.now()} | MODEL: {model_name}\n")
            f.write(f"Test Metrics: {metrics}\n")
    torch.save(model.state_dict(), f'{dataset_name}_{model_name}.pt')
    return metrics

def main():

    data_loader = RPSDataLoaderAugmented(
        data_dir='../dataset',  # Update this path to your dataset
        img_size=(200, 300),)

    # model = Net()
    models = [(NetDropout, "NetDropout")]
    best_params = {}
    test_values = {}
    for model_class, model_name in models:
        print(f"Testing model: {model_name}")
        best_params[model_name] = grid_search_with_K_fold(
            model_class=model_class,
            param_grid={
                'n_epoch': [10, 30, 50],
                'learning_rate': [0.01, 0.001, 0.0001],
                'batch_size': [32, 64, 128],
                'optimizer': [optim.Adam, optim.SGD]
            },
            train_dataset=data_loader.train_dataset,
            k=3,
            log_file=f'{model_name}_DataAug_grid_search_log.txt',
            model_name=model_name
        )
        print(f"Best parameters for {model_name}: {best_params[model_name]}")
        test_values[model_name] = test_model_best_params(
            model_class=model_class,
            best_params=best_params[model_name],
            train_dataset=data_loader.train_dataset,
            test_loader=data_loader.get_test_loader(),
            log_file=f'{model_name}_DataAugmented_test_log.txt',
            model_name=model_name,
            dataset_name="DataAugmented"
        )
        print(f"Test results for {model_name}: {test_values[model_name]}")
    
    print(f"Best parameters for all models: {best_params}")
    print(f"Test results for all models: {test_values}")
    # model = DeepNet()
    # model = ResBlock()
    # model().to(device)
    # # optimizer = optim.SGD(model.parameters(), lr=0.001, weight_decay=0.001)
    # optimizer = optim.Adam(model.parameters(), lr=0.001)
    # loss_fn = torch.nn.CrossEntropyLoss()
    # print(f"Training on device {device}.")
    # training_loop(
    #     n_epochs=50,
    #     optimizer=optimizer,
    #     model=model,
    #     loss_fn=loss_fn,
    #     train_loader=train_loader
    # )

    # validate(model, train_loader, val_loader)
    # validate(model, train_loader, test_loader)

main()