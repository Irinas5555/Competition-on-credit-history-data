from typing import List
import os
import torch
import numpy as np
import pandas as pd
import random
import torch.nn as nn
from tqdm import tqdm_notebook
from sklearn.metrics import roc_auc_score

from data_generators_with_mask import batches_generator
from torchvision.ops import sigmoid_focal_loss

from training_aux import EarlyStopping
from torch.nn.utils import clip_grad_norm_

def seed_everything(seed: int):   
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = True

SEED = 1171
seed_everything(SEED)

def train(model: torch.nn.Module,
                num_epochs: int,
                optimizer: torch.optim.Optimizer,
                loss_function,
                alpha, 
                dataset_train: List[str],
                dataset_val: List[str],
                path_to_checkpoints,
                best_model_name,
                scheduler=None,
                train_batch_size: int = 64,
                val_batch_size: int = 64, 
                shuffle: bool = True, 
                print_loss_every_n_batches: int = 500,
                device: torch.device = None,
                mask: bool = False,
                lenght: bool = False,
                ids: bool = False):
    
    es = EarlyStopping(patience=3, mode="max", verbose=True, save_path=os.path.join(path_to_checkpoints, best_model_name), 
                   metric_name="ROC-AUC", save_format="torch")
    val_scores = []
    for epoch in range(num_epochs):
        print(f"Starting epoch {epoch+1}")
        train_epoch(model, optimizer, loss_function,  dataset_train, alpha, batch_size=train_batch_size, 
                    shuffle=True, print_loss_every_n_batches=500, device=device, mask=mask, lenght=lenght, ids=ids)

        val_roc_auc = eval_model(model, loss_function, dataset_val, alpha, batch_size=val_batch_size,   device=device, mask=mask, lenght=lenght, ids=ids)
        val_scores.append(val_roc_auc)
        es(val_roc_auc, model)

        if es.early_stop:
            print("Early stopping reached. Stop training...")
            break
        torch.save(model.state_dict(), os.path.join(path_to_checkpoints, f"epoch_{epoch+1}_val_{val_roc_auc:.3f}.pt"))
        
        if scheduler is not None:
            scheduler.step(val_scores[-1])

        train_roc_auc = eval_model(model, loss_function,  dataset_train, alpha, batch_size=val_batch_size, device=device, mask=mask, lenght=lenght, ids=ids)
        print(f"Epoch {epoch+1} completed. Train ROC AUC: {train_roc_auc}, val ROC AUC: {val_roc_auc}")


def train_epoch(model: torch.nn.Module, 
                optimizer: torch.optim.Optimizer,
                loss_function,               
                dataset_train: List[str],
                alpha = 0.9,
                batch_size: int = 64, 
                shuffle: bool = True, 
                print_loss_every_n_batches: int = 500,
                device: torch.device = None,
                mask: bool = False,
                lenght: bool = False,
                ids: bool = False):
    """
    Делает одну эпоху обучения модели, логируя промежуточные значения функции потерь.

    Параметры:
    -----------
    model: torch.nn.Module
        Обучаемая модель.
    optimizer: torch.optim.Optimizer
        Оптимизатор.
    dataset_train: List[str]
        Список путей до файлов с предобработанными последовательностями.
    loss_function
        Функция потерь.
    alpha
        Коэффициент alpha, в случае если функция потерь Focal Loss
    batch_size: int, default=64
        Размер батча.
    shuffle: bool, default=False
        Перемешивать ли данные перед подачей в модель.
    print_loss_every_n_batches: int, default=500
        Число батчей.
    device: torch.device, default=None
        Девайс, на который переместить данные.
    mask: bool = False
        Используются ли в модели mask_sequences.
    lenght: bool = False
        Используются ли в модели lenght_sequences.
    ids: bool = False
        Используются ли в модели ids_segments.

    Возвращаемое значение:
    ----------------------
    None
    """
    model.train()
    losses = torch.Tensor().to(device)

    samples_counter = 0
    train_generator = batches_generator(dataset_train, batch_size=batch_size, shuffle=shuffle,
                                        device=device, is_train=True)

    for num_batch, batch in tqdm_notebook(enumerate(train_generator, start=1), desc="Training"):
        seed_everything(SEED)

        if mask & ids:
            output = torch.flatten(model(batch["features"], batch["mask"], batch["id_"]))
        elif mask:
            output = torch.flatten(model(batch["features"], batch["mask"]))
        elif lenght:
            output = torch.flatten(model(batch["features"], batch["lenght"]))
        else:
            output = torch.flatten(model(batch["features"]))

        if loss_function == sigmoid_focal_loss:
            batch_loss = loss_function(output, batch["label"].float(), alpha=alpha)
        else:
            batch_loss = loss_function(output, batch["label"].float())
                
        batch_loss.mean().backward()

        torch.nn.utils.clip_grad_norm_(model.parameters(), 1)

        seed_everything(SEED)
        optimizer.step()
        optimizer.zero_grad()

        samples_counter += batch_loss.size(0)

        losses = torch.cat([losses, batch_loss], dim=0)
        if num_batch % print_loss_every_n_batches == 0:
            print(f"Batches {num_batch - print_loss_every_n_batches + 1} - {num_batch} loss:"
                  f"{losses[-samples_counter:].mean()}", end="\r")
            samples_counter = 0

    print(f"Training loss after epoch: {losses.mean()}", end="\r")


def eval_model(model: torch.nn.Module,
               loss_function,               
               dataset_val: List[str], 
               alpha = 0.9,
               batch_size: int = 32, 
               device: torch.device = None,
               mask: bool = False,
               lenght: bool = False,
               ids: bool = False):
    """
    Скорит выборку моделью и вычисляет метрику ROC AUC.

    Параметры:
    -----------
    model: torch.nn.Module
        Модель, которой необходимо проскорить выборку.
    loss_function
        Функция потерь.
    alpha
        Коэффициент alpha, в случае если функция потерь Focal Loss
    dataset_val: List[str]
        Список путей до файлов с предобработанными последовательностями.
    batch_size: int, default=32
        Размер батча.
    device: torch.device, default=None
        Девайс, на который переместить данные.
    mask: bool = False
        Используются ли в модели mask_sequences.
    lenght: bool = False
        Используются ли в модели lenght_sequences.
    ids: bool = False
        Используются ли в модели ids_segments.

    Возвращаемое значение:
    ----------------------
    auc: float
    """
    model.eval()
    preds = []
    targets = []
    val_generator = batches_generator(dataset_val, batch_size=batch_size, shuffle=False,
                                      device=device, is_train=True)

    for batch in tqdm_notebook(val_generator, desc="Evaluating model"):
        targets.extend(batch["label"].detach().cpu().numpy().flatten())
        if mask & ids:
            output = torch.flatten(model(batch["features"], batch["mask"], batch["id_"]))
        elif mask:
            output = torch.flatten(model(batch["features"], batch["mask"]))
        elif lenght:
            output = torch.flatten(model(batch["features"], batch["lenght"]))
        else:
            output = model(batch["features"])
        
        preds.extend(output.detach().cpu().numpy().flatten())
        
    return roc_auc_score(targets, preds)


def inference(model: torch.nn.Module, 
              dataset_test: List[str], 
              batch_size: int = 32, 
              device: torch.device = None,
              mask: bool = False,
              lenght: bool = False,
              ids: bool = False) -> pd.DataFrame:
    """
    Скорит выборку моделью.

    Параметры:
    -----------
    model: torch.nn.Module
        Модель, которой необходимо проскорить выборку.
    dataset_test: List[str]
        Список путей до файлов с предобработанными последовательностями.
    batch_size: int, default=32
        Размер батча.
    device: torch.device, default=None
        Девайс, на который переместить данные.
    mask: bool = False
        Используются ли в модели mask_sequences.
    lenght: bool = False
        Используются ли в модели lenght_sequences.
    ids: bool = False
        Используются ли в модели ids_segments.

    Возвращаемое значение:
    ----------------------
    scores: pandas.DataFrame
        Датафрейм с двумя колонками: "id" - идентификатор заявки и "score" - скор модели.
    """
    model.eval()
    preds = []
    ids_ = []
    test_generator = batches_generator(dataset_test, batch_size=batch_size, shuffle=False,
                                       verbose=False, device=device, is_train=False)

    for batch in tqdm_notebook(test_generator, desc="Test predictions"):
        ids_.extend(batch["id_"])

        if mask & ids:
            output = torch.flatten(model(batch["features"], batch["mask"], batch["id_"]))
        elif mask:
            output = torch.flatten(model(batch["features"], batch["mask"]))
        elif lenght:
            output = torch.flatten(model(batch["features"], batch["lenght"]))
        else:
            output = torch.flatten(model(batch["features"]))
            
        output = torch.sigmoid(output)
        preds.extend(output.detach().cpu().numpy().flatten())

    return pd.DataFrame({
        "id": ids_,
        "score": preds
    })
