# -*- coding: utf-8 -*-
import warnings
import numpy as np
import torch
import torch.nn.functional as F
from sklearn.metrics import roc_auc_score, average_precision_score, accuracy_score, precision_score, f1_score, \
    recall_score


warnings.filterwarnings("ignore", message="No positive class found in y_true*")
def Model_Train(model, optimizer, scheduler, data_iter, device, loss, configs):
    model.train()
    loss.reset()
    torch.autograd.set_detect_anomaly(True)
    for x_t, x_f, _ in data_iter:
        optimizer.zero_grad()
        x_t, x_f = x_t.to(device), x_f.to(device)  #t: torch.Size([64, 1, 1024])  f : torch.Size([64, 1, 512])

        loss_tf = model(x_t, x_f)
        batch_size= x_t.shape[0]

        l = loss_tf

        l.backward()
        optimizer.step()
        with torch.no_grad():
            loss.update((l, batch_size))
    scheduler.step()
    return loss.compute()

def Model_Finetune(model, optimizer, scheduler, data_iter, device, loss, acc):
    model.train()
    loss.reset()
    acc.reset()

    for x_t, x_f, y in data_iter:
        optimizer.zero_grad()
        x_t, x_f = x_t.to(device), x_f.to(device)  # t: torch.Size([64, 1, 1024])  f : torch.Size([64, 1, 512])
        y = y.to(device)

        y_hat = model(x_t,x_f)

        batch_size= x_t.shape[0]
        l = F.cross_entropy(y_hat, y)
        l.backward()
        optimizer.step()
        with torch.no_grad():
            loss.update((l, batch_size))
            acc.update((y_hat, y))

    scheduler.step()
    total_loss = loss.compute()
    total_acc = acc.compute()

    return total_loss, total_acc

def Model_Test(model, data_iter, device, acc,logger):
    model.eval()
    acc.reset()

    total_auc = []
    total_prc = []

    with torch.no_grad():

        labels_numpy_all, pred_numpy_all = np.zeros(1), np.zeros(1)

        for x_t, x_f, y in data_iter:
            x_t, x_f = x_t.to(device), x_f.to(device)  # t: torch.Size([64, 1, 1024])  f : torch.Size([64, 1, 512])
            y = y.to(device)

            y_hat, features = model(x_t,x_f, return_features=True)

            acc.update((y_hat, y))
            #####################################
            onehot_label = F.one_hot(y, num_classes=y_hat.shape[1])
            pred_numpy = y_hat.detach().cpu().numpy()

            labels_numpy = y.detach().cpu().numpy()

            try:
                auc_bs = roc_auc_score(onehot_label.detach().cpu().numpy(), pred_numpy, average="macro",
                                       multi_class="ovr")
            except:
                auc_bs = float(0)
            prc_bs = average_precision_score(onehot_label.detach().cpu().numpy(), pred_numpy)
            pred_numpy = np.argmax(pred_numpy, axis=1)

            total_auc.append(auc_bs)
            total_prc.append(prc_bs)

            labels_numpy_all = np.concatenate((labels_numpy_all, labels_numpy))
            pred_numpy_all = np.concatenate((pred_numpy_all, pred_numpy))

        total_acc = acc.compute()

    labels_numpy_all = labels_numpy_all[1:]
    pred_numpy_all = pred_numpy_all[1:]

    precision = precision_score(labels_numpy_all, pred_numpy_all, average='macro', zero_division=0)
    recall = recall_score(labels_numpy_all, pred_numpy_all, average='macro', zero_division=0)
    F1 = f1_score(labels_numpy_all, pred_numpy_all, average='macro', zero_division=0)
    acc = accuracy_score(labels_numpy_all, pred_numpy_all, )

    total_auc = torch.tensor(total_auc).mean()
    total_prc = torch.tensor(total_prc).mean()

    logger.debug("\n################## Best testing performance! #########################\n")
    logger.debug('MLP Testing: Acc=%.4f| Precision = %.4f | Recall = %.4f | F1 = %.4f | AUROC= %.4f | AUPRC=%.4f'
                 % (acc * 100, precision * 100, recall * 100, F1 * 100, total_auc * 100, total_prc * 100))

    return total_acc