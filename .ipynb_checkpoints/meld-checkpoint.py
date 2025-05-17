import sys
import numpy as np
import argparse, time, pickle
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torch.utils.data.sampler import SubsetRandomSampler
from sklearn.metrics import f1_score, confusion_matrix, accuracy_score, classification_report
from models import MELD3LSTMModel, MELD7LSTMModel, MaskedFocalLoss,MaskedNLLLoss
from dataloader import MELDDataset, Dataset_M
from torch.optim import lr_scheduler
import os

def set_seed(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    np.random.seed(seed)
    

def get_train_valid_sampler(trainset, valid=0.1):
    size = len(trainset)
    idx = list(range(size))
    split = int(valid * size)
    return SubsetRandomSampler(idx[split:]), SubsetRandomSampler(idx[:split])


def get_Meld_loaders(path, classify, batch_size=32, valid=0.1, num_workers=0, pin_memory=False):
    
    # trainset = Dataset_M(path,classify,'train')
    trainset = MELDDataset(classify,'train')
    validset = MELDDataset(classify,'valid')
    testset = MELDDataset(classify,'test')
    
    train_loader = DataLoader(trainset,
                              batch_size=batch_size,
                              collate_fn=trainset.collate_fn,
                              num_workers=num_workers,
                              pin_memory=pin_memory)
    valid_loader = DataLoader(validset,
                              batch_size=batch_size,
                              collate_fn=trainset.collate_fn,
                              num_workers=num_workers,
                              pin_memory=pin_memory)
    test_loader = DataLoader(testset,
                             batch_size=batch_size,
                             collate_fn=testset.collate_fn,
                             num_workers=num_workers,
                             pin_memory=pin_memory)

    return train_loader, valid_loader, test_loader


def train_or_eval_model(model, loss_function, dataloader, scheduler, epoch, optimizer=None, train=False):
    losses = []
    preds = []
    labels = []
    masks = []
    alphas, alphas_f, alphas_b, vids = [], [], [], []
    assert not train or optimizer != None
    if train:
        model.train()
    else:
        model.eval()
    for data in dataloader:
        if train:
            optimizer.zero_grad()
        r1, r2, r3, r4, acouf, uu, sk, lk, nu, qmask, umask, label = \
            [d.cuda() for d in data[:-1]] if cuda else data[:-1]

        
        log_prob, alpha, alpha_f, alpha_b = model(r1,r2,r3,r4, acouf, qmask,umask,uu,sk,lk,nu)
        lp_ = log_prob.transpose(0, 1).contiguous().view(-1, log_prob.size()[2])
        labels_ = label.view(-1)
        loss = loss_function(lp_, labels_, umask)
        
        pred_ = torch.argmax(lp_, 1)
        preds.append(pred_.data.cpu().numpy())
        labels.append(labels_.data.cpu().numpy())
        masks.append(umask.view(-1).cpu().numpy())

        losses.append(loss.item() * masks[-1].sum())
        if train:
            loss.backward()
            if args.tensorboard:
                for param in model.named_parameters():
                    writer.add_histogram(param[0], param[1].grad, epoch)
            optimizer.step()
            scheduler.step()
        else:
            alphas += alpha
            alphas_f += alpha_f
            alphas_b += alpha_b
            vids += data[-1]

    if preds != []:
        preds = np.concatenate(preds)
        labels = np.concatenate(labels)
        masks = np.concatenate(masks)
    else:
        return float('nan'), float('nan'), [], [], [], float('nan'), []

    avg_loss = round(np.sum(losses) / np.sum(masks), 4)
    avg_accuracy = round(accuracy_score(labels, preds, sample_weight=masks) * 100, 2)
    avg_fscore = round(f1_score(labels, preds, sample_weight=masks, average='weighted') * 100, 2)
    m_fscore = round(f1_score(labels, preds, sample_weight=masks, average='macro') * 100, 2)
    return avg_loss, avg_accuracy, labels, preds, masks, avg_fscore, m_fscore, [alphas, alphas_f, alphas_b, vids]


if __name__ == '__main__':

    parser = argparse.ArgumentParser()
    parser.add_argument('--seed', type=int, default=4027, metavar='S', help='random seed (default: 1314)')
    parser.add_argument('--no-cuda', action='store_true', default=False, help='does not use GPU')
    parser.add_argument('--lr', type=float, default=0.0001, metavar='LR', help='learning rate')
    parser.add_argument('--l2', type=float, default=0.001, metavar='L2', help='L2 regularization weight')
    parser.add_argument('--dropout', type=float, default=0.3, metavar='dropout', help='dropout rate')
    parser.add_argument('--batch-size', type=int, default=64, metavar='BS', help='batch size')
    parser.add_argument('--epochs', type=int, default=25, metavar='E', help='number of epochs')
    parser.add_argument('--class-weight', action='store_true', default=True, help='use class weight')
    parser.add_argument('--attention', action='store_true', default=False, help='use attention on top of lstm')
    parser.add_argument('--tensorboard', action='store_true', default=False, help='Enables tensorboard log')
    parser.add_argument('--mode1', type=int, default=0, help='Roberta features to use')
    parser.add_argument('--norm', type=int, default=0, help='normalization strategy')
    args = parser.parse_args()

    print(args)

    set_seed(args.seed)

    args.cuda = torch.cuda.is_available() and not args.no_cuda
    if args.cuda:
        print('Running on GPU')
    else:
        print('Running on CPU')

    if args.tensorboard:
        from tensorboardX import SummaryWriter

        writer = SummaryWriter()

    batch_size = args.batch_size
    cuda = args.cuda
    n_epochs = args.epochs
    
    classify = 'emotion' 
    D_r = 500
    D_m = 600
    D_e = 500
    D_h = 500
    D_u = 500
    D_k = 500
    N_s = 9
    
    path = 'meld/MELD_features_raw.pkl'
    best_model_path = os.path.join("model/", "best_model.pth")
    
    if classify == 'emotion':
        n_classes = 7
        model = MELD7LSTMModel(D_u, D_k, D_r, D_m, D_e, D_h, N_s,
                      n_classes=n_classes,
                      dropout=args.dropout,
                      attention=args.attention,mode1=args.mode1,norm=args.norm)
        
        weight = [0.30427062, 1.19699616, 5.47007183, 1.95437696, 
                0.84847735, 5.42461417, 1.21859721]
        weight = np.log1p(weight)  # log(1 + weight)
        weight = weight / np.max(weight)
        weight = weight * 0.5 + 0.5
        
        loss_weights = torch.FloatTensor(weight)
    else:
        n_classes = 3
        model = MELD3LSTMModel(D_u, D_k, D_r, D_m, D_e, D_h,N_s,
                      n_classes=n_classes,
                      dropout=args.dropout,
                      attention=args.attention,mode1=args.mode1,norm=args.norm)
   
   
    if cuda:
        model.cuda()


    if args.class_weight:
        print("use weighted_loss")
        loss_function = MaskedFocalLoss(weight = loss_weights.cuda() if cuda else loss_weights, gamma = 2.0)
    else:
        loss_function = MaskedFocalLoss()
        
    optimizer = optim.AdamW(model.parameters(),
                           lr=args.lr,
                           weight_decay=args.l2)
    scheduler = lr_scheduler.CosineAnnealingLR(optimizer,T_max=10)

    train_loader, valid_loader, test_loader = get_Meld_loaders(path=path,classify=classify,
                                                                  batch_size=batch_size,
                                                                  valid=0.0)
    print('n_classes=',n_classes)

    best_fscore, best_loss, best_label, best_pred, best_mask = None, None, None, None, None

    for e in range(n_epochs):
        start_time = time.time()
        train_loss, train_acc, _, _, _, train_fscore_w, train_fscore_m, _ = train_or_eval_model(model, loss_function,train_loader,scheduler, e, optimizer, True)
        valid_loss, valid_acc, _, _, _, val_fscore_w, val_fscore_m, _= train_or_eval_model(model, loss_function, valid_loader,scheduler, e)
        test_loss, test_acc, test_label, test_pred, test_mask, test_fscore_w, test_fscore_m, attentions = train_or_eval_model(model,loss_function,test_loader,scheduler,e)

        if best_fscore == None or best_fscore < test_fscore_w:
            best_fscore, best_loss, best_label, best_pred, best_mask, best_attn = \
                test_fscore_w, test_loss, test_label, test_pred, test_mask, attentions
            torch.save({
            'epoch': e + 1,
            'model_state_dict': model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'best_fscore_w': best_fscore,
            'best_loss': best_loss
        }, best_model_path)
        print(f"New best model saved at epoch {e + 1} with test_fscore_w: {best_fscore:.4f}")

        if args.tensorboard:
            writer.add_scalar('test: accuracy/loss', test_acc / test_loss, e)
            writer.add_scalar('train: accuracy/loss', train_acc / train_loss, e)
        print(
            '===========epoch {}===========\n train_loss {} train_acc {} train_fscore_w {} train_fscore_m {}\n valid_loss {} valid_acc {} val_fscore_w {} val_fscore_m {}\n test_loss {} test_acc {} test_fscore_w {} test_fscore_m {}\n time {}'. \
            format(e + 1, train_loss, train_acc, train_fscore_w, train_fscore_m, valid_loss, valid_acc, val_fscore_w, val_fscore_m,\
                   test_loss, test_acc, test_fscore_w, test_fscore_m, round(time.time() - start_time, 2)))
    if args.tensorboard:
        writer.close()

    print('Test performance..')
    print('Loss {} weightedAvgF1-score {} macroF1-score {}'.format(best_loss,
                                                                   round(f1_score(best_label, best_pred, sample_weight=best_mask,average='weighted') *100, 2),
                                                                   round(f1_score(best_label, best_pred, sample_weight=best_mask,average='macro') * 100, 2)))
    
    print(classification_report(best_label, best_pred, sample_weight=best_mask, digits=4))
print(confusion_matrix(best_label, best_pred, sample_weight=best_mask))















