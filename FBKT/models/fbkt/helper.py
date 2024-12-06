import math
import random
from .Network import MYNET
from utils import *
from tqdm import tqdm
import torch
from torch import nn
import torch.nn.functional as F
from .mask_util import mask_transform, color_jitter, mask_qk_transform
from losses import SupContrastive, incftContrastive, PriorDist, JSDivLoss


def base_train(model, trainloader, criterion, class_criterion, optimizer, scheduler, epoch, transform, args):
    tl = Averager()
    tl_joint = Averager()
    tl_class_loss = Averager()
    tl_mask = Averager()
    tl_IAC = Averager()
    ta = Averager()
    model = model.train()
    tqdm_gen = tqdm(trainloader)


    for i, batch in enumerate(tqdm_gen, 1):
        data, single_labels = [_ for _ in batch]
        b, c, h, w = data[1].shape
        original = data[0].cuda(non_blocking=True)
        data[1] = data[1].cuda(non_blocking=True)
        data[2] = data[2].cuda(non_blocking=True)
        single_labels = single_labels.cuda(non_blocking=True)
        if len(args.num_crops) > 1:
            data_small = data[args.num_crops[0] + 1].unsqueeze(1)
            for j in range(1, args.num_crops[1]):
                data_small = torch.cat((data_small, data[j + args.num_crops[0] + 1].unsqueeze(1)), dim=1)
            data_small = data_small.view(-1, c, args.size_crops[1], args.size_crops[1]).cuda(non_blocking=True)
        else:
            data_small = None

        im_size = original.shape[1:]
        data_classify = transform(original)
        data_mask, data_unmask_q, data_unmask_k = mask_qk_transform(data_classify, data[1], data[2])
        data_mask = data_mask.cuda()
        data_unmask_q = data_unmask_q.cuda()
        data_unmask_k = data_unmask_k.cuda()
        data_query = torch.stack([data[1], data_unmask_q], 1).view(-1, *im_size)
        data_key = torch.stack([data[2], data_unmask_k], 1).view(-1, *im_size)

        data_small = transform(data_small)

        data_aug = color_jitter(data_classify)
        data_aug = data_aug.cuda()

        m = data_query.size()[0] // b
        joint_labels = torch.stack([single_labels * m + ii for ii in range(m)], 1).view(-1)

        joint_preds, aug_preds, mask_preds, output_global, output_small, target_global, target_small = model(
            im_cla=data_classify, im_aug=data_aug, im_mask=data_mask, im_q=data_query, im_k=data_key,
            labels=joint_labels, im_q_small=data_small)

        loss_IAC_global = criterion(output_global, target_global)
        loss_IAC_small = criterion(output_small, target_small)
        loss_IAC = 0.8 * loss_IAC_global + 0.2 * loss_IAC_small

        joint_preds = joint_preds[:, :args.base_class * m]
        joint_loss = F.cross_entropy(joint_preds, joint_labels)
        mask_preds = mask_preds[:, :args.base_class * m]
        kl_loss_function = nn.KLDivLoss(reduction='batchmean')  # 设置reduction='sum'以对KL散度项进行求和
        kl_loss = kl_loss_function(mask_preds.log_softmax(dim=-1), joint_preds.softmax(dim=-1))

        aug_preds = aug_preds[:, :args.base_class * m]
        class_loss = class_criterion(joint_preds.softmax(dim=1), aug_preds.softmax(dim=1))

        agg_preds = 0
        for i in range(m):
            agg_preds = agg_preds + joint_preds[i::m, i::m] / m

        loss = joint_loss + loss_IAC + args.alpha * class_loss + args.beta * kl_loss

        total_loss = loss

        acc = count_acc(agg_preds, single_labels)

        lrc = scheduler.get_last_lr()[0]
        tqdm_gen.set_description(
            'Session 0, epo {}, lrc={:.4f},total loss={:.4f} acc={:.4f}'.format(epoch, lrc, total_loss.item(), acc))
        tl.add(total_loss.item())
        tl_joint.add(joint_loss.item())
        tl_class_loss.add(class_loss.item())
        tl_mask.add(kl_loss.item())
        tl_IAC.add(loss_IAC.item())
        ta.add(acc)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

    tl = tl.item()
    ta = ta.item()
    tl_joint = tl_joint.item()
    tl_class_loss = tl_class_loss.item()
    tl_mask = tl_mask.item()
    tl_IAC = tl_IAC.item()
    return tl, tl_mask, tl_joint, tl_class_loss, tl_IAC, ta


def replace_base_fc(trainset, test_transform, data_transform, model, args):
    # replace fc.weight with the embedding average of train data
    model = model.eval()

    trainloader = torch.utils.data.DataLoader(dataset=trainset, batch_size=128,
                                              num_workers=8, pin_memory=True, shuffle=False)
    trainloader.dataset.transform = test_transform
    embedding_list = []
    label_list = []
    # data_list=[]
    with torch.no_grad():
        for i, batch in enumerate(trainloader):
            data, label = [_.cuda() for _ in batch]
            b = data.size()[0]
            data = data_transform(data)
            m = data.size()[0] // b
            labels = torch.stack([label*m+ii for ii in range(m)], 1).view(-1)
            model.mode = 'encoder'
            embedding = model(data)
            embedding_list.append(embedding.cpu())
            label_list.append(labels.cpu())
    embedding_list = torch.cat(embedding_list, dim=0)
    label_list = torch.cat(label_list, dim=0)

    proto_list = []

    for class_index in range(args.base_class*m):
        data_index = (label_list == class_index).nonzero()
        embedding_this = embedding_list[data_index.squeeze(-1)]
        embedding_this = embedding_this.mean(0)
        proto_list.append(embedding_this)

    proto_list = torch.stack(proto_list, dim=0)

    model.fc.weight.data[:args.base_class*m] = proto_list

    return model


def update_fc_ft(trainloader, data_transform, model, pre_model, m, session, args, epoch):
    # incremental finetuning
    old_class = args.base_class + args.way * (session - 1)
    new_class = args.base_class + args.way * session 
    new_fc = nn.Parameter(
        torch.rand(args.way*m, model.num_features, device="cuda"),
        requires_grad=True)
    new_fc.data.copy_(model.fc.weight[old_class*m : new_class*m, :].data)

    if args.dataset == 'mini_imagenet':
        optimizer = torch.optim.SGD([{'params': new_fc, 'lr': args.lr_new},
                                     {'params': model.encoder_q.fc.parameters(), 'lr': 0.11*args.lr_new},
                                     {'params': model.encoder_q.layer4.parameters(), 'lr': 0.001 * args.lr_new},],
                                    momentum=0.9, dampening=0.9, weight_decay=0)


    if args.dataset == 'cub200':
        optimizer = torch.optim.SGD([{'params': new_fc, 'lr': args.lr_new},
                                     {'params': model.encoder_q.fc.parameters(), 'lr': 0.05 * args.lr_new},
                                     {'params': model.encoder_q.layer4.parameters(), 'lr': 0.001 * args.lr_new}, ],
                                    momentum=0.9, dampening=0.9, weight_decay=0)



    if args.dataset == 'cifar100':
        optimizer = torch.optim.SGD([{'params': new_fc, 'lr': args.lr_new},
                                     {'params': model.encoder_q.fc.parameters(), 'lr': args.lr_new},
                                     {'params': model.encoder_q.layer3.parameters(), 'lr': 0.001 * args.lr_new}, ],
                                    momentum=0.9, dampening=0.9, weight_decay=0)

    if args.dataset == 'stanfordcar':
        optimizer = torch.optim.SGD([{'params': new_fc, 'lr': args.lr_new},
                                     {'params': model.encoder_q.fc.parameters(), 'lr': 0.05 * args.lr_new},
                                     {'params': model.encoder_q.layer4.parameters(), 'lr': 0.001 * args.lr_new}, ],
                                    momentum=0.9, dampening=0.9, weight_decay=0)

        
    criterion = SupContrastive().cuda()
    incft_criterion = incftContrastive().cuda()

    with torch.enable_grad():
        # for epoch in range(args.epochs_new):
        for batch in trainloader:
            data, single_labels = [_ for _ in batch]
            b, c, h, w = data[1].shape
            origin = data[0].cuda(non_blocking=True)
            data[1] = data[1].cuda(non_blocking=True)
            data[2] = data[2].cuda(non_blocking=True)
            single_labels = single_labels.cuda(non_blocking=True)
            if len(args.num_crops) > 1:
                data_small = data[args.num_crops[0] + 1].unsqueeze(1)
                for j in range(1, args.num_crops[1]):
                    data_small = torch.cat((data_small, data[j + args.num_crops[0] + 1].unsqueeze(1)), dim=1)
                data_small = data_small.view(-1, c, args.size_crops[1], args.size_crops[1]).cuda(non_blocking=True)
            else:
                data_small = None

        data_classify = data_transform(origin)
        im_size = origin.shape[1:]
        data_mask, _ = mask_transform(data_classify)
        _, data_unmask_q = mask_transform(data[1])
        _, data_unmask_k = mask_transform(data[2])
        data_mask = data_mask.cuda()
        data_unmask_q = data_unmask_q.cuda()
        data_unmask_k = data_unmask_k.cuda()
        data_query = torch.stack([data[1], data_unmask_q], 1).view(-1, *im_size)
        data_key = torch.stack([data[2], data_unmask_k], 1).view(-1, *im_size)

        data_small = data_transform(data_small)

        joint_labels = torch.stack([single_labels * m + ii for ii in range(m)], 1).view(-1)
        old_fc = model.fc.weight[:old_class * m, :].clone().detach()
        fc = torch.cat([old_fc, new_fc], dim=0)
        features, _ = model.encode_q(data_classify)
        features.detach()
        logits = model.get_logits(features, fc)
        joint_loss = F.cross_entropy(logits, joint_labels)

        pre_fc = pre_model.fc.weight[:new_class * m, :].clone().detach()
        pre_features, _ = pre_model.encode_q(data_classify)
        pre_features.detach()
        pre_logits = pre_model.get_logits(pre_features, pre_fc)

        _, _, mask_preds, output_global, output_small, target_global, target_small = model(im_cla=data_classify, im_aug=None, im_mask=data_mask, im_q=data_query, im_k=data_key, labels=joint_labels, im_q_small=data_small, base_sess=False, last_epochs_new=(epoch == args.epochs_new - 1))
        loss_IAC_global = criterion(output_global, target_global)
        loss_IAC_small = criterion(output_small, target_small)
        loss_IAC = 0.8 * loss_IAC_global + 0.2 * loss_IAC_small

        joint_preds = logits[:, args.base_class * m:new_class * m]
        mask_preds = mask_preds[:, args.base_class * m:new_class * m]

        kl_loss_function = nn.KLDivLoss(reduction='batchmean')  # 设置reduction='sum'以对KL散度项进行求和
        kl_loss = kl_loss_function(mask_preds.log_softmax(dim=-1), joint_preds.softmax(dim=-1))

        origin_class = logits[:, :old_class * m]
        pre_class = pre_logits[:, :old_class * m]

        session_t_loss = incft_criterion(origin_class.softmax(dim=1), pre_class.softmax(dim=1))

        loss = joint_loss + loss_IAC + args.beta * kl_loss + session_t_loss

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

    model.fc.weight.data[old_class * m: new_class * m, :].copy_(new_fc.data)

def test(model, testloader, epoch, transform, args, session):
    test_class = args.base_class + session * args.way
    model = model.eval()
    vl = Averager()
    va = Averager()
    with torch.no_grad():
        tqdm_gen = tqdm(testloader)
        for i, batch in enumerate(tqdm_gen, 1):
            data, test_label = [_.cuda() for _ in batch]
            b = data.size()[0]
            data = transform(data)
            m = data.size()[0] // b
            joint_preds = model(data)
            joint_preds = joint_preds[:, :test_class*m]
            agg_preds = 0
            for j in range(m):
                agg_preds = agg_preds + joint_preds[j::m, j::m] / m

            loss = F.cross_entropy(agg_preds, test_label)
            acc = count_acc(agg_preds, test_label)

            vl.add(loss.item())
            va.add(acc)

        vl = vl.item()
        va = va.item()
    print('epo {}, test, loss={:.4f} acc={:.4f}'.format(epoch, vl, va))

    return vl,va


