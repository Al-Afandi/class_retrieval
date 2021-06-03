import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.autograd import Variable
from torch.utils.data import Dataset, DataLoader
from torch.autograd.gradcheck import zero_gradients
from torch.autograd import Variable
from torchvision import datasets, transforms
from torchvision.datasets import CIFAR10

from cifar10_models.cifar10_models import *
import torchattacks as ta

import matplotlib.pyplot as plt
import numpy as np
import pickle as p
import seaborn as sn
import pandas as pd
import copy
from tqdm import tqdm


device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")


def pgd_momen(model, image_tensor, eps, alpha, num_iter, y,decay_factor=0.1,targeted = False):
    """ Construct targeted adversarial examples on the examples X"""
    img_variable = Variable(image_tensor, requires_grad=True)
    g = torch.zeros_like(image_tensor)
    for i in range(num_iter):
        output = model(img_variable)
        loss = torch.nn.CrossEntropyLoss()
        loss_cal = loss(output, torch.LongTensor([y]).cuda())    
        loss_cal.backward()
        g = decay_factor * g + img_variable.grad.data
        x_grad = alpha * torch.sign(g)
        if not targeted:
            x_grad *= -1
        adv_temp = img_variable.data - x_grad
        total_grad = adv_temp - image_tensor
        total_grad = torch.clamp(total_grad, -eps, eps)
        x_adv = image_tensor + total_grad
        img_variable.data = x_adv
        img_variable.grad.zero_()
    return img_variable.detach()


def pgd(model, image_tensor, eps, alpha, num_iter, y,targeted = False):
    """ Construct targeted adversarial examples on the examples X"""
    img_variable = Variable(image_tensor, requires_grad=True)
    for i in range(num_iter):
        output = model(img_variable)
        loss = torch.nn.CrossEntropyLoss()
        loss_cal = loss(output, torch.LongTensor([y]).cuda())    
        loss_cal.backward()
        x_grad = alpha * torch.sign(img_variable.grad.data)
        if not targeted:
            x_grad *= -1
        adv_temp = img_variable.data - x_grad
        total_grad = adv_temp - image_tensor
        total_grad = torch.clamp(total_grad, -eps, eps)
        x_adv = image_tensor + total_grad
        img_variable.data = x_adv
        img_variable.grad.zero_()
    return img_variable.detach()


def deepfool(image, net, num_classes=10, overshoot=0.02,eps= 0.2 , max_iter=50,min_acc=0.8):

    """
       :param image: Image of size HxWx3
       :param net: network (input: images, output: values of activation **BEFORE** softmax).
       :param num_classes: num_classes (limits the number of classes to test against, by default = 10)
       :param overshoot: used as a termination criterion to prevent vanishing updates (default = 0.02).
       :param max_iter: maximum number of iterations for deepfool (default = 50)
       :return: minimal perturbation that fools the classifier, number of iterations that it required, new estimated_label and perturbed image
    """
    is_cuda = torch.cuda.is_available()

    if is_cuda:
        image = image.cuda()
        net = net.cuda()


    f_image = net.forward(Variable(image[None, :, :, :], requires_grad=True)).data.cpu().numpy().flatten()
    I = (np.array(f_image)).flatten().argsort()[::-1]

    I = I[0:num_classes]
    label = I[0]

    input_shape = image.cpu().numpy().shape
    pert_image = copy.deepcopy(image)
    w = np.zeros(input_shape)
    r_tot = np.zeros(input_shape)

    loop_i = 0

    x = Variable(pert_image[None, :], requires_grad=True)
    fs = net.forward(x)
    fs_list = [fs[0,I[k]] for k in range(num_classes)]
    k_i = label

    while (k_i == label or AttackProb <min_acc) and (loop_i < max_iter):

        pert = np.inf
        fs[0, I[0]].backward(retain_graph=True)
        grad_orig = x.grad.data.cpu().numpy().copy()

        for k in range(1, num_classes):
            zero_gradients(x)

            fs[0, I[k]].backward(retain_graph=True)
            cur_grad = x.grad.data.cpu().numpy().copy()

            # set new w_k and new f_k
            w_k = cur_grad - grad_orig
            f_k = (fs[0, I[k]] - fs[0, I[0]]).data.cpu().numpy()

            pert_k = abs(f_k)/np.linalg.norm(w_k.flatten())

            # determine which w_k to use
            if pert_k < pert:
                pert = pert_k
                w = w_k

        # compute r_i and r_tot
        # Added 1e-4 for numerical stability
        r_i =  (pert+1e-4) * w / np.linalg.norm(w)
        r_tot = np.float32(r_tot + r_i)
        
        r_tot = np.clip(r_tot,-eps,eps)

        if is_cuda:
            pert_image = image + (1+overshoot)*torch.from_numpy(r_tot).cuda()
        else:
            pert_image = image + (1+overshoot)*torch.from_numpy(r_tot)

        x = Variable(pert_image, requires_grad=True)
        fs = net.forward(x)
        k_i = np.argmax(fs.data.cpu().numpy().flatten())
        AttackProb = F.softmax(fs,-1)[0].max(dim=0)[0]

        loop_i += 1

    r_tot = (1+overshoot)*r_tot

    return pert_image, k_i, AttackProb,loop_i 
def patch(model,image,AdvLabel=None,max_iter=100,min_acc=0.8):

    logits = model(image)
    OrigLabel = F.softmax(logits,-1)[0].max(dim=0)[1]
    OrigProb = F.softmax(logits,-1)[0].max(dim=0)[0]

    #attack
    Patch=nn.Parameter(torch.rand(5, 5), requires_grad=True)

    opt = optim.Adam([Patch],lr=10.0)
    
    
    if AdvLabel==None:
        AdvLabel=torch.randint(0, 9, (1,)).cuda()
        if AdvLabel==OrigLabel:
                AdvLabel+=1
    else:
        AdvLabel = torch.tensor([AdvLabel]).cuda()
    i=0
    max_val = image.max()
    min_val = image.min()
    PredLabel=OrigLabel
    AttackProb = 0
    CenterIndex = np.random.randint(5,32)
    while(PredLabel==OrigLabel or  AttackProb<min_acc) and (i<max_iter):
        Sampleimage=image.clone()
        opt.zero_grad()
        Sampleimage[:,:,CenterIndex-5:CenterIndex,CenterIndex-5:CenterIndex]+=Patch.cuda()
        logits = model(Sampleimage)
        PredLabel = F.softmax(logits,-1)[0].max(dim=0)[1]
        AttackProb = F.softmax(logits,-1)[0].max(dim=0)[0]
        loss = F.nll_loss(logits, AdvLabel)
        loss.backward(retain_graph=True)
        opt.step()
        #Sampleimage = torch.clamp(Sampleimage,min_val,max_val)
        i +=1
    return Sampleimage.detach(), PredLabel, AttackProb, i 

def recovery(model,x, device,attack=0,counter_attack=0 ,class_num=10, min_acc_att=0.7, eps=0.1,
                    att_nb_iter=8, att_max_iter=50, catt_nb_iter=3,max_attempt=10,plot=False):
    
    worked = False
    recovery = 0
    loss = torch.nn.CrossEntropyLoss()
    
    logits = model(x)
    predlabel = F.softmax(logits,-1)[0].max(dim=0)[1]
    predprob = F.softmax(logits,-1)[0].max(dim=0)[0]
    
    #print('original:', predprob.item())
    if predprob>min_acc_att:
        if attack == 0:
            adv_X = pgd(model, x, eps=eps, alpha=eps/att_nb_iter, num_iter=att_nb_iter, y = predlabel, targeted = False)
        elif attack == 1:
            adv_X = pgd_momen(model, x, eps=eps, alpha=eps/att_nb_iter, num_iter=att_nb_iter, y = predlabel, targeted = False)
        elif attack == 2:
            adv_X, attlabel, attprob, max_iter = deepfool(x[0], model,num_classes=class_num,max_iter=att_max_iter,eps=eps*5)
        elif attack == 3:
            attacks = ta.TPGD(model,eps=eps)
            adv_X = attacks(x,torch.tensor([predlabel]))
        elif attack == 4:
            #CW , PGDDLR, TPGD, EOTPGD
            attacks = ta.PGDDLR(model,eps=eps)
            adv_X = attacks(x,torch.tensor([predlabel]))
        elif attack == 5:
            target_ind = np.random.choice(range(class_num))  
            while target_ind == predlabel.item():
                target_ind = np.random.choice(range(class_num))

            attlabel = torch.tensor([target_ind],dtype=torch.long).to(device)

            adv_X, attlabel_, attprob, max_iter = patch(model,x,AdvLabel=attlabel,min_acc=min_acc_att,max_iter=att_max_iter)
            i = 0
            while (attprob<min_acc_att or predlabel==attlabel_ )and i<max_attempt:  
                i+=1
                adv_X, attlabel_, attprob, max_iter = patch(model,x,AdvLabel=attlabel,min_acc=min_acc_att,max_iter=att_max_iter)

        adv_logits = model(adv_X)
        attlabel = F.softmax(adv_logits,-1)[0].max(dim=0)[1]
        attprob = F.softmax(adv_logits,-1)[0].max(dim=0)[0]
        #print('attack: ',attprob.item())
        if attlabel!=predlabel and attprob>min_acc_att:
            worked = True
            losses = np.zeros(class_num)
            if counter_attack == 0:

                for i in range(class_num):
                    if i != attlabel:
                        target = torch.tensor([i]).to(device)
                        cont_adv_X =pgd(model, adv_X, eps=eps, alpha=eps/catt_nb_iter, num_iter=catt_nb_iter, y = target, targeted = True)
                        output = model(cont_adv_X)
                        losses[i] = loss(output, target)
            elif counter_attack == 1:
                for i in range(class_num):
                    if i != attlabel:
                        target = torch.tensor([i]).to(device)
                        cont_adv_X =pgd_momen(model, adv_X, eps=eps, alpha=eps/catt_nb_iter, num_iter=catt_nb_iter, y = target, targeted = True)
                        output = model(cont_adv_X)
                        losses[i] = loss(output, target)

            losses[attlabel] = losses.max()
            rec_label = losses.argmin()
            #print(losses)
            if rec_label == predlabel.item():
                recovery = 1
            else:
                recovery = 0

            if plot == True:
                attacks = ['PGD', 'MPGD','DeepFool' ,'TPGD', 'PGDDLR'] 
                print('The current attack is ', attacks[attack])
                fig, ax = plt.subplots(1,2)
                x = x.cpu().numpy()[0]
                x = np.moveaxis(x,0,-1)
                x = 0.5 * (x + 1)
                x = np.clip(x,0,1)
                ax[0].imshow(x)
                ad_x = adv_X.cpu().numpy()[0]
                ad_x = np.moveaxis(ad_x,0,-1)
                ad_x = 0.5 * (ad_x + 1)
                ad_x = np.clip(ad_x,0,1)
                ax[1].imshow(ad_x)
                title = fig.suptitle("Original class: {}, Probability: {}, Attack class: {}:, Probability: {}, recovered class: {}"
                                     .format(predlabel.item(),
                                             round(predprob.item(),2),
                                             attlabel,
                                             round(attprob.item(),2),
                                             rec_label.item()))
                plt.setp(title, color=('g' if rec_label == predlabel.item() else 'r'))
                plt.axis('off')
                plt.show()
                    
    return worked , recovery

    
def recovery_report(model,ds, device, class_num=10, min_acc_att=0.7, 
                          att_nb_iter=11, att_max_iter=50,eps=0.1, catt_nb_iter=3,max_attempt=5,batch_nb = 10,plot=False):
    '''
    eps: epsilon
    attacks ['PGD',
            'MomentumPGD',
            'DeepFool',
            'Patch',
            'TPGD',
            'PGDDLR'] 
    class_num: number of classes in the dataset
    min_acc_att: minimum accracy from the attack
    att_nb_iter: number of iterations for the attack
    att_max_iter: maximum number of iterations for the attack
    catt_nb_iter: number of iterations for the counter attack
    
    return: Accuracy score
    '''
    confusion_matrix = np.zeros((5,2))
    
    for i in tqdm(range(batch_nb)): 
        for i in range(5):
            for j in range(2):

                worked = False 
                c=0
                while worked==False:                            
                    ind = np.random.choice(range(len(ds)),1)[0]
                    x = ds[ind][0].to(device)
                    x = x.unsqueeze(0)
                    worked, succsess = recovery(model,x, device,attack=i,
                                                counter_attack=j ,class_num=class_num, min_acc_att=min_acc_att, 
                                                att_nb_iter=att_nb_iter, att_max_iter=att_max_iter, catt_nb_iter=catt_nb_iter,
                                                max_attempt=max_attempt,eps=eps,
                                               plot=plot)
                    c+=1
                if plot:
                    print('number of trials is ',c)
                    print('')
                    print('')
                        
                confusion_matrix[i][j] += succsess

        
    return confusion_matrix/batch_nb




torch.manual_seed(0)
download = False

img_transform = transforms.Compose([transforms.ToTensor(),
                                    transforms.Normalize((0.4914, 0.4822, 0.4465), 
                                                         (0.2023, 0.1994, 0.2010))])

train_dataset = datasets.CIFAR10('./data',download=download, transform=img_transform)
test_dataset = datasets.CIFAR10('./data',download=download,train=False, transform=img_transform)

model = resnet18(pretrained=True)
model.cuda()
model.eval()
conf = recovery_report(model,test_dataset, device,class_num=10,att_nb_iter=3, catt_nb_iter=2,min_acc_att=0.8,
                       batch_nb=1000,eps=0.03,plot=False)

df_cm = pd.DataFrame(conf, range(5), range(2))
plt.figure(figsize=(30,18))
sn.set(font_scale=2.2) # for label size
sn.heatmap(df_cm, annot=True, xticklabels=['PGD','MPGD' ] ,yticklabels=['PGD','MPGD','Deepfool','TPGD','PGDDLR'],annot_kws={"size": 46}) # font size
plt.xlabel('Recovery Counter attacks',fontsize=50)
plt.ylabel('Adversarial attacks',fontsize=50)
p.dump(conf,open( "cifar10_1000_inception_5_attacks.p", "wb" ))
plt.savefig( "cifar10_1000_inception_5_attacks")
plt.show()

print(df_cm.mean())


