import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.autograd import Variable
from torch.utils.data import Dataset, DataLoader
from torch.autograd.gradcheck import zero_gradients
from torch.autograd import Variable
from torchvision import datasets, transforms
import torchvision.models as models
import torchattacks as ta

import matplotlib.pyplot as plt
import numpy as np
import pickle as p
import seaborn as sn
import pandas as pd
import copy
from tqdm import tqdm

import json
import time
import random
import os
from PIL import Image

torch.manual_seed(0)
download = False
batch_size=32
class_number = 10

IMG_SIZE = (299,299)

data_path='./data/tiny-imagenet-200/train'


class_idx = json.load(open("imagenet_class_index.json"))
idx2label = [class_idx[str(k)][1] for k in range(len(class_idx))]


class ImageNetDataset(Dataset):
    def __init__(self, data_path, is_train, train_split = 0.99, random_seed = 42, target_transform = None, num_classes = None):
        super(ImageNetDataset, self).__init__()
        self.data_path = data_path

        self.is_classes_limited = False

        if num_classes != None:
            self.is_classes_limited = True
            self.num_classes = num_classes

        self.classes = []
        class_idx = 0
        
        m = random.sample(os.listdir(data_path),num_classes)
        
        m = np.sort(m)

        self.indexs = []
        
        ImgNet_idx = json.load(open("imagenet_class_index.json"))

        for k in m:
            for i in ImgNet_idx:    
                if ImgNet_idx[i][0]== k:
                    self.indexs.append(int(i))
                    break

        for class_name in m:
            self.classes.append(
               dict(
                   class_idx = class_idx,
                   class_name = class_name,
               ))
            class_idx += 1

            if self.is_classes_limited:
                if class_idx == self.num_classes:
                    break

        if not self.is_classes_limited:
            self.num_classes = len(self.classes)

        self.image_list = []
        for cls in self.classes:
            class_path = data_path +'/' +cls['class_name']+ '/images'
            for image_name in os.listdir(class_path):
                image_path = class_path +'/' + image_name
                self.image_list.append(dict(
                    cls = cls,
                    image_path = image_path,
                    image_name = image_name,
                ))

        self.img_idxes = np.arange(0,len(self.image_list))

        np.random.seed(random_seed)
        np.random.shuffle(self.img_idxes)

        last_train_sample = int(len(self.img_idxes) * train_split)
        if is_train:
            self.img_idxes = self.img_idxes[:last_train_sample]
        else:
            self.img_idxes = self.img_idxes[last_train_sample:]

    def __len__(self):
        return len(self.img_idxes)

    def __getitem__(self, index):

        img_idx = self.img_idxes[index]
        img_info = self.image_list[img_idx]

        img = Image.open(img_info['image_path'])

        if img.mode == 'L':
            tr = transforms.Grayscale(num_output_channels=3)
            img = tr(img)

        tr = transforms.Resize(IMG_SIZE)
        img = tr(img)

        tr = transforms.ToTensor()
        img = tr(img)
        if (img.shape[0] != 3):
            img = img[0:3]
        normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                 std=[0.229, 0.224, 0.225])

        img = normalize(img)
        
        return [img, img_info['cls']['class_idx']]
        #return dict(image = img, cls = img_info['cls']['class_idx'], class_name = img_info['cls']['class_name'])

    def get_number_of_classes(self):
        return self.num_classes

    def get_number_of_samples(self):
        return self.__len__()

    def get_class_names(self):
        return [cls['class_name'] for cls in self.classes]

    def get_class_name(self, class_idx):
        return self.classes[class_idx]['class_name']


def get_imagenet_datasets(data_path, num_classes = None):

    random_seed = int(time.time())

    dataset_train = ImageNetDataset(data_path,is_train = True, random_seed=random_seed, num_classes = num_classes)
    #dataset_test = ImageNetDataset(data_path, is_train = False, random_seed=random_seed, num_classes = num_classes)

    return dataset_train


dataset_train = get_imagenet_datasets(data_path,num_classes=10)

#print(f"Number of train samplest {dataset_train.__len__()}")

#data_loader_train = DataLoader(dataset_train, batch_size, shuffle = True)

indx = dataset_train.indexs 

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


def recovery(model,x, indx, device,attack=0,counter_attack=0 ,class_num=10, min_acc_att=0.7,eps=0.1, 
                    att_nb_iter=11, att_max_iter=50, catt_nb_iter=3,max_attempt=10,batch_nb = 10,plot=False):
    
    worked = False
    recovery = 0
    loss = torch.nn.CrossEntropyLoss()
    
    logits = model(x)
    predlabel = F.softmax(logits,-1)[0].max(dim=0)[1]
    predprob = F.softmax(logits,-1)[0].max(dim=0)[0]
    
    if predprob>min_acc_att and predlabel.item() in indx:
        
        target_ind = np.random.choice(range(10))  
        while indx[target_ind] == predlabel.item():
            target_ind = np.random.choice(range(10))

        if attack == 0:
            target = torch.tensor([indx[target_ind]]).to(device)
            adv_X = pgd(model, x, eps=eps, alpha=eps/att_nb_iter, num_iter=att_nb_iter, y = target,targeted = True)
        elif attack == 1:
            target = torch.tensor([indx[target_ind]]).to(device)
            adv_X = pgd_momen(model, x, eps=eps, alpha=eps/att_nb_iter, num_iter=att_nb_iter, y = target, targeted = True)
        elif attack == 2:
            attacks = ta.TPGD(model,eps=eps/att_nb_iter)
            attacks.targeted = True
            target = torch.tensor([indx[target_ind]]).to(device)
            adv_X = attacks(x,target)
        elif attack == 3:
            #CW , PGDDLR, TPGD, EOTPGD
            attacks = ta.PGDDLR(model,eps=eps)
            attacks.targeted = True
            target = torch.tensor([indx[target_ind]]).to(device)
            adv_X = attacks(x,target)
            
        adv_logits = model(adv_X)
        attlabel = F.softmax(adv_logits,-1)[0].max(dim=0)[1]
        attprob = F.softmax(adv_logits,-1)[0].max(dim=0)[0]
        rec_label = -1
        attlabel_indx = -1
        
        if attlabel!=predlabel and attprob>min_acc_att:
            worked = True
            losses = np.zeros(class_num)
            if counter_attack == 0:
                for i in range(class_num):
                    if indx[i] != attlabel:
                        target = torch.tensor([indx[i]]).to(device)
                        cont_adv_X =pgd(model, adv_X, eps=eps, alpha=eps/catt_nb_iter, num_iter=catt_nb_iter, y = target, targeted = True)
                        output = model(cont_adv_X)
                        losses[i] = loss(output, target)
                    else:
                        attlabel_indx = i

            elif counter_attack == 1:
                for i in range(class_num):
                    if indx[i]  != attlabel:
                        target = torch.tensor([indx[i]]).to(device)
                        cont_adv_X =pgd_momen(model, adv_X, eps=eps, alpha=eps/catt_nb_iter, num_iter=catt_nb_iter, y = target, targeted = True)
                        output = model(cont_adv_X)
                        losses[i] = loss(output, target)
                    else:
                        attlabel_indx = i


            losses[attlabel_indx] = losses.max()
            rec_label = losses.argmin()
            #print(losses)
            if indx[rec_label] == predlabel.item():
                recovery = 1
            else:
                recovery = 0

            if plot == True: 
                attacks = ['PGD', 'MPGD', 'TPGD', 'PGDDLR'] 
                print('The current attack is ', attacks[attack])
                fig, ax = plt.subplots(1,2)
                x = x.cpu().numpy()[0]
                x = np.moveaxis(x,0,-1)
                x = 0.5 * (x + 1)
                x = np.clip(x,0,1)
                ax[0].imshow(x)
                ax[0].set_axis_off()
                ad_x = adv_X.cpu().numpy()[0]
                ad_x = np.moveaxis(ad_x,0,-1)
                ad_x = 0.5 * (ad_x + 1)
                ad_x = np.clip(ad_x,0,1)
                ax[1].set_axis_off()
                ax[1].imshow(ad_x)
                title = fig.suptitle("Original class: {}, Probability: {}, Attack class: {}:, Probability: {}, recovered class: {}"
                                     .format(predlabel.item(),
                                             round(predprob.item(),2),
                                             attlabel,
                                             round(attprob.item(),2),
                                             indx[rec_label]))
                plt.setp(title, color=('g' if indx[rec_label] == predlabel.item() else 'r'))
                plt.show()

    return worked , recovery

    
def recovery_report(model,ds,indx, device, class_num=10, min_acc_att=0.7, eps=0.1,
                          att_nb_iter=11, att_max_iter=50, catt_nb_iter=3,max_attempt=10,batch_nb = 10,plot=False):
    '''
    eps: epsilon
    attacks ['LinfPGD',
            'LinfMomentumPGD',
            'TPGD',
            'PGDDLR'] 
    class_num: number of classes in the dataset
    min_acc_att: minimum accracy from the attack
    att_nb_iter: number of iterations for the attack
    att_max_iter: maximum number of iterations for the attack
    catt_nb_iter: number of iterations for the counter attack
    
    return: confusion matrix
    '''
    confusion_matrix = np.zeros((4,2))
    attempts = 0
    
    for i in tqdm(range(batch_nb)):
        for i in range(4):
            for j in range(2):
                worked = False 
                c=0
                while worked==False:                            
                    ind = np.random.choice(range(len(ds)),1)[0]
                    x = ds[ind][0].to(device)
                    x = x.unsqueeze(0)
                    worked, succsess = recovery(model,x, indx,device,attack=i,
                                                counter_attack=j ,class_num=10, min_acc_att=0.7, 
                                                att_nb_iter=11, att_max_iter=50, catt_nb_iter=3,
                                                max_attempt=10,batch_nb = 10,plot=plot)
                    c+=1
                if plot:
                    print('number of trials is ',c)
                    print('')
                    print('')
                confusion_matrix[i][j] += succsess
        
        
    return confusion_matrix/batch_nb

model = models.inception_v3(pretrained = True,aux_logits =False)
model.cuda()
model.eval()

samples = 10

conf = np.zeros((samples,4,2))

for i in range(samples):
    
    dataset_train = get_imagenet_datasets(data_path,num_classes=10)

    indx = dataset_train.indexs 

    conf[i] = recovery_report(model,dataset_train, indx, device, class_num=10, min_acc_att=0.6, eps=0.2,
                              att_nb_iter=10, att_max_iter=50, catt_nb_iter=2,max_attempt=10,batch_nb = 100,plot=False)

confs = conf.mean(0)    
df_cm = pd.DataFrame(confs, range(4), range(2))
plt.figure(figsize=(30,18))
sn.set(font_scale=2.2) # for label size
sn.heatmap(df_cm, annot=True, xticklabels=['PGD','MPGD' ] ,yticklabels=['PGD','MPGD','TPGD','PGDDLR'],annot_kws={"size": 46}) # font size
plt.xlabel('Recovery Counter attacks',fontsize=50)
plt.ylabel('Adversarial attacks',fontsize=50)
p.dump(conf,open( "ImageNet10_10x100_resnet18.p", "wb" ))
plt.savefig( "ImageNet10_10x100_resnet18")
plt.show()

print(confs.mean())