import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.autograd import Variable
from torch.utils.data import Dataset, DataLoader
from torch.autograd.gradcheck import zero_gradients
from torch.autograd import Variable
from torchvision import datasets, transforms
import torchattacks as ta
import matplotlib.pyplot as plt
import numpy as np
import pickle as p
import seaborn as sn
import pandas as pd
import copy
from tqdm import tqdm


torch.manual_seed(0)
download = False
batch_size=32

# download and transform test dataset

test_dataset = datasets.MNIST('../../mnist_data', 
                             download=download, 
                             train=False,
                             transform=transforms.Compose([transforms.ToTensor(),
                                                           transforms.Normalize((0.5,), (0.5,)) ]))
test_loader = torch.utils.data.DataLoader(test_dataset,
                                          batch_size=batch_size, 
                                          shuffle=True)

train_dataset = datasets.MNIST('../../mnist_data', 
                             download=download, 
                             train=True,
                             transform=transforms.Compose([transforms.ToTensor(),
                                                           transforms.Normalize((0.5,), (0.5,)) ]))


train_loader = torch.utils.data.DataLoader(datasets.MNIST('../../mnist_data', 
                                         download=download, 
                                         train=True,
                                         transform=transforms.Compose([transforms.ToTensor(),
                                                                       transforms.Normalize((0.1307,), (0.3081,)) ])),
                                          batch_size=batch_size, 
                                          shuffle=True)

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")





class CNNClassifier(nn.Module):
    def __init__(self):
        super(CNNClassifier, self).__init__()
        self.conv1 = nn.Conv2d(1, 6, kernel_size=5, padding=2)
        self.conv2 = nn.Conv2d(6, 16, kernel_size=5)
        self.fc1 = nn.Linear(16*5*5, 120)
        self.fc2 = nn.Linear(120, 84)
        self.fc3 = nn.Linear(84, 10)
      
           
    def forward(self, x):
                x=self.conv1(x)
                x=F.max_pool2d(x, 2 ,2)
                x = F.relu(x)
                
                x=self.conv2(x)
                x=F.max_pool2d(x, 2 ,2)
                x = F.relu(x)
                
                x= x.view(-1,16*5*5)
                
                x=self.fc1(x)
                x = F.relu(x)
                
                x=self.fc2(x)
                x = F.relu(x)
                
                x=self.fc3(x)
                
                
                return x
            
            
def epoch(loader, model, opt=None):
    total_loss, total_err = 0.,0.
    for X,y in loader:
        X,y = X.to(device), y.to(device)
        yp = model(X)
        loss = nn.CrossEntropyLoss()(yp,y)
        if opt:
            opt.zero_grad()
            loss.backward()
            opt.step()
        
        total_err += (yp.max(dim=1)[1] != y).double().sum().item()
        total_loss += loss.item() * X.shape[0]
    return 1 - (total_err / len(loader.dataset)), total_loss / len(loader.dataset)


model = CNNClassifier()
model.cuda()
opt = optim.SGD(model.parameters(), lr=1e-1)



for _ in range(5):
    train_err, train_loss = epoch(train_loader, model, opt)
    test_err, test_loss = epoch(test_loader, model)
    print(*("{:.6f}".format(i) for i in (train_err, train_loss, test_err, test_loss)), sep="\t")


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
            attacks = ta.TPGD(model,eps=eps,steps=att_nb_iter,alpha=eps)
            adv_X = attacks(x,torch.tensor([predlabel]))
        elif attack == 4:
            #CW , PGDDLR, TPGD, EOTPGD
            attacks = ta.PGDDLR(model,eps=eps,steps=att_nb_iter,alpha=eps)
            adv_X = attacks(x,torch.tensor([predlabel]))

        adv_logits = model(adv_X)
        attlabel = F.softmax(adv_logits,-1)[0].max(dim=0)[1]
        attprob = F.softmax(adv_logits,-1)[0].max(dim=0)[0]
        rec_label=torch.tensor([-1])
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
            attacks=['PGD', 'MPGD', 'DeepFool','TPGD', 'PGDDLR'] 
            print(attacks[attack])
            fig, ax = plt.subplots(1,2)
            x = x.cpu().numpy()[0][0]
            x = 0.5 * (x + 1)
            x = np.clip(x,0,1)
            ax[0].imshow(x)
            ad_x = adv_X.cpu().numpy()[0][0]
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
                c=0
                worked = False 
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

conf = recovery_report(model.eval(),train_dataset, device,class_num=10,att_nb_iter=10, catt_nb_iter=2,min_acc_att=0.7,
                       batch_nb=1000,eps=0.3,plot=False)

df_cm = pd.DataFrame(conf, range(5), range(2))
plt.figure(figsize=(10,7))
sn.set(font_scale=1.4) # for label size
sn.heatmap(df_cm, annot=True, annot_kws={"size": 16}) # font size
#plt.savefig('mnist')
plt.show()

print(conf.mean())