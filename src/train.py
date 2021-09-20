# author: ayberk kose as mralbino

from model import UNet
from preprocess import tensorize_image, tensorize_mask, image_mask_check
import os
import glob
import numpy as np
import torch.nn as nn
import torch.optim as optim
import tqdm
import torch,gc
import cv2
from matplotlib import pyplot as plt
import matplotlib.ticker as mticker

print("GPU Name: "+torch.cuda.get_device_name(0))
print("Cuda Available: "+str(torch.cuda.is_available()))
gc.collect()
torch.cuda.empty_cache()
######### PARAMETERS ##########
valid_size = 0.1
test_size  = 0.04
batch_size = 16
epochs = 30
cuda = True
input_shape = (224, 224)
n_classes = 2
###############################

######### DIRECTORIES #########
SRC_DIR = os.getcwd()
ROOT_DIR = os.path.join(SRC_DIR, '..')
DATA_DIR = os.path.join(ROOT_DIR, 'data')
IMAGE_DIR = os.path.join(DATA_DIR, '/content/sample_data/images')
MASK_DIR = os.path.join(DATA_DIR, '/content/sample_data/masks')
AUG_IMAGE=os.path.join(DATA_DIR,'/content/sample_data/aug_photo')
AUG_MASK=os.path.join(DATA_DIR,'/content/sample_data/aug_masks')
AUG2_IMAGE=os.path.join(DATA_DIR,'/content/sample_data/aug2_photos')
AUG2_MASK=os.path.join(DATA_DIR,'/content/sample_data/aug2_masks')
###############################
aug2_count=len(os.listdir(AUG2_IMAGE))
print("Aug2 Size:"+str(aug2_count))
# PREPARE IMAGE , MASK ,AUGMENTATION LISTS
image_path_list = glob.glob(os.path.join(IMAGE_DIR, '*'))
image_path_list.sort()

mask_path_list = glob.glob(os.path.join(MASK_DIR, '*'))
mask_path_list.sort()

aug_path_list = glob.glob(os.path.join(AUG_IMAGE, '*'))
aug_path_list.sort()
aug_mask_path_list = glob.glob(os.path.join(AUG_MASK, '*'))
aug_mask_path_list.sort()

aug2_path_list = glob.glob(os.path.join(AUG2_IMAGE, '*'))
aug2_path_list.sort()
aug2_mask_path_list = glob.glob(os.path.join(AUG2_MASK, '*'))
aug2_mask_path_list.sort()
# PREPARE IMAGE , MASK ,AUGMENTATION LISTS

# DATA CHECK
image_mask_check(image_path_list, mask_path_list)
image_mask_check(aug_path_list, aug_mask_path_list)
image_mask_check(aug2_path_list,aug2_mask_path_list)
# SHUFFLE INDICES
indices = np.random.permutation(len(image_path_list))
# DEFINE TEST AND VALID INDICES
test_ind  = int(len(indices) * test_size)
valid_ind = int(test_ind + len(indices) * valid_size)

# SLICE TEST DATASET FROM THE WHOLE DATASET
test_input_path_list = image_path_list[:test_ind]
test_label_path_list = mask_path_list[:test_ind]

# SLICE VALID DATASET FROM THE WHOLE DATASET
valid_input_path_list = image_path_list[test_ind:valid_ind]
valid_label_path_list = mask_path_list[test_ind:valid_ind]

# SLICE TRAIN DATASET FROM THE WHOLE DATASET
train_input_path_list = image_path_list[valid_ind:]
train_label_path_list = mask_path_list[valid_ind:]

print("Train Size:"+str(len(train_input_path_list))+"+ Aug Size:"+str(len(aug_path_list))+"+ Aug2 Size:"+str(aug2_count))
train_input_path_list=train_input_path_list+aug_path_list+aug2_path_list
train_label_path_list=train_label_path_list+aug_mask_path_list+aug2_mask_path_list

with open("losses.txt", "a") as file_object:
    file_object.write("Batch Size:{} Epoch Size:{}  Image Count:{}".format(batch_size,epochs,len(train_input_path_list)))
    file_object.write("\n")
# DEFINE STEPS PER EPOCH
steps_per_epoch = len(train_input_path_list)//batch_size

# CALL MODEL
#             model = FoInternNet(input_size=input_shape, n_classes=2)
model = UNet(n_channels=3, n_classes=2, bilinear=True)
# DEFINE LOSS FUNCTION AND OPTIMIZER
#                 criterion = nn.BCELoss()
criterion =  nn.BCELoss()

# try new below  - optimizer = optim.SGD(model.parameters(), lr=0.001, momentum=0.9)
optimizer = optim.Adam(model.parameters(), lr=0.0003)
#optimizer = optim.SGD(model.parameters(), lr=0.001, momentum=0.9)
# IF CUDA IS USED, IMPORT THE MODEL INTO CUDA
if cuda:
    model = model.cuda()
val_losses=[]
train_losses=[]

test_counter=0

# TRAINING THE NEURAL NETWORK
for epoch in range(epochs):
    running_loss = 0
    #In each epoch, images and masks are mixed randomly in order not to output images sequentially.
    pair_IM=list(zip(train_input_path_list,train_label_path_list))
    np.random.shuffle(pair_IM)
    unzipped_object=zip(*pair_IM)
    zipped_list=list(unzipped_object)
    train_input_path_list=list(zipped_list[0])
    train_label_path_list=list(zipped_list[1])
    test_counter+=1
    if(test_counter==1):
        print(train_input_path_list[:20])
        print("----")
        print(train_label_path_list[:20])
    for ind in tqdm.tqdm(range(steps_per_epoch)):
        batch_input_path_list = train_input_path_list[batch_size*ind:batch_size*(ind+1)]
        #train_input_path_list [0: 4] gets first 4 elements on first entry
        #in the second loop train_input_list [4: 8] gets the second 4 elements
        #element advances each time until batch_size
        batch_label_path_list = train_label_path_list[batch_size*ind:batch_size*(ind+1)]
        batch_input = tensorize_image(batch_input_path_list, input_shape, cuda)
        batch_label = tensorize_mask(batch_label_path_list, input_shape, n_classes, cuda)
        #Our data that we will insert into the model in the preprocess section is prepared by entering the parameters.
        
        optimizer.zero_grad()#gresets the radian otherwise accumulation occurs on each iteration
        # Manually reset gradients after updating Weights

        outputs = model(batch_input) # Give the model batch_input as a parameter and assign the resulting output to the variable.
        

        # Forward passes the input data
        loss = criterion(outputs, batch_label)
        loss.backward()# Calculates the gradient, how much each parameter needs to be updated
        optimizer.step()# Updates each parameter according to the gradient

        running_loss += loss.item()# loss.item () takes the scalar value held in loss.

        #validation 
        if ind == steps_per_epoch-1:
            
            train_losses.append(running_loss)
            print('training loss on epoch {}: {}'.format(epoch, running_loss))
            val_loss = 0
            model.eval()
            with torch.no_grad():
                for (valid_input_path, valid_label_path) in zip(valid_input_path_list, valid_label_path_list):
                    batch_input = tensorize_image([valid_input_path], input_shape, cuda)
                    batch_label = tensorize_mask([valid_label_path], input_shape, n_classes, cuda)
                    outputs = model(batch_input)
                    loss = criterion(outputs, batch_label)
                    val_loss += loss.item()
                
                    break
            val_losses.append(val_loss)
            print('validation loss on epoch {}: {}'.format(epoch, val_loss))
            torch.save(model, 'colab_unet_adam_aug_30_last.pt')
            print("Model Saved!")
            model.train()
    with open("losses.txt", "a") as file_object:
    # Append 'hello' at the end of file
        file_object.write("{}.Epoch => Train Loss: {} Validation Loss:{}".format(epoch,running_loss,val_loss))
        file_object.write("\n")
with open("losses.txt", "a") as file_object:
    file_object.write("\n")

best_model = torch.load('colab_unet_adam_aug_30_last.pt')

test_data_path='../data/test_data'
test_data = glob.glob(os.path.join(test_data_path, '*'))
def predict(test_input_path_list):

    for i in tqdm.tqdm(range(len(test_input_path_list))):
        batch_test = test_input_path_list[i:i+1]
        test_input = tensorize_image(batch_test, input_shape, cuda)
        outs = model(test_input)
        out=torch.argmax(outs,axis=1)
        out_cpu = out.cpu()
        outputs_list=out_cpu.detach().numpy()
        mask=np.squeeze(outputs_list,axis=0)
            
            
        img=cv2.imread(batch_test[0])
        mg=cv2.resize(img,(224,224))
        mask_ind   = mask == 1
        cpy_img  = mg.copy()
        mg[mask==0 ,:] = (255, 0, 125)
        opac_image=(mg/2+cpy_img/2).astype(np.uint8)
        predict_name=batch_test[0]
        predict_path=predict_name.replace('images', 'predicted_masked')
        cv2.imwrite(predict_path,opac_image.astype(np.uint8))

predict(test_input_path_list)
 
def draw_graph(val_losses,train_losses,epochs):
    norm_validation = [float(i)/sum(val_losses) for i in val_losses]
    norm_train = [float(i)/sum(train_losses) for i in train_losses]
    epoch_numbers=list(range(1,epochs+1,1))
    plt.figure(figsize=(12,6))
    plt.subplot(2, 2, 1)
    plt.plot(epoch_numbers,norm_validation,color="red") 
    plt.gca().xaxis.set_major_locator(mticker.MultipleLocator(1))
    plt.title('Validation losses')
    plt.subplot(2, 2, 2)
    plt.plot(epoch_numbers,norm_train,color="blue")
    plt.gca().xaxis.set_major_locator(mticker.MultipleLocator(1))
    plt.title('Train losses')
    plt.subplot(2, 1, 2)
    plt.plot(epoch_numbers,norm_validation, 'r-',color="red")
    plt.plot(epoch_numbers,norm_train, 'r-',color="blue")
    plt.legend(['w=1','w=2'])
    plt.title('Train and Validation Losses')
    plt.gca().xaxis.set_major_locator(mticker.MultipleLocator(1))
    
    
    plt.show()

draw_graph(val_losses,train_losses,epochs)