import torch
import glob
import os
import tqdm
import numpy as np
import cv2

input_shape = (224, 224)
n_classes=2
cuda=True
from preprocess import tensorize_image
TEST_DIR='../data/p1_test/img'
MASK_DIR='../data/p1_predict/img'
model_path='colab_model_test.pt'
model=torch.load(model_path)

if cuda:
  model=model.cuda()
model.eval()
test=os.listdir(TEST_DIR)
#test=glob.glob('../data/test_data/*')
#test_mask=glob.glob('../data/test_mask_data/*')

'''y=0
for test_img in tqdm.tqdm(test):
    image_path=os.path.join(TEST_DIR,test_img)
    tensorized_test_image = tensorize_image([image_path], input_shape, cuda)
    output=model(tensorized_test_image)
    mask_name_without_ex = test_img.split('.')[0]

    # Access required folders
    mask_path     = os.path.join(MASK_DIR, mask_name_without_ex+'.png')
    torchvision.utils.save_image(output,mask_path)
    y+=1
    #('../data/predicted_image/'+str(y)+'.png')
    '''
    
test_data = glob.glob(os.path.join(TEST_DIR, '*'))
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
        predict_path=predict_name.replace('p1_test', 'p1_predict')
        cv2.imwrite(predict_path,opac_image.astype(np.uint8))

predict(test_data)