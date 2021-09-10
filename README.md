# **Build**
1. ## **Installation**

    In order to clone the repository to your local computer open command prompt or terminal and run the command given below.
    ```bash
    git clone https://github.com/MrAlbino/Semantic_Segmentation_Ford_Otosan.git
    ```
&nbsp;

2. ## **Creating Virtual Environment (Optional)**
    > You have to be in the root directory for these commands.
    * ### **For Windows**:
      * Create Virtual Environment:

        ```bash
        python -m venv venv
        ``` 
       * Activate Virtual Environment:

            ```bash
            venv\Scripts\activate.bat
            ```
        * Deactivate Virtual Environment:

            ```bash
            deactivate
            ```
    * ### **For Linux**:
        * Create Virtual Environment:

            ```bash
            python3 -m venv venv
            ``` 
       * Activate Virtual Environment:

            ```bash
            source venv/bin/activate
            ```
        * Deactivate Virtual Environment:

            ```bash
            deactivate
            ```
    > After all these processes the expected folder hierarcy can be seen below.

     * **Expected Hierarcy**:
         
        ![image](images_for_readme/folder_hierarcy.PNG)


&nbsp;

3. ## **Install Required Libraries**
   * Virtual environment must be activated before running this command.
  
        ```bash
        pip install -r requirements.txt
        ``` 
    &nbsp;
--- 
&nbsp;
# **Explanation**

1. ## **Purpose**
    Our main purpose in this project is to detecting driveable areas for autonomous vehicles.

&nbsp;

2. ## **Expected Result:**

   ![image](images_for_readme/result.png)
&nbsp;
3. ## **Create Mask with Json**
   
   >This is the explanation of [json2mask.py](/src/json2mask.py)

   In this part of the project, we will use the json files that created after the image labeling process to obtain masks. These files contain information of the freespace class exterior point location. You can see an example of the json file format below.
   

   ![image](images_for_readme/json.PNG)

    With those informations we can now draw the mask of the freespace.
     ```bash
     if obj['classTitle']=='Freespace':
            mask = cv2.fillPoly(mask, np.array([obj['points']['exterior']]), color=1)
     ```  
     The code block you can see above, determines which object  has "Freespace" as a classTitle. After that it draws mask using __fillPoly__ function from cv2 library.

     ### **Expected Mask Format:**

    ![image](images_for_readme/example_mask.png)
&nbsp;

3. ## **Colorize and Test Masks**
   >This is the explanation of [mask_on_image.py](/src/mask_on_image.py)

   We are ready to test our masks now. In this part we will apply our mask on raw image and then applying some colors too. So we can check, is our mask ready to go ?
   
   **See transaction visualization below, left-to-right:**

   ![image](images_for_readme/mask_on_image.png)

   As you can see our masks are working fine, we can continue.

4. ## **Preprocess**
   >This is the explanation of [preprocess.py](/src/preprocess.py)

   In this section we will use our masks and images as inputs to the model. Before that we have to convert these inputs to the tensor format because our model will be expecting tensor formatted inputs.
   ```bash
    def torchlike_data(data):
        n_channels = data.shape[2]
        torchlike_data=np.empty((n_channels,data.shape[0],data.shape[1]))
        for ch in range(n_channels):
            torchlike_data[ch] = data[:,:,ch]
        return torchlike_data
   ```
   ***torchlike_data(data)*** function takes an input(image) and returns torch like data. We use this function inside the ***tensorize_image()*** and ***tensorize_mask()*** methods.

   Inside the ***tensorize_mask()*** function, procedure works same as ***tensorize_image()*** but there is an addition named ***one_hot_encoder()*** method you can see below.

    ```bash
    def one_hot_encoder(data, n_class):
        encoded_data = np.zeros((*data.shape, n_class), dtype=np.int)

        encoded_labels = [[0,1], [1,0]]
        for lbl in range(n_class):

            encoded_label = encoded_labels[lbl]
            numerical_class_inds = data[:,:] == lbl
            encoded_data[numerical_class_inds] = encoded_label
        return encoded_data
    ```

    One hot encoding stands for classifying categorical data. One hot encoding transforms our categorical labels into vectors of ones and zeros. Length of these vectors depends on the number of categories, in this example we have 2 categories; ***driveable area***, ***undriveable area***. Elements of these vectors are zeros except for the element that corresponds to the __driveable are__, this element will be one (1).  

    There is another function named ___image_mask_check()___, it takes inputs images and masks path and checks if they match or not. If they don't match we can not use them.
    ```bash
    def image_mask_check(image_path_list, mask_path_list):
        for image_path, mask_path in zip(image_path_list, mask_path_list):
            image_name = image_path.split('/')[-1].split('.')[0]
            mask_name  = mask_path.split('/')[-1].split('.')[0]
            assert image_name == mask_name, "Image and mask name does not match {} - {}".format(image_name, mask_name)
    ```

6. ## **Model**

    >This is the explanation of [model.py](/src/model.py)
    
    In this project we will be using U-Net model. There are other models like FCN, Mask RCNN but U-Net is much faster to run so we will continue with it. Name of U-Net comes from the architecture itself, similar to 'U' shape you can see below.

    ![image](images_for_readme/unet.png)

7. ## **Train**
    >This is the explanation of [train.py](/src/train.py)

    Now we are ready to train our model. First we define our essential parameters for training.
    ```bash
    valid_size = 0.3
    test_size  = 0.1
    batch_size = 8
    epochs = 25
    cuda = True
    input_shape = (224, 224)
    n_classes = 2
    ```
    After that we define our directory paths.
    ```bash
    SRC_DIR = os.getcwd()
    ROOT_DIR = os.path.join(SRC_DIR, '..')
    DATA_DIR = os.path.join(ROOT_DIR, 'data')
    IMAGE_DIR = os.path.join(DATA_DIR, 'images')
    MASK_DIR = os.path.join(DATA_DIR, 'masks')
    AUG_IMAGE=os.path.join(DATA_DIR,'aug_photo')
    AUG_MASK=os.path.join(DATA_DIR,'aug_masks')
    ```
    Now we are ready to obtain our image paths into the lists.

    ```bash
    image_path_list = glob.glob(os.path.join(IMAGE_DIR, '*'))
    image_path_list.sort()

    mask_path_list = glob.glob(os.path.join(MASK_DIR, '*'))
    mask_path_list.sort()

    aug_path_list = glob.glob(os.path.join(AUG_IMAGE, '*'))
    aug_path_list.sort()

    aug_mask_path_list = glob.glob(os.path.join(AUG_MASK, '*'))
    aug_mask_path_list.sort()
    ```
    Before continuing we must check our masks and images, are they matching ?

    ```bash
    image_mask_check(image_path_list, mask_path_list)
    image_mask_check(aug_path_list, aug_mask_path_list)
    ```

    If there are no errors we can continue. Next step is slicing dataset into three parts; test,train,validation.

    ```bash
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
    ```

    Last step is to adding augmentated images into train dataset.

    ```bash
    aug_size=int(len(aug_mask_path_list)/2)
    train_input_path_list=aug_path_list[:aug_size]+train_input_path_list+aug_path_list[aug_size:]
    train_label_path_list=aug_mask_path_list[:aug_size]+train_label_path_list+aug_mask_path_list[aug_size:]
    ```

    Dataset slicing is done. Now we are going to call our model, define loss function and optimizer.

    ```bash
    model = UNet(n_channels=3, n_classes=2, bilinear=True)
    criterion =  nn.BCELoss()
    optimizer = optim.Adam(model.parameters(), lr=0.001)
    #optimizer = optim.SGD(model.parameters(), lr=0.001, momentum=0.9) #you can use this too
    ```
    
    I used CUDA in this project so the following code is necessary. At the beginning we assigned to cuda variable 'True' if you don't use CUDA set it to 'False'.
    ```bash
    if cuda:
        model = model.cuda()
    ```

8. ## **Augmentation**

    >This is the explanation of [augmentation.py](/src/augmentation.py)
    and [augmentation_mirror.py](/src/augmentation_mirror.py)








    

