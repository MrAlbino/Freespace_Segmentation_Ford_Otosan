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

1. ## **Expected Result:**

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

1. ## **Colorize and Test Masks**
   >This is the explanation of [mask_on_image.py](/src/mask_on_image.py)

   We are ready to test our masks now. In this part we will apply our mask on raw image and then applying some colors too. So we can check, is our mask ready to go ?
   
   **See transaction visualization below, left-to-right:**

   ![image](images_for_readme/mask_on_image.png)

