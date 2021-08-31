# Build
1. ## Installation

    In order to clone the repository to your local computer open command prompt or terminal and run the command given below.
    ```bash
    git clone https://github.com/MrAlbino/Semantic_Segmentation_Ford_Otosan.git
    ```

2. ## Creating Virtual Environment (Optional)
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
         
        ![image](folder_hierarcy.png)

  
3. ## Install Required Libraries
   * Virtual environment must be activated before running this command.
  
        ```bash
        pip install -r requirements.txt
        ``` 
