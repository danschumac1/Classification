# Classification

## DATA SETUP

1. Setup environment
    a. Using conda
       - conda create -n ClassificationEnv
       - conda activate ClassificationEnv
       - conda install pip
       - pip install -r ./resources/requirements.txt

    b. Using venv
       - python -m venv venv
       - source ./venv/bin/activate
       - pip install -r ./resources/requirements.txt


2. Get the raw data
    - chmod +x ./bin/_get_data.sh
    - ./bin/_get_data.sh


3. Clean the data
    - chmod +x ./bin/clean_data.sh
    - ./bin/clean_data.sh


4. Load train/test splits in your Python code
    - Example of loading train/test with artifacts
        train, test = load_train_test(
            input_folder="./data/samples/ctu",  # for example, if using the CTU dataset
            n_shots=0,                          # can be 0 to 5 (I think) loads n shots per unique class in dataset
            mmap=False,                         # not important
            attach_artifacts=True,              # just set to true
            normalize=0,                        # or 1 if you want normalization
        )

