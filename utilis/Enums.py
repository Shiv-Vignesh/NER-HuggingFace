class Parameters:

    '''
    Class containing constant values such as BATCH_SIZE, EPOCHS & Other parameters
    
    '''

    MAX_LEN = 75
    bs = 32
    EPOCHS = 10
    TEST_SPLIT = 0.15
    EPOCHS = 10
    MAX_GRAD_NORM = 1.0

    FULL_FINETUNING = True

    PLOTS_DIR = 'graph plots'

    MODEL_DIR = 'saved model'

    TRAIN_REPORT_DIR = 'report/train'

    VALID_REPORT_DIR = 'report/valid'

    USE_L1 = False

    DOWNLOAD_WEIGHTS = True
