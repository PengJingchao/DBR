class Config:
    max_epoch_num = 100
    max_test_num = 12000
    mini_batch_size = 64

    # StepLR
    lr = 1e-4
    step_size = 30  # step size of LR_Schedular
    gamma = 0.1  # decay rate of LR_Schedular

    # CyclicLR
    base_lr = 5e-6
    max_lr = 2e-5
    step_size_up = 50
    step_size_down = 50


config = Config()
