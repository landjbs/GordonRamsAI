class Config:
    def __init__(self, **kwargs):
        # data processing
        self.batch_size = 100       # size of batch per update step
        self.train_frac = 0.7       # fraction of dataset to use for train
        self.val_frac = 0.2         # fraction of dataset to use for validation
        self.test_frac = 0.1        # fraction of dataset to use for testing
        # model architecture
        self.mask_frac = 0.1
        # setup
        # self.overwrite_kwargs()
        self.run_assertions()

    def data_split_assertions(self):
        split_fracs = (self.train_frac, self.val_frac, self.test_frac)
        assert (sum(split_fracs) == 1), 'Data split fractions must sum to 1.'
        assert all(split_fracs >= 0), 'Data split fractions must be geq 0.'
        return True

    def run_assertions(self):
        self.data_split_assertions()
