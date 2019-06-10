def params():
    return {
        # trainning iteration
        'iter' : 6000,
        # batch size
        'batch_size' : 128,
        # when tokenlize text, only take max_char_num chars which appear most frequently
        'max_char_num' : 128,
        # forward and backward rnn cell args
        'fw_num_units' : 20,
        'bw_num_units' : 20,
        # the weight fraction between good char (0) and bad char (1)
        'loss_weight_fraction' : 8,
        'checkpoint_save_path' : './trained_models/bi_directional_gru/',
        'log_path' : './logs/bi_directional_gru/'
    }