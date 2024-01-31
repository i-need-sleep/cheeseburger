def print_and_save_args_uglobals(args, logger):
    # Print args
    print('\n=====Args:')
    if type(args) == dict:
        for key, val in args.items():
            print(f'{key}: {val}')
    else:
        for key, val in vars(args).items():
            print(f'{key}: {val}')
    print()

    # Save args
    logger.log_hyperparams(args)
    return

def module_to_dict(m):
    m_dict = {}
    for key, val in vars(m).items():
        if key[: 2] in ['_', '__']:
            continue
        m_dict[key] = val
    return m_dict