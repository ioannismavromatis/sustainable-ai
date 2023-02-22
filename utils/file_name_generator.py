def file_name_generator(args, mode):
    if mode != "train" and mode != "test":
        raise ValueError("A wrong mode type was given. Give either 'train' or 'test'.")

    if args.tool is None:
        return mode

    return mode + "_" + args.tool
