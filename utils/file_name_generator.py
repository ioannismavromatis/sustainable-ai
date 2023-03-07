def file_name_generator(args, mode):
    if mode != "train" and mode != "test":
        raise ValueError("A wrong mode type was given. Give either 'train' or 'test'.")

    if args.get_stats:
        mode = "stats_" + mode
    else:
        mode = "nostats_" + mode

    if args.tool is not "":
        mode = args.tool +  "_" + mode

    return mode
