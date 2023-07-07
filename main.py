import os
import ssl
import sys
import time

import torch
import torch.backends.cudnn as cudnn
import torch.nn as nn
import torch.optim as optim
from torchvision import datasets, transforms

import utils.log as logger
from models import *
from power.generic_tracker import GenericTracker
from power.stats import Stats
from power.tool_results import ToolResults
from utils import (
    arguments,
    ascii,
    clean,
    file_name_generator,
    format_time,
    model_parameters,
    progress_bar,
)

ascii.print_ascii()

LOGGER = os.environ.get("LOGGER", "info")
SAMPLING_RATE = os.environ.get("SAMPLING_RATE", 0.1)

custom_logger = logger.get_logger(__name__)
custom_logger = logger.set_level(__name__, LOGGER)
custom_logger.debug("Logger initiated: %s", custom_logger)

ssl._create_default_https_context = ssl._create_unverified_context


# Training
def train(args, model, device, train_loader, optimizer, criterion, epoch, results):
    custom_logger.info("Training for Epoch: %s", epoch)
    model.train()
    train_loss = 0
    correct = 0
    total = 0
    begin_time = time.time()
    for batch_idx, (inputs, targets) in enumerate(train_loader):
        inputs, targets = inputs.to(device), targets.to(device)
        optimizer.zero_grad()
        outputs = model(inputs)
        loss = criterion(outputs, targets)
        loss.backward()
        optimizer.step()

        train_loss += loss.item()
        _, predicted = outputs.max(1)
        total += targets.size(0)
        correct += predicted.eq(targets).sum().item()

        if args.no_progress_bar:
            progress_bar(
                batch_idx,
                len(train_loader),
                "Loss: %.3f | Accuracy: %.3f%% (%d/%d)"
                % (
                    train_loss / (batch_idx + 1),
                    100.0 * correct / total,
                    correct,
                    total,
                ),
            )

    if args.no_results:
        end_time = time.time()
        tot_time = format_time(end_time - begin_time)
        step_time = format_time((end_time - begin_time) / len(train_loader))
        final_loss = "%.3f" % (train_loss / (batch_idx + 1))
        accuracy = 100.0 * correct / total
        mode = file_name_generator(args, "train")
        results.save_results(mode, epoch, tot_time, step_time, final_loss, accuracy)


def str_to_class(field):
    try:
        identifier = getattr(sys.modules[__name__], field)
    except AttributeError:
        raise NameError("%s doesn't exist." % field)
    return identifier


def make_network_list():
    network_list = []
    custom_logger.info("Building model list..")
    if args.network != "":
        custom_logger.info("Model chosen is: %s", args.network)
        network_list.append(str_to_class(args.network)())

    else:
        network_list.append(LeNet())
        network_list.append(SimpleDLA())
        network_list.append(VGG("VGG19"))
        network_list.append(ResNet18())
        network_list.append(PreActResNet18())
        network_list.append(GoogLeNet())
        network_list.append(DenseNet121())
        network_list.append(ResNeXt29_2x64d())
        network_list.append(MobileNet())
        network_list.append(MobileNetV2())
        network_list.append(SENet18())
        network_list.append(EfficientNetB0())
        network_list.append(RegNetX_200MF())

    return network_list


# Testing
def test(args, model, criterion, device, test_loader, epoch, net, results):
    model.eval()
    test_loss = 0
    correct = 0
    total = 0
    begin_time = time.time()
    with torch.no_grad():
        for batch_idx, (inputs, targets) in enumerate(test_loader):
            inputs, targets = inputs.to(device), targets.to(device)
            outputs = model(inputs)
            loss = criterion(outputs, targets)

            test_loss += loss.item()
            _, predicted = outputs.max(1)
            total += targets.size(0)
            correct += predicted.eq(targets).sum().item()

            if args.no_progress_bar:
                progress_bar(
                    batch_idx,
                    len(test_loader),
                    "Loss: %.3f | Accuracy: %.3f%% (%d/%d)"
                    % (
                        test_loss / (batch_idx + 1),
                        100.0 * correct / total,
                        correct,
                        total,
                    ),
                )
    if args.no_results:
        end_time = time.time()
        tot_time = format_time(end_time - begin_time)
        step_time = format_time((end_time - begin_time) / len(test_loader))
        final_loss = "%.3f" % (test_loss / (batch_idx + 1))
        accuracy = 100.0 * correct / total
        mode = file_name_generator(args, "test")
        results.save_results(mode, epoch, tot_time, step_time, final_loss, accuracy)

    # Save model.
    acc = 100.0 * correct / total
    if args.save_model:
        custom_logger.info("Saving Current Model...")
        state = {"model": model.state_dict(), "acc": acc, "epoch": epoch, "net": net}
        if not os.path.isdir("model"):
            os.mkdir("model")
        model_path = "./model/" + net.__class__.__name__ + ".pth"
        torch.save(state, model_path)


def main(args):
    use_cuda = not args.no_cuda and torch.cuda.is_available()
    use_mps = not args.no_mps and torch.backends.mps.is_available()

    if use_cuda:
        device = torch.device("cuda")
        custom_logger.info("NVIDIA GPU is detected: CUDA will used")
    elif use_mps:
        device = torch.device("mps")
        custom_logger.info("Apple Silicon is detected: MPS will be used")
    else:
        device = torch.device("cpu")
        custom_logger.info("CPU will be used")

    train_kwargs = {"batch_size": args.batch_size}
    test_kwargs = {"batch_size": args.test_size}
    if use_cuda:
        cuda_kwargs = {"num_workers": 1, "pin_memory": True, "shuffle": True}
        train_kwargs.update(cuda_kwargs)
        test_kwargs.update(cuda_kwargs)
    else:
        cuda_kwargs = {"shuffle": True}
        train_kwargs.update(cuda_kwargs)
        test_kwargs.update(cuda_kwargs)

    transform_train = transforms.Compose(
        [
            transforms.RandomCrop(32, padding=4),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
        ]
    )

    transform_test = transforms.Compose(
        [
            transforms.ToTensor(),
            transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
        ]
    )

    trainset = datasets.CIFAR10(
        root="./dataset", train=True, download=True, transform=transform_train
    )
    testset = datasets.CIFAR10(
        root="./dataset", train=False, download=True, transform=transform_test
    )

    testset = [testset] * 5
    replicated_testset = torch.utils.data.ConcatDataset(testset)

    train_loader = torch.utils.data.DataLoader(trainset, **train_kwargs)
    test_loader = torch.utils.data.DataLoader(replicated_testset, **test_kwargs)

    network_list = make_network_list()

    if args.get_stats:
        stats = Stats(
            SAMPLING_RATE,
            device,
            args.generic_cpu,
            run_id=args.run_id,
        )

        stats.start()
        # wait for 2 second to intialise thread
        # required for small models
        time.sleep(2)

    for net in network_list:
        start_epoch = 1  # start from epoch 1 or last model epoch
        custom_logger.info("Run experiment for network: %s", net.__class__.__name__)
        if args.resume:
            # Load model.
            load_path = "./model/" + net.__class__.__name__ + ".pth"
            if os.path.isfile(load_path):
                custom_logger.info("Resuming model.. %s", net.__class__.__name__)
                model = torch.load(load_path)
                net.load_state_dict(model["model"])
                start_epoch = model["epoch"] + 1
            else:
                custom_logger.info(
                    'No "model" was found for network "%s"! Skip loading and start from first epoch!',
                    net.__class__.__name__,
                )
        model = net.to(device)
        if device == "cuda":
            model = torch.nn.DataParallel(model)
            torch.cuda.synchronize()
            cudnn.benchmark = True

        criterion = nn.CrossEntropyLoss()
        optimizer = optim.SGD(
            model.parameters(), lr=args.learning_rate, momentum=0.9, weight_decay=5e-4
        )
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=200)

        model_parameters(net, input_size=(3, 32, 32))

        custom_logger.info(
            "Create a new tracker for network: %s", net.__class__.__name__
        )
        tracker = GenericTracker(args, net)
        results = ToolResults(net.__class__.__name__, run_id=args.run_id)

        if args.get_stats:
            stats.set_network(net.__class__.__name__)

        for epoch in range(start_epoch, args.epochs + 1):
            if args.get_stats:
                stats.reset()
            if tracker.get_tracker():
                tracker.start()

            train(
                args, model, device, train_loader, optimizer, criterion, epoch, results
            )

            if args.get_stats:
                mode = file_name_generator(args, "train")
                stats.save_results(mode, epoch)
            if tracker.get_tracker():
                tracker.stop()

            time.sleep(2)

            if args.get_stats:
                stats.reset()
            if tracker.get_tracker():
                tracker.start()

            test(args, model, criterion, device, test_loader, epoch, net, results)

            if args.get_stats:
                mode = file_name_generator(args, "test")
                stats.save_results(mode, epoch)
            if tracker.get_tracker():
                tracker.stop()

            scheduler.step()

        custom_logger.info("Deleting current tracker and results objects...")
        del tracker
        del results

    if args.get_stats:
        stats.stop()
        del stats


if __name__ == "__main__":
    args = arguments()
    custom_logger.info(
        "============ Pytorch CIFAR10 Power Consumption Investigation ============"
    )

    if args.run_id != 0:
        custom_logger.info("Starting experiment with ID: %s", args.run_id)
    if args.fresh:
        clean.clean_all(args)
    if not args.no_results:
        custom_logger.info("No results will be recorded for this experiment")

    try:
        main(args)
    except (KeyboardInterrupt, SystemExit):
        custom_logger.info("Program was interrupted. Gracefully stop it.")
        os._exit(0)
