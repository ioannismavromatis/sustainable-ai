import logging
import os
import ssl

import torch
import torch.backends.cudnn as cudnn
import torch.nn as nn
import torch.optim as optim
from torchvision import datasets, transforms

from models import *
from utils import progress_bar, arguments

logging.basicConfig(encoding="utf-8", level=logging.INFO)
ssl._create_default_https_context = ssl._create_unverified_context

# Training
def train(args, model, device, train_loader, optimizer, criterion, epoch):
    print("\nEpoch: %d" % epoch)
    model.train()
    train_loss = 0
    correct = 0
    total = 0
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

        progress_bar(
            batch_idx,
            len(train_loader),
            "Loss: %.3f | Acc: %.3f%% (%d/%d)"
            % (train_loss / (batch_idx + 1), 100.0 * correct / total, correct, total),
        )


# Testing
def test(args, model, criterion, device, test_loader, epoch, net):
    model.eval()
    test_loss = 0
    correct = 0
    total = 0
    with torch.no_grad():
        for batch_idx, (inputs, targets) in enumerate(test_loader):
            inputs, targets = inputs.to(device), targets.to(device)
            outputs = model(inputs)
            loss = criterion(outputs, targets)

            test_loss += loss.item()
            _, predicted = outputs.max(1)
            total += targets.size(0)
            correct += predicted.eq(targets).sum().item()

            progress_bar(
                batch_idx,
                len(test_loader),
                "Loss: %.3f | Acc: %.3f%% (%d/%d)"
                % (
                    test_loss / (batch_idx + 1),
                    100.0 * correct / total,
                    correct,
                    total,
                ),
            )

    # Save model.
    acc = 100.0 * correct / total
    if args.save_model:
        print("Saving Current Model...")
        state = {"model": model.state_dict(), "acc": acc, "epoch": epoch, "net": net}
        if not os.path.isdir("model"):
            os.mkdir("model")
        model_path = "./model/" + net.__class__.__name__ + ".pth"
        torch.save(state, model_path)


def import_module(args):
    power_tool = __import__(args.evaluation_tool)

    if args.evaluation_tool == "eco2ai":
        power_tool.set_params(
            project_name="eco2ai",
            experiment_description=model.__class__.__name__,
            file_name="results.csv",
        )

    return power_tool


def power_evaluation(args):
    valid_tools = ["eco2ai", "codecarbon", "carbontracker"]

    if args.evaluation_tool is not None:
        tool = args.evaluation_tool.lower()
        if args.evaluation_tool not in valid_tools:
            raise ValueError(
                f'Tool "{args.evaluation_tool}" is not available tool for evaluation. Available tools are "Eco2AI", "CodeCarbon", and "CarbonTracker"'
            )

        logging.info(
            f"Tool '{args.evaluation_tool}' will be used for power performance evaluation"
        )

        power_tool = import_module(args)

        return power_tool
    else:
        logging.info("No tool will be used for the power performance evaluation")

        return None


def main(args):
    power_tool = power_evaluation(args)

    use_cuda = not args.no_cuda and torch.cuda.is_available()
    use_mps = not args.no_mps and torch.backends.mps.is_available()

    if use_cuda:
        device = torch.device("cuda")
        print("CUDA is used")
    elif use_mps:
        device = torch.device("mps")
        print("MPS is used")
    else:
        device = torch.device("cpu")
        print("CPU is used")

    train_kwargs = {"batch_size": args.batch_size}
    test_kwargs = {"batch_size": args.test_batch_size}
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
        root="./data", train=True, download=True, transform=transform_train
    )
    testset = datasets.CIFAR10(
        root="./data", train=False, download=True, transform=transform_test
    )

    train_loader = torch.utils.data.DataLoader(trainset, **train_kwargs)
    test_loader = torch.utils.data.DataLoader(testset, **test_kwargs)

    # Model
    network_list = []
    print("==> Building model list..")
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
    network_list.append(RegNetX_200MF())

    for net in network_list:
        start_epoch = 1  # start from epoch 1 or last model epoch
        print(f"==> Run experiment for network: {net.__class__.__name__}")
        if args.resume:
            # Load model.
            load_path = "./model/" + net.__class__.__name__ + ".pth"
            if os.path.isfile(load_path):
                print(f"==> Resuming model.. {net.__class__.__name__}")
                model = torch.load(load_path)
                net.load_state_dict(model["model"])
                start_epoch = model["epoch"] + 1
            else:
                print(
                    f'==> No "model" was found for network "{net.__class__.__name__}"! Skip loading and start from first epoch!'
                )

        model = net.to(device)
        if device == "cuda":
            model = torch.nn.DataParallel(model)
            cudnn.benchmark = True

        criterion = nn.CrossEntropyLoss()
        optimizer = optim.SGD(
            model.parameters(), lr=args.learning_rate, momentum=0.9, weight_decay=5e-4
        )
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=200)

        if power_tool is not None:
            trackerEco2AI = power_tool.Tracker(
                cpu_processes="all", ignore_warnings=True
            )

        for epoch in range(start_epoch, args.epochs + 1):
            if power_tool is not None:
                trackerEco2AI.start()

            train(args, model, device, train_loader, optimizer, criterion, epoch)

            if power_tool is not None:
                trackerEco2AI.stop()
            test(args, model, criterion, device, test_loader, epoch, net)
            scheduler.step()


if __name__ == "__main__":
    args = arguments()

    try:
        main(args)
    except (KeyboardInterrupt, SystemExit):
        logging.info("Program was interrupted. Gracefully stop it.")
        os._exit(0)
