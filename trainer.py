from model import *
from dataset import *
from prefetch_generator import BackgroundGenerator
from torch.backends import cudnn
import aimodelshare as ai
from aimodelshare.aimsonnx import model_to_onnx
from aimodelshare.aws import set_credentials
import glob
from scipy import stats
import torch.nn.functional as F
from sklearn.metrics import f1_score
import argparse
from collections import OrderedDict

cudnn.enabled = True
cudnn.benchmark = True


class DataLoaderX(DataLoader):
    def __iter__(self):
        return BackgroundGenerator(super().__iter__())


def F1score(predict, target):
    predict = np.asarray(predict)
    target = np.asarray(target)
    return f1_score(target, predict, average="macro")


def load_model(backend="resnet50", checkpoint=None):
    epoch = 0
    net = Baseline(backend=backend).cuda()
    # net = nn.DataParallel(net)
    if checkpoint and os.path.exists(checkpoint):
        _, epoch = os.path.basename(checkpoint).split('_')
        epoch = int(epoch[:-3])
        net.load_state_dict(torch.load(checkpoint))
    return net, epoch


def load_custom_model(checkpoint=None):
    epoch = 0
    net = Baseline(1, True).cuda()
    # net = nn.DataParallel(net)
    if checkpoint and os.path.exists(checkpoint):
        _, epoch = os.path.basename(checkpoint).split('_')
        epoch = int(epoch[:-3])
        net.load_state_dict(torch.load(checkpoint))
    return net, epoch


def evaluate(val_ds, model, batch_size=50, isCustom=False):
    """
    Evaluate with accuracy
    """
    model.eval()
    val_loader = DataLoaderX(val_ds, batch_size, num_workers=4, pin_memory=True)
    correct = 0
    predicts = []
    labels = []
    with torch.no_grad():
        for image, label in tqdm(val_loader):
            image, label = image.cuda(), label.cuda()
            if not isCustom:
                predict = torch.argmax(model(image), dim=-1)
            else:
                predict = model(image)
                # threshold1 = torch.quantile(predict, 0.67, dim=-1, keepdim=False, interpolation="nearest")
                # threshold2 = torch.quantile(predict, 0.33, dim=-1, keepdim=False, interpolation="nearest")
                for i in range(predict.size(0)):
                    if predict[i] >= 2 / 3:
                        predict[i] = 5 / 6
                    elif predict[i] > 1 / 3:
                        predict[i] = 0.5
                    else:
                        predict[i] = 1 / 6
            correct += torch.count_nonzero(torch.eq(predict, label))
            predicts.extend(predict.detach().cpu().numpy())
            labels.extend(label.detach().cpu().numpy())
    f1score = F1score(predicts, labels)
    accuracy = correct / (len(val_loader) * batch_size)
    accuracy = accuracy.item()
    print("validation accuracy {:.4f}, f1_score {:.4f}".format(accuracy, f1score))
    model.train()
    return accuracy, f1score  # , threshold1, threshold2


def baseline(train_ds, val_ds, backend, class_weights=None, batch_size=50, epochs=20, start_lr=0.001,
             milestones="500, 1500, 3500, 5500, 7500",
             gpu="0", snapshot=None, models_path="./checkpoint",
             use_cosinelr=False, test_only=False, weight_decay=5e-4):
    if milestones is None:
        milestones = "100,200,300"
    os.environ["CUDA_VISIBLE_DEVICES"] = gpu
    net, starting_epoch = load_model(backend, snapshot)
    if starting_epoch >= epochs or test_only:
        return net
    epochs = epochs - starting_epoch
    '''
        To follow this training routine you need a DataLoader that yields the tuples of the following format:
        (Bx3xHxW FloatTensor x, BxHxW LongTensor y, BxN LongTensor y_cls) where
        x - batch of input images,
        y - batch of groung truth seg maps,
        y_cls - batch of 1D tensors of dimensionality N: N total number of classes, 
        y_cls[i, T] = 1 if class T is present in image i, 0 otherwise
    '''

    optimizer = optim.Adam(net.parameters(), lr=start_lr, weight_decay=weight_decay)
    if use_cosinelr:
        scheduler = CosineAnnealingWarmRestarts(optimizer, 500, 2, eta_min=1e-5)
    else:
        scheduler = MultiStepLR(optimizer, milestones=[int(x) for x in milestones.split(',')], gamma=0.5)
    criterion = nn.CrossEntropyLoss(weight=class_weights, label_smoothing=0.)
    iter = 0
    best_score = 0.89
    net.train()
    for epoch in range(starting_epoch, starting_epoch + epochs):
        train_loader = DataLoaderX(train_ds, batch_size, shuffle=True, pin_memory=True,
                                   num_workers=4)
        accuracy, f1_score = evaluate(val_ds, net, batch_size)
        if f1_score >= best_score:
            net.eval()
            files = glob.glob("checkpoint/*.pt")
            for f in files:
                os.remove(f)
            torch.save(net.state_dict(), os.path.join(models_path, '_'.join(["CCNet", str(epoch + 1)]) + ".pt"))
            net.train()
            best_score = f1_score
        iterator = tqdm(train_loader)
        for image, label in iterator:
            optimizer.zero_grad()
            image, label = image.cuda(), label.cuda()
            predict = net(image)
            iter += 1
            loss = criterion(predict, label)
            loss.backward()
            nn.utils.clip_grad_norm_(net.parameters(), 10.)
            optimizer.step()
            scheduler.step()
            status = "epoch: {}, iter: {}, loss: {:.4f}, lr: {:.5f}".format(epoch, iter, loss,
                                                                            scheduler.get_last_lr()[0])
            iterator.set_description(status)
    net.eval()
    file = glob.glob("checkpoint/*.pt")[0]
    net.load_state_dict(torch.load(file))
    return net


def trainInception(train_ds, val_ds, backend, class_weights=None, batch_size=50, epochs=20, start_lr=0.001,
             milestones="500, 1500, 3500, 5500, 7500",
             gpu="0", snapshot=None, models_path="./checkpoint",
             use_cosinelr=False, test_only=False, weight_decay=5e-4):
    if milestones is None:
        milestones = "100,200,300"
    os.environ["CUDA_VISIBLE_DEVICES"] = gpu
    net, starting_epoch = load_model(backend, snapshot)
    if starting_epoch >= epochs or test_only:
        return net
    epochs = epochs - starting_epoch
    '''
        To follow this training routine you need a DataLoader that yields the tuples of the following format:
        (Bx3xHxW FloatTensor x, BxHxW LongTensor y, BxN LongTensor y_cls) where
        x - batch of input images,
        y - batch of groung truth seg maps,
        y_cls - batch of 1D tensors of dimensionality N: N total number of classes, 
        y_cls[i, T] = 1 if class T is present in image i, 0 otherwise
    '''
    optimizer = optim.Adam(net.parameters(), lr=start_lr, weight_decay=weight_decay)
    if use_cosinelr:
        scheduler = CosineAnnealingWarmRestarts(optimizer, 500, 2, eta_min=1e-5)
    else:
        scheduler = MultiStepLR(optimizer, milestones=[int(x) for x in milestones.split(',')], gamma=0.5,
                                last_epoch=starting_epoch-1)
    criterion = nn.CrossEntropyLoss(weight=class_weights, label_smoothing=0.)
    iter = 0
    best_score = 0.89
    net.train()
    for epoch in range(starting_epoch, starting_epoch + epochs):
        train_loader = DataLoaderX(train_ds, batch_size, shuffle=True, pin_memory=True,
                                   num_workers=4)
        accuracy, f1_score = evaluate(val_ds, net, batch_size)
        if f1_score >= best_score:
            net.eval()
            files = glob.glob("checkpoint/*.pt")
            for f in files:
                os.remove(f)
            torch.save(net.state_dict(), os.path.join(models_path, '_'.join(["CCNet", str(epoch + 1)]) + ".pt"))
            net.train()
            best_score = f1_score
        iterator = tqdm(train_loader)
        for image, label in iterator:
            optimizer.zero_grad()
            image, label = image.cuda(), label.cuda()
            predict, aux_predict = net(image)
            iter += 1
            loss = criterion(predict, label) + 0.4 * criterion(aux_predict, label)
            loss.backward()
            nn.utils.clip_grad_norm_(net.parameters(), 10.)
            optimizer.step()
            scheduler.step()
            status = "epoch: {}, iter: {}, loss: {:.4f}, lr: {:.5f}".format(epoch, iter, loss,
                                                                            scheduler.get_last_lr()[0])
            iterator.set_description(status)
    net.eval()
    file = glob.glob("checkpoint/*.pt")[0]

    # Initialize an inference version
    eval_net = Inception(training=False).cuda()
    eval_net.eval()
    states = torch.load(file)
    d = OrderedDict()
    for key in states:
        if "aux" not in key:
            d[key] = states[key]
    eval_net.load_state_dict(d)
    return eval_net


def trainCustomLoss(train_ds, val_ds, class_weights=None, batch_size=50, epochs=40, start_lr=0.001,
                    milestones="500, 1500, 3500",
                    gpu="0", snapshot=None, models_path="./checkpoint", test_only=False):
    if milestones is None:
        milestones = "1000,2000,3000"
    os.environ["CUDA_VISIBLE_DEVICES"] = gpu
    net, starting_epoch = load_custom_model(snapshot)
    if starting_epoch >= epochs or test_only:
        return net
    epochs = epochs - starting_epoch
    optimizer = optim.Adam(net.parameters(), lr=start_lr)
    scheduler = MultiStepLR(optimizer, milestones=[int(x) for x in milestones.split(',')], gamma=0.5)
    criterion = nn.MSELoss(reduction="sum")
    iter = 0
    best_score = 0.85
    ult_threshold1 = 2 / 3
    ult_threshold2 = 1 / 3
    net.train()
    for epoch in range(starting_epoch, starting_epoch + epochs):
        train_loader = DataLoaderX(train_ds, batch_size, shuffle=True, pin_memory=True,
                                   num_workers=4)
        accuracy, f1_score = evaluate(val_ds, net, batch_size, True)
        if f1_score >= best_score:
            # ult_threshold1 = threshold1
            # ult_threshold2 = threshold2
            net.eval()
            files = glob.glob("checkpoint/*.pt")
            for f in files:
                os.remove(f)
            torch.save(net.state_dict(), os.path.join(models_path, '_'.join(["CCNet", str(epoch + 1)]) + ".pt"))
            net.train()
            best_score = f1_score
        iterator = tqdm(train_loader)
        for image, label in iterator:
            optimizer.zero_grad()
            image, label = image.cuda(), label.cuda()
            predict = net(image)
            iter += 1
            loss = criterion(predict, label)
            loss.backward()
            nn.utils.clip_grad_norm_(net.parameters(), 10.)
            optimizer.step()
            scheduler.step()
            status = "epoch: {}, iter: {}, loss: {:.4f}, lr: {:.5f}".format(epoch, iter, loss,
                                                                            scheduler.get_last_lr()[0])
            iterator.set_description(status)
    net.eval()
    file = glob.glob("checkpoint/*.pt")[0]
    net.load_state_dict(torch.load(file))
    return net, ult_threshold1, ult_threshold2


def toONNX(net):
    ai.export_preprocessor(preprocessor, "")
    example_input = torch.randn(1, 3, 120, 120, requires_grad=True).cuda()

    onnx_model = model_to_onnx(net, framework='pytorch',
                               model_input=example_input,
                               transfer_learning=False,
                               deep_learning=True)

    with open("model.onnx", "wb") as f:
        f.write(onnx_model.SerializeToString())
    return onnx_model


def test(model, batch_size=50, isCustom=False, threshold1=None, threshold2=None, softVote=False, TTA=True):
    filenumbers = [str(x) for x in range(1, 5001)]
    filenames = ["competition_data/testdata/test/test" + x for x in filenumbers]

    # preprocess rbg images into 120,120,3 numpy ndarray
    preprocessed_image_data = []
    for i in filenames:
        try:
            preprocessed_image_data.append(preprocessor(i))
        except:
            pass
    X_test_submissiondata = np.vstack(preprocessed_image_data)
    X_test_submissiondata = X_test_submissiondata.transpose(0, 3, 1, 2)
    tensor_X_test_submissiondata = torch.Tensor(X_test_submissiondata).cuda()

    # Note -- This is the unique rest api that powers this climate change image classification  Model Plaground
    # ... Update the apiurl if submitting to a new competition

    apiurl = "https://srdmat3yhf.execute-api.us-east-1.amazonaws.com/prod/m"
    set_credentials(apiurl=apiurl)
    # Instantiate Competition

    mycompetition = ai.Competition(apiurl)
    prediction_column_index = list()

    if TTA:
        # Test Time Augmentation
        test_transform = transforms.Compose([
            transforms.RandomHorizontalFlip(),
            transforms.RandomVerticalFlip(),
            transforms.Pad(10),
            transforms.RandomCrop((120, 120)),
        ])
        with torch.no_grad():
            for i in range(0, len(tensor_X_test_submissiondata), batch_size):
                images = tensor_X_test_submissiondata[i:min(i + batch_size, len(tensor_X_test_submissiondata))]
                augmented_predict = list()
                for _ in range(4):
                    augmented_images = test_transform(images)
                    predict = model(augmented_images)
                    if not isCustom and not softVote:
                        predict = torch.argmax(predict, dim=-1)
                    predict = predict.cpu().detach().numpy()
                    augmented_predict.append(predict)
                augmented_predict = np.asarray(augmented_predict)
                if isCustom:
                    batched_predict = []
                    for i in range(augmented_predict.shape[1]):
                        raw_predict = augmented_predict[:, i].flatten().sum() / augmented_predict.shape[0]
                        if raw_predict > threshold1:  # non-forest
                            raw_predict = 1
                        elif raw_predict > threshold2:  # others
                            raw_predict = 2
                        else:
                            raw_predict = 0  # forest
                        batched_predict.append(raw_predict)
                else:
                    if softVote:
                        batched_predict = []
                        for i in range(augmented_predict.shape[1]):
                            raw_predict = augmented_predict[:, i, :].sum(0) / augmented_predict.shape[0]
                            batched_predict.append(np.argmax(raw_predict))
                    else:
                        augmented_predict = augmented_predict.T
                        batched_predict = stats.mode(augmented_predict, axis=1)[0].flatten().tolist()
                assert len(batched_predict) in {batch_size, 5000%batch_size}
                prediction_column_index.extend(batched_predict)
    else:
        with torch.no_grad():
            for i in range(0, len(tensor_X_test_submissiondata), batch_size):
                images = tensor_X_test_submissiondata[i:min(i + batch_size, len(tensor_X_test_submissiondata))]
                predict = torch.argmax(model(images), dim=-1).cpu().numpy()
                prediction_column_index.extend(predict)

    # extract correct prediction labels
    prediction_labels = [['forest', 'nonforest', 'snow_shadow_cloud'][i] for i in prediction_column_index]
    # Submit Model 1 to Competition Leaderboard
    mycompetition.submit_model(model_filepath="model.onnx",
                               preprocessor_filepath="preprocessor.zip",
                               prediction_submission=prediction_labels)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="climate change machine learning workshop")
    # params
    parser.add_argument("--batch_size", type=int, default=50)
    parser.add_argument("--epoch", type=int, default=40)
    parser.add_argument("--lr", type=float, default=0.001)
    parser.add_argument("--isCustom", action="store_true")
    parser.add_argument("--backbone", type=str, default="resnet50")
    parser.add_argument("--cyclical", action="store_true")
    parser.add_argument("--softvote", action="store_true")
    parser.add_argument("--tta", action="store_true")
    parser.add_argument("--test_only", action="store_true")
    parser.add_argument("--weight_decay", type=float, default=5e-4)
    args = parser.parse_args()

    train_ds, val_ds, class_weights = data_prepare(args.isCustom)
    checkpoints = glob.glob("checkpoint/*.pt")
    if checkpoints:
        checkpoint = sorted(checkpoints)[-1]
    else:
        checkpoint = None
    if args.isCustom:
        net, threshold1, threshold2 = trainCustomLoss(train_ds, val_ds, class_weights, args.batch_size, args.epoch,
                                                      snapshot=checkpoint, test_only=args.test_only)
    else:
        net = trainInception(train_ds, val_ds, args.backbone, class_weights, args.batch_size, args.epoch,
                             args.lr, snapshot=checkpoint, use_cosinelr=args.cyclical, test_only=args.test_only, weight_decay=args.weight_decay)
        threshold1 = threshold2 = None
    onnx_model = toONNX(net)
    test(net, args.batch_size, args.isCustom, threshold1, threshold2, args.softvote, args.tta)
