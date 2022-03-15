from model import SEDense18, SEDense34, ResNet50
from dataset import *
from prefetch_generator import BackgroundGenerator
from torch.backends import cudnn
import aimodelshare as ai
from aimodelshare.aimsonnx import model_to_onnx
from aimodelshare.aws import set_credentials
import glob
from scipy import stats

cudnn.enabled = True
cudnn.benchmark = True


class DataLoaderX(DataLoader):
    def __iter__(self):
        return BackgroundGenerator(super().__iter__())


def load_model(checkpoint=None):
    epoch = 0
    net = ResNet50().cuda()
    # net = nn.DataParallel(net)
    if checkpoint and os.path.exists(checkpoint):
        _, epoch = os.path.basename(checkpoint).split('_')
        epoch = int(epoch[:-3])
        net.load_state_dict(torch.load(checkpoint))
    return net, epoch


def evaluate(val_ds, model, batch_size=50):
    """
    Evaluate with accuracy
    """
    val_loader = DataLoaderX(val_ds, batch_size, num_workers=4, pin_memory=True)
    correct = 0
    with torch.no_grad():
        for image, label in tqdm(val_loader):
            image, label = image.cuda(), label.cuda()
            predict = torch.argmax(model(image), dim=-1)
            correct += torch.count_nonzero(torch.eq(predict, label))
    accuracy = correct / (len(val_loader) * batch_size)
    accuracy = accuracy.item()
    print("validation accuracy {:.4f}".format(accuracy))
    return accuracy


def baseline(train_ds, val_ds, batch_size=50, epochs=40, start_lr=0.001, milestones="500, 1500, 3500",
             gpu="0", snapshot=None, models_path="./checkpoint"):
    if milestones is None:
        milestones = "100,200,300"
    os.environ["CUDA_VISIBLE_DEVICES"] = gpu
    net, starting_epoch = load_model(snapshot)
    if starting_epoch >= epochs:
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

    optimizer = optim.Adam(net.parameters(), lr=start_lr)
    scheduler = MultiStepLR(optimizer, milestones=[int(x) for x in milestones.split(',')], gamma=0.5)
    criterion = nn.CrossEntropyLoss(label_smoothing=0.1)
    iter = 0
    best_score = 0.895
    net.train()
    for epoch in range(starting_epoch, starting_epoch + epochs):
        train_loader = DataLoaderX(train_ds, batch_size, shuffle=True, pin_memory=True,
                                   num_workers=4)
        f1_score = evaluate(val_ds, net, batch_size)
        if f1_score > best_score:
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


def test(model):
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

    # Test Time Augmentation
    test_transform = transforms.Compose([
        transforms.RandomHorizontalFlip(),
        transforms.RandomVerticalFlip(),
        transforms.Pad(10),
        transforms.RandomCrop((120, 120))
    ])
    for i in range(0, len(tensor_X_test_submissiondata), 64):
        images = tensor_X_test_submissiondata[i:min(i + 64, len(tensor_X_test_submissiondata))]
        augmented_predict = list()
        for _ in range(4):
            augmented_images = test_transform(images)
            predict = model(augmented_images).argmax(axis=1).cpu().numpy()
            augmented_predict.append(predict)
        augmented_predict = np.asarray(augmented_predict).T
        batched_predict = stats.mode(augmented_predict, axis=1)[0].flatten().tolist()
        prediction_column_index.extend(batched_predict)

    # extract correct prediction labels
    prediction_labels = [['forest', 'nonforest', 'snow_shadow_cloud'][i] for i in prediction_column_index]
    # Submit Model 1 to Competition Leaderboard
    mycompetition.submit_model(model_filepath="model.onnx",
                               preprocessor_filepath="preprocessor.zip",
                               prediction_submission=prediction_labels)


if __name__ == "__main__":
    train_ds, val_ds = data_prepare()
    checkpoints = glob.glob("checkpoint/*.pt")
    if checkpoints:
        checkpoint = sorted(checkpoints)[-1]
    else:
        checkpoint = None
    net = baseline(train_ds, val_ds, snapshot=checkpoint)
    onnx_model = toONNX(net)
    test(net)

