import torch
import torch.nn as nn
from torchvision import models


class SEBlock(nn.Module):
    def __init__(self, c_in):
        super().__init__()
        self.globalavgpooling = nn.AdaptiveAvgPool2d(1)
        self.fc1 = nn.Linear(c_in, max(1, c_in // 16))
        self.relu = nn.ReLU(inplace=True)
        self.fc2 = nn.Linear(max(1, c_in // 16), c_in)
        self.sigmoid = nn.Sigmoid()
        self.c_in = c_in

    def forward(self, x):
        assert self.c_in == x.size(1)
        x = self.globalavgpooling(x)
        x = x.squeeze()
        x = self.fc1(x)
        x = self.relu(x)
        x = self.fc2(x)
        x = x.unsqueeze(-1).unsqueeze(-1)
        x = self.sigmoid(x)
        return x


class ResNet50(nn.Module):
    def __init__(self, num_class=3):
        super(ResNet50, self).__init__()
        model = models.resnet50(pretrained=True)
        self.conv0 = model.conv1
        self.bn0 = model.bn1
        self.relu0 = model.relu
        self.pooling0 = model.maxpool
        self.layer1 = model.layer1
        self.layer2 = model.layer2
        self.layer3 = model.layer3
        self.layer4 = model.layer4
        self.avgpool = model.avgpool

        self.fc = nn.Linear(2048, 128)
        self.bnlast = nn.BatchNorm1d(128)
        self.relulast = nn.ReLU(inplace=True)
        self.dropout = nn.Dropout()

        self.classifier = nn.Linear(128, num_class)

    def forward(self, x):
        x = self.conv0(x)
        x = self.bn0(x)
        x = self.relu0(x)
        x = self.pooling0(x)

        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)
        x = self.avgpool(x)
        x = x.view(x.size(0), -1)
        x = self.fc(x)
        x = self.bnlast(x)
        x = self.relulast(x)
        x = self.dropout(x)
        x = self.classifier(x)
        return x


class SEDense34(nn.Module):
    def __init__(self, num_class=3, needs_norm=True):
        super().__init__()
        model = models.resnet34(pretrained=True)
        self.conv0 = model.conv1
        self.bn0 = model.bn1
        self.relu0 = model.relu
        self.pooling0 = model.maxpool
        # layer1
        self.bottleneck11 = model.layer1[0]
        self.bottleneck12 = model.layer1[1]
        self.bottleneck13 = model.layer1[2]

        self.seblock11 = SEBlock(64)
        self.seblock12 = SEBlock(64)
        self.seblock13 = SEBlock(64)
        # layer2
        self.bottleneck21 = model.layer2[0]
        self.bottleneck22 = model.layer2[1]
        self.bottleneck23 = model.layer2[2]
        self.bottleneck24 = model.layer2[3]

        self.auxconv1 = nn.Conv2d(64, 128, 1, 2, 0)
        self.optionalbn1 = nn.BatchNorm2d(128)
        self.seblock21 = SEBlock(128)
        self.seblock22 = SEBlock(128)
        self.seblock23 = SEBlock(128)
        self.seblock24 = SEBlock(128)
        # layer3
        self.bottleneck31 = model.layer3[0]
        self.bottleneck32 = model.layer3[1]
        self.bottleneck33 = model.layer3[2]
        self.bottleneck34 = model.layer3[3]
        self.bottleneck35 = model.layer3[4]
        self.bottleneck36 = model.layer3[5]

        self.auxconv2 = nn.Conv2d(128, 256, 1, 2, 0)
        self.optionalbn2 = nn.BatchNorm2d(256)
        self.seblock31 = SEBlock(256)
        self.seblock32 = SEBlock(256)
        self.seblock33 = SEBlock(256)
        self.seblock34 = SEBlock(256)
        self.seblock35 = SEBlock(256)
        self.seblock36 = SEBlock(256)
        # layer4
        self.bottleneck41 = model.layer4[0]
        self.bottleneck42 = model.layer4[1]
        self.bottleneck43 = model.layer4[2]

        self.auxconv3 = nn.Conv2d(256, 512, 1, 2, 0)
        self.optionalbn3 = nn.BatchNorm2d(512)
        self.seblock41 = SEBlock(512)
        self.seblock42 = SEBlock(512)
        self.seblock43 = SEBlock(512)

        self.avgpool = model.avgpool
        self.fc = nn.Linear(512, 128)
        self.bnlast = nn.BatchNorm1d(128)
        self.relulast = nn.ReLU(inplace=True)
        self.dropout = nn.Dropout()

        self.classifier = nn.Linear(128, num_class)
        self.norm = needs_norm

    def forward(self, x):
        x = self.conv0(x)
        x = self.bn0(x)
        x = self.relu0(x)
        x = self.pooling0(x)

        branch1 = x
        x = self.bottleneck11(x)
        scale1 = self.seblock11(x)
        x = scale1 * x + branch1

        branch2 = x
        x = self.bottleneck12(x)
        scale2 = self.seblock12(x)
        x = scale2 * x + branch2

        branch3 = x
        x = self.bottleneck13(x)
        scale3 = self.seblock13(x)
        x = scale3 * x + branch3

        branch4 = x
        x = self.bottleneck21(x)
        scale4 = self.seblock21(x)
        if self.norm:
            x = scale4 * x + self.optionalbn1(self.auxconv1(branch4))
        else:
            x = scale4 * x + self.auxconv1(branch4)

        branch5 = x
        x = self.bottleneck22(x)
        scale5 = self.seblock22(x)
        x = scale5 * x + branch5

        branch6 = x
        x = self.bottleneck23(x)
        scale6 = self.seblock23(x)
        x = scale6 * x + branch6

        branch7 = x
        x = self.bottleneck24(x)
        scale7 = self.seblock24(x)
        x = scale7 * x + branch7

        branch8 = x
        x = self.bottleneck31(x)
        scale8 = self.seblock31(x)
        if self.norm:
            x = scale8 * x + self.optionalbn2(self.auxconv2(branch8))
        else:
            x = scale8 * x + self.auxconv2(branch8)

        branch9 = x
        x = self.bottleneck32(x)
        scale9 = self.seblock32(x)
        x = scale9 * x + branch9

        branch10 = x
        x = self.bottleneck33(x)
        scale10 = self.seblock33(x)
        x = scale10 * x + branch10

        branch11 = x
        x = self.bottleneck34(x)
        scale11 = self.seblock34(x)
        x = scale11 * x + branch11

        branch12 = x
        x = self.bottleneck35(x)
        scale12 = self.seblock35(x)
        x = scale12 * x + branch12

        branch13 = x
        x = self.bottleneck36(x)
        scale13 = self.seblock36(x)
        x = scale13 * x + branch13

        branch14 = x
        x = self.bottleneck41(x)
        scale14 = self.seblock41(x)
        if self.norm:
            x = scale14 * x + self.optionalbn3(self.auxconv3(branch14))
        else:
            x = scale14 * x + self.auxconv3(branch14)

        branch15 = x
        x = self.bottleneck42(x)
        scale15 = self.seblock42(x)
        x = scale15 * x + branch15

        branch16 = x
        x = self.bottleneck43(x)
        scale16 = self.seblock43(x)
        x = scale16 * x + branch16

        x = self.avgpool(x)
        x = x.view(x.size(0), -1)
        x = self.fc(x)
        x = self.bnlast(x)
        x = self.relulast(x)
        x = self.dropout(x)
        x = self.classifier(x)
        return x


class SEDense18(nn.Module):
    def __init__(self, num_class=3, needs_norm=True):
        super().__init__()
        model = models.resnet18(pretrained=True)
        self.conv0 = model.conv1
        self.bn0 = model.bn1
        self.relu0 = model.relu
        self.pooling0 = model.maxpool
        self.basicBlock11 = model.layer1[0]
        self.seblock1 = SEBlock(64)

        self.basicBlock12 = model.layer1[1]
        self.seblock2 = SEBlock(64)

        self.basicBlock21 = model.layer2[0]
        self.seblock3 = SEBlock(128)
        self.ancillaryconv3 = nn.Conv2d(64, 128, 1, 2, 0)
        self.optionalNorm2dconv3 = nn.BatchNorm2d(128)

        self.basicBlock22 = model.layer2[1]
        self.seblock4 = SEBlock(128)

        self.basicBlock31 = model.layer3[0]
        self.seblock5 = SEBlock(256)
        self.ancillaryconv5 = nn.Conv2d(128, 256, 1, 2, 0)
        self.optionalNorm2dconv5 = nn.BatchNorm2d(256)

        self.basicBlock32 = model.layer3[1]
        self.seblock6 = SEBlock(256)

        self.basicBlock41 = model.layer4[0]
        # last stride = 1
        self.basicBlock41.conv1 = nn.Conv2d(256, 512, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), bias=False,
                                            device="cuda:0")
        self.basicBlock41.downsample[0] = nn.Conv2d(256, 512, kernel_size=(1, 1), stride=(1, 1), bias=False,
                                                    device="cuda:0")
        self.seblock7 = SEBlock(512)
        self.ancillaryconv7 = nn.Conv2d(256, 512, 1, 1, 0)
        self.optionalNorm2dconv7 = nn.BatchNorm2d(512)

        self.basicBlock42 = model.layer4[1]
        self.seblock8 = SEBlock(512)

        self.avgpooling = model.avgpool
        # self.fc = nn.Linear(512, num_class)
        self.classifier = nn.Sequential(
            nn.Linear(512, 256),
            nn.BatchNorm1d(256),
            nn.ReLU(inplace=True),
            nn.Dropout(),
            nn.Linear(256, num_class),
        )
        self.needs_norm = needs_norm

    def forward(self, x):
        x = self.conv0(x)
        x = self.bn0(x)
        x = self.relu0(x)
        x = self.pooling0(x)
        branch1 = x
        x = self.basicBlock11(x)
        scale1 = self.seblock1(x)
        x = scale1 * x + branch1

        branch2 = x
        x = self.basicBlock12(x)
        scale2 = self.seblock2(x)
        x = scale2 * x + branch2

        branch3 = x
        x = self.basicBlock21(x)
        scale3 = self.seblock3(x)
        if self.needs_norm:
            x = scale3 * x + self.optionalNorm2dconv3(self.ancillaryconv3(branch3))
        else:
            x = scale3 * x + self.ancillaryconv3(branch3)

        branch4 = x
        x = self.basicBlock22(x)
        scale4 = self.seblock4(x)
        x = scale4 * x + branch4

        branch5 = x
        x = self.basicBlock31(x)
        scale5 = self.seblock5(x)
        if self.needs_norm:
            x = scale5 * x + self.optionalNorm2dconv5(self.ancillaryconv5(branch5))
        else:
            x = scale5 * x + self.ancillaryconv5(branch5)

        branch6 = x
        x = self.basicBlock32(x)
        scale6 = self.seblock6(x)
        x = scale6 * x + branch6

        branch7 = x
        x = self.basicBlock41(x)
        scale7 = self.seblock7(x)
        if self.needs_norm:
            x = scale7 * x + self.optionalNorm2dconv7(self.ancillaryconv7(branch7))
        else:
            x = scale7 * x + self.ancillaryconv7(branch7)

        branch8 = x
        x = self.basicBlock42(x)
        scale8 = self.seblock8(x)
        x = scale8 * x + branch8

        x = self.avgpooling(x)
        x = x.view(x.size(0), -1)
        x = self.classifier(x)

        return x

