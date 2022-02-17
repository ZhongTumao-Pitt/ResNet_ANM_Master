import torch.nn as nn
import torch.utils.model_zoo as model_zoo
import torch.nn.functional as F
import torch



def conv3x3(in_planes, out_planes, stride=1):
    """3x3 convolution with padding"""
    return nn.Conv1d(in_planes, out_planes, kernel_size=3, stride=stride,
                     padding=1, bias=False)


def conv1x1(in_planes, out_planes, stride=1):
    """1x1 convolution"""
    return nn.Conv1d(in_planes, out_planes, kernel_size=1, stride=stride, bias=False)


class BasicBlock(nn.Module):
    expansion = 1

    def __init__(self, inplanes, planes, stride=1, downsample=None):
        super(BasicBlock, self).__init__()
        # Both self.conv1 and self.downsample layers downsample the input when stride != 1
        self.conv1 = conv3x3(inplanes, planes, stride)
        self.bn1 = nn.BatchNorm1d(planes)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = conv3x3(planes, planes)
        self.bn2 = nn.BatchNorm1d(planes)
        self.downsample = downsample
        self.stride = stride

    def forward(self, x):
        identity = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)

        if self.downsample is not None:
            identity = self.downsample(x)

        out += identity
        out = self.relu(out)

        return out


class Bottleneck(nn.Module):
    expansion = 4

    def __init__(self, inplanes, planes, stride=1, downsample=None):
        super(Bottleneck, self).__init__()
        # Both self.conv2 and self.downsample layers downsample the input when stride != 1
        self.conv1 = conv1x1(inplanes, planes)
        self.bn1 = nn.BatchNorm1d(planes)
        self.conv2 = conv3x3(planes, planes, stride)
        self.bn2 = nn.BatchNorm1d(planes)
        self.conv3 = conv1x1(planes, planes * self.expansion)
        self.bn3 = nn.BatchNorm1d(planes * self.expansion)
        self.relu = nn.ReLU(inplace=True)
        self.downsample = downsample
        self.stride = stride

    def forward(self, x):
        identity = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)
        out = self.relu(out)

        out = self.conv3(out)
        out = self.bn3(out)

        if self.downsample is not None:
            identity = self.downsample(x)

        out += identity
        out = self.relu(out)

        return out

'''
Model 0: ResNet itself
'''
class ResNet(nn.Module):

    def __init__(self, block, layers,  inchannel=1, num_classes=6):
        super(ResNet, self).__init__()
        self.inplanes = 128
        self.conv1 = nn.Conv1d(inchannel, 128, kernel_size=7, stride=2, padding=3,
                                 bias=False)

        self.bn1 = nn.BatchNorm1d(128)
        self.relu = nn.ReLU(inplace=True)
        self.maxpool = nn.MaxPool1d(kernel_size=3, stride=2, padding=1)

        self.layer1 = self._make_layer(block, 128, layers[0], stride=1)
        self.layer2 = self._make_layer(block, 128, layers[1], stride=2)
        self.layer3 = self._make_layer(block, 256, layers[2], stride=2)
        self.layer4 = self._make_layer(block, 512, layers[3], stride=2)

        self.pool = nn.AdaptiveAvgPool1d(1)
        self.ACTClassifier = nn.Sequential(
            nn.Conv1d(512 * block.expansion, 512 * block.expansion, kernel_size=3, stride=1, padding=0, bias=False),
            nn.BatchNorm1d(512 * block.expansion),
            nn.ReLU(inplace=True),
            nn.AdaptiveAvgPool1d(1),
        )
        self.act_fc = nn.Linear(512 * block.expansion, num_classes)

    def _make_layer(self, block, planes, blocks, stride=1):
        downsample = None
        if stride != 1 or self.inplanes != planes * block.expansion:
            downsample = nn.Sequential(
                conv1x1(self.inplanes, planes * block.expansion, stride),
                nn.BatchNorm1d(planes * block.expansion),
            )

        layers = []
        layers.append(block(self.inplanes, planes, stride, downsample))
        self.inplanes = planes * block.expansion
        for _ in range(1, blocks):
            layers.append(block(self.inplanes, planes))

        return nn.Sequential(*layers)


    def _upsample_add(self, x, y):
        '''Upsample and add two feature maps.
        Args:
          x: (Variable) top feature map to be upsampled.
          y: (Variable) lateral feature map.
        Returns:
          (Variable) added feature map.
        Note in PyTorch, when input size is odd, the upsampled feature map
        with `F.upsample(..., scale_factor=2, mode='nearest')`
        maybe not equal to the lateral feature map size.
        e.g.
        original input size: [N,_,15,15] ->
        conv2d feature map size: [N,_,8,8] ->
        upsampled feature map size: [N,_,16,16]
        So we choose bilinear upsample which supports arbitrary output sizes.
        '''
        _,_,L = y.size()
        return F.interpolate(x, size=L) + y


    def forward(self, x):
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.maxpool(x)

        c1 = self.layer1(x)
        c2 = self.layer2(c1)
        c3 = self.layer3(c2)
        c4 = self.layer4(c3)

        #act = self.pool(c4)
        act = self.ACTClassifier(c4)
        act = act.view(act.size(0), -1)
        act1 = self.act_fc(act)

        return act1, x, c1, c2, c3, c4, act


'''
Model 1:
'''
class ResNetANM_1(nn.Module):
    def __init__(self, block, layers,  inchannel=1, num_classes=6):
        super(ResNetANM_1, self).__init__()
        self.inplanes = 128
        self.conv1 = nn.Conv1d(inchannel, 128, kernel_size=7, stride=2, padding=3,
                                 bias=False)

        self.bn1 = nn.BatchNorm1d(128)
        self.relu = nn.ReLU(inplace=True)
        self.maxpool = nn.MaxPool1d(kernel_size=3, stride=2, padding=1)

        self.layer1 = self._make_layer(block, 128, layers[0], stride=1)
        self.layer2 = self._make_layer(block, 128, layers[1], stride=2)
        self.layer3 = self._make_layer(block, 256, layers[2], stride=2)
        self.layer4 = self._make_layer(block, 512, layers[3], stride=2)

        self.pool = nn.AdaptiveAvgPool1d(1)
        self.ACTClassifier = nn.Sequential(
            nn.Conv1d(512 * block.expansion, 512 * block.expansion, kernel_size=3, stride=1, padding=0, bias=False),
            nn.BatchNorm1d(512 * block.expansion),
            nn.ReLU(inplace=True),
            nn.AdaptiveAvgPool1d(1),
        )
        self.act_fc = nn.Linear(512 * block.expansion, num_classes)

        self.attention1 = nn.Sequential(
            nn.Conv1d(128, 1, kernel_size=1),
            nn.Tanh()
        )
        self.attention2 = nn.Sequential(
            nn.Conv1d(128, 1, kernel_size=1),
            nn.Tanh()
        )
        self.attention3 = nn.Sequential(
            nn.Conv1d(256, 1, kernel_size=1),
            nn.Tanh()
        )
        self.attention4 = nn.Sequential(
            nn.Conv1d(512, 1, kernel_size=1, stride=1, padding=0),
            nn.Tanh()
        )

    def _make_layer(self, block, planes, blocks, stride=1):
        downsample = None
        if stride != 1 or self.inplanes != planes * block.expansion:
            downsample = nn.Sequential(
                conv1x1(self.inplanes, planes * block.expansion, stride),
                nn.BatchNorm1d(planes * block.expansion),
            )

        layers = []
        layers.append(block(self.inplanes, planes, stride, downsample))
        self.inplanes = planes * block.expansion
        for _ in range(1, blocks):
            layers.append(block(self.inplanes, planes))

        return nn.Sequential(*layers)


    def _upsample_add(self, x, y):
        '''Upsample and add two feature maps.
        Args:
          x: (Variable) top feature map to be upsampled.
          y: (Variable) lateral feature map.
        Returns:
          (Variable) added feature map.
        Note in PyTorch, when input size is odd, the upsampled feature map
        with `F.upsample(..., scale_factor=2, mode='nearest')`
        maybe not equal to the lateral feature map size.
        e.g.
        original input size: [N,_,15,15] ->
        conv2d feature map size: [N,_,8,8] ->
        upsampled feature map size: [N,_,16,16]
        So we choose bilinear upsample which supports arbitrary output sizes.
        '''
        _,_,L = y.size()
        return F.interpolate(x, size=L) + y

    def forward(self, x):
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.maxpool(x)

        c1 = self.layer1(x)
        a1 = self.attention1(c1)
        a1 = a1.expand(c1.shape[0], c1.shape[1], c1.shape[2])
        p1 = self.pool(c1)
        p1 = p1.expand(c1.shape[0], c1.shape[1], c1.shape[2])
        c11 = p1*a1
        c1 = c1 + c11

        c2 = self.layer2(c1)
        # a2 = self.attention2(c2)
        # a2 = a2.expand(c2.shape[0], c2.shape[1], c2.shape[2])
        # p2 = self.pool(c2)
        # p2 = p2.expand(c2.shape[0], c2.shape[1], c2.shape[2])
        # c22 = p2*a2
        # c2 = c2 + c22

        c3 = self.layer3(c2)
        # a3 = self.attention3(c3)
        # a3 = a3.expand(c3.shape[0], c3.shape[1], c3.shape[2])
        # p3 = self.pool(c3)
        # p3 = p3.expand(c3.shape[0], c3.shape[1], c3.shape[2])
        # c33 = p3*a3
        # c3 = c3 + c33

        c4 = self.layer4(c3)
        # a4 = self.attention4(c4)
        # a4 = a4.expand(c4.shape[0], c4.shape[1], c4.shape[2])
        # p4 = self.pool(c4)
        # p4 = p4.expand(c4.shape[0], c4.shape[1], c4.shape[2])
        # c44 = p4*a4
        # c4 = c4 + c44

        #act = self.pool(c4)
        act = self.ACTClassifier(c4)
        act = act.view(act.size(0), -1)
        act1 = self.act_fc(act)

        return act1, x, c1, c2, c3, c4, act


'''
Model 2:
'''
class ResNetANM_2(nn.Module):
    def __init__(self, block, layers,  inchannel=1, num_classes=6):
        super(ResNetANM_2, self).__init__()
        self.inplanes = 128
        self.conv1 = nn.Conv1d(inchannel, 128, kernel_size=7, stride=2, padding=3,
                                 bias=False)

        self.bn1 = nn.BatchNorm1d(128)
        self.relu = nn.ReLU(inplace=True)
        self.maxpool = nn.MaxPool1d(kernel_size=3, stride=2, padding=1)

        self.layer1 = self._make_layer(block, 128, layers[0], stride=1)
        self.layer2 = self._make_layer(block, 128, layers[1], stride=2)
        self.layer3 = self._make_layer(block, 256, layers[2], stride=2)
        self.layer4 = self._make_layer(block, 512, layers[3], stride=2)

        self.pool = nn.AdaptiveAvgPool1d(1)
        self.ACTClassifier = nn.Sequential(
            nn.Conv1d(512 * block.expansion, 512 * block.expansion, kernel_size=3, stride=1, padding=0, bias=False),
            nn.BatchNorm1d(512 * block.expansion),
            nn.ReLU(inplace=True),
            nn.AdaptiveAvgPool1d(1),
        )
        self.act_fc = nn.Linear(512 * block.expansion, num_classes)

        self.attention1 = nn.Sequential(
            nn.Conv1d(128, 1, kernel_size=1),
            nn.Tanh()
        )
        self.attention2 = nn.Sequential(
            nn.Conv1d(128, 1, kernel_size=1),
            nn.Tanh()
        )
        self.attention3 = nn.Sequential(
            nn.Conv1d(256, 1, kernel_size=1),
            nn.Tanh()
        )
        self.attention4 = nn.Sequential(
            nn.Conv1d(512, 1, kernel_size=1, stride=1, padding=0),
            nn.Tanh()
        )

    def _make_layer(self, block, planes, blocks, stride=1):
        downsample = None
        if stride != 1 or self.inplanes != planes * block.expansion:
            downsample = nn.Sequential(
                conv1x1(self.inplanes, planes * block.expansion, stride),
                nn.BatchNorm1d(planes * block.expansion),
            )

        layers = []
        layers.append(block(self.inplanes, planes, stride, downsample))
        self.inplanes = planes * block.expansion
        for _ in range(1, blocks):
            layers.append(block(self.inplanes, planes))

        return nn.Sequential(*layers)


    def _upsample_add(self, x, y):
        '''Upsample and add two feature maps.
        Args:
          x: (Variable) top feature map to be upsampled.
          y: (Variable) lateral feature map.
        Returns:
          (Variable) added feature map.
        Note in PyTorch, when input size is odd, the upsampled feature map
        with `F.upsample(..., scale_factor=2, mode='nearest')`
        maybe not equal to the lateral feature map size.
        e.g.
        original input size: [N,_,15,15] ->
        conv2d feature map size: [N,_,8,8] ->
        upsampled feature map size: [N,_,16,16]
        So we choose bilinear upsample which supports arbitrary output sizes.
        '''
        _,_,L = y.size()
        return F.interpolate(x, size=L) + y

    def forward(self, x):
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.maxpool(x)

        c1 = self.layer1(x)
        a1 = self.attention1(c1)
        a1 = a1.expand(c1.shape[0], c1.shape[1], c1.shape[2])
        p1 = self.pool(c1)
        p1 = p1.expand(c1.shape[0], c1.shape[1], c1.shape[2])
        c11 = p1*a1
        c1 = c1 + c11

        c2 = self.layer2(c1)
        a2 = self.attention2(c2)
        a2 = a2.expand(c2.shape[0], c2.shape[1], c2.shape[2])
        p2 = self.pool(c2)
        p2 = p2.expand(c2.shape[0], c2.shape[1], c2.shape[2])
        c22 = p2*a2
        c2 = c2 + c22

        c3 = self.layer3(c2)
        # a3 = self.attention3(c3)
        # a3 = a3.expand(c3.shape[0], c3.shape[1], c3.shape[2])
        # p3 = self.pool(c3)
        # p3 = p3.expand(c3.shape[0], c3.shape[1], c3.shape[2])
        # c33 = p3*a3
        # c3 = c3 + c33

        c4 = self.layer4(c3)
        # a4 = self.attention4(c4)
        # a4 = a4.expand(c4.shape[0], c4.shape[1], c4.shape[2])
        # p4 = self.pool(c4)
        # p4 = p4.expand(c4.shape[0], c4.shape[1], c4.shape[2])
        # c44 = p4*a4
        # c4 = c4 + c44

        #act = self.pool(c4)
        act = self.ACTClassifier(c4)
        act = act.view(act.size(0), -1)
        act1 = self.act_fc(act)

        return act1, x, c1, c2, c3, c4, act


'''
Model 3:
'''
class ResNetANM_3(nn.Module):
    def __init__(self, block, layers,  inchannel=1, num_classes=6):
        super(ResNetANM_3, self).__init__()
        self.inplanes = 128
        self.conv1 = nn.Conv1d(inchannel, 128, kernel_size=7, stride=2, padding=3,
                                 bias=False)

        self.bn1 = nn.BatchNorm1d(128)
        self.relu = nn.ReLU(inplace=True)
        self.maxpool = nn.MaxPool1d(kernel_size=3, stride=2, padding=1)

        self.layer1 = self._make_layer(block, 128, layers[0], stride=1)
        self.layer2 = self._make_layer(block, 128, layers[1], stride=2)
        self.layer3 = self._make_layer(block, 256, layers[2], stride=2)
        self.layer4 = self._make_layer(block, 512, layers[3], stride=2)

        self.pool = nn.AdaptiveAvgPool1d(1)
        self.ACTClassifier = nn.Sequential(
            nn.Conv1d(512 * block.expansion, 512 * block.expansion, kernel_size=3, stride=1, padding=0, bias=False),
            nn.BatchNorm1d(512 * block.expansion),
            nn.ReLU(inplace=True),
            nn.AdaptiveAvgPool1d(1),
        )
        self.act_fc = nn.Linear(512 * block.expansion, num_classes)

        self.attention1 = nn.Sequential(
            nn.Conv1d(128, 1, kernel_size=1),
            nn.Tanh()
        )
        self.attention2 = nn.Sequential(
            nn.Conv1d(128, 1, kernel_size=1),
            nn.Tanh()
        )
        self.attention3 = nn.Sequential(
            nn.Conv1d(256, 1, kernel_size=1),
            nn.Tanh()
        )
        self.attention4 = nn.Sequential(
            nn.Conv1d(512, 1, kernel_size=1, stride=1, padding=0),
            nn.Tanh()
        )

    def _make_layer(self, block, planes, blocks, stride=1):
        downsample = None
        if stride != 1 or self.inplanes != planes * block.expansion:
            downsample = nn.Sequential(
                conv1x1(self.inplanes, planes * block.expansion, stride),
                nn.BatchNorm1d(planes * block.expansion),
            )

        layers = []
        layers.append(block(self.inplanes, planes, stride, downsample))
        self.inplanes = planes * block.expansion
        for _ in range(1, blocks):
            layers.append(block(self.inplanes, planes))

        return nn.Sequential(*layers)


    def _upsample_add(self, x, y):
        '''Upsample and add two feature maps.
        Args:
          x: (Variable) top feature map to be upsampled.
          y: (Variable) lateral feature map.
        Returns:
          (Variable) added feature map.
        Note in PyTorch, when input size is odd, the upsampled feature map
        with `F.upsample(..., scale_factor=2, mode='nearest')`
        maybe not equal to the lateral feature map size.
        e.g.
        original input size: [N,_,15,15] ->
        conv2d feature map size: [N,_,8,8] ->
        upsampled feature map size: [N,_,16,16]
        So we choose bilinear upsample which supports arbitrary output sizes.
        '''
        _,_,L = y.size()
        return F.interpolate(x, size=L) + y

    def forward(self, x):
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.maxpool(x)

        c1 = self.layer1(x)
        a1 = self.attention1(c1)
        a1 = a1.expand(c1.shape[0], c1.shape[1], c1.shape[2])
        p1 = self.pool(c1)
        p1 = p1.expand(c1.shape[0], c1.shape[1], c1.shape[2])
        c11 = p1*a1
        c1 = c1 + c11

        c2 = self.layer2(c1)
        a2 = self.attention2(c2)
        a2 = a2.expand(c2.shape[0], c2.shape[1], c2.shape[2])
        p2 = self.pool(c2)
        p2 = p2.expand(c2.shape[0], c2.shape[1], c2.shape[2])
        c22 = p2*a2
        c2 = c2 + c22

        c3 = self.layer3(c2)
        a3 = self.attention3(c3)
        a3 = a3.expand(c3.shape[0], c3.shape[1], c3.shape[2])
        p3 = self.pool(c3)
        p3 = p3.expand(c3.shape[0], c3.shape[1], c3.shape[2])
        c33 = p3*a3
        c3 = c3 + c33

        c4 = self.layer4(c3)
        # a4 = self.attention4(c4)
        # a4 = a4.expand(c4.shape[0], c4.shape[1], c4.shape[2])
        # p4 = self.pool(c4)
        # p4 = p4.expand(c4.shape[0], c4.shape[1], c4.shape[2])
        # c44 = p4*a4
        # c4 = c4 + c44

        #act = self.pool(c4)
        act = self.ACTClassifier(c4)
        act = act.view(act.size(0), -1)
        act1 = self.act_fc(act)

        return act1, x, c1, c2, c3, c4, act


'''
Model 4:
'''
class ResNetANM_4(nn.Module):
    def __init__(self, block, layers,  inchannel=1, num_classes=6):
        super(ResNetANM_4, self).__init__()
        self.inplanes = 128
        self.conv1 = nn.Conv1d(inchannel, 128, kernel_size=7, stride=2, padding=3,
                                 bias=False)

        self.bn1 = nn.BatchNorm1d(128)
        self.relu = nn.ReLU(inplace=True)
        self.maxpool = nn.MaxPool1d(kernel_size=3, stride=2, padding=1)

        self.layer1 = self._make_layer(block, 128, layers[0], stride=1)
        self.layer2 = self._make_layer(block, 128, layers[1], stride=2)
        self.layer3 = self._make_layer(block, 256, layers[2], stride=2)
        self.layer4 = self._make_layer(block, 512, layers[3], stride=2)

        self.pool = nn.AdaptiveAvgPool1d(1)
        self.ACTClassifier = nn.Sequential(
            nn.Conv1d(512 * block.expansion, 512 * block.expansion, kernel_size=3, stride=1, padding=0, bias=False),
            nn.BatchNorm1d(512 * block.expansion),
            nn.ReLU(inplace=True),
            nn.AdaptiveAvgPool1d(1),
        )
        self.act_fc = nn.Linear(512 * block.expansion, num_classes)

        self.attention1 = nn.Sequential(
            nn.Conv1d(128, 1, kernel_size=1),
            nn.Tanh()
        )
        self.attention2 = nn.Sequential(
            nn.Conv1d(128, 1, kernel_size=1),
            nn.Tanh()
        )
        self.attention3 = nn.Sequential(
            nn.Conv1d(256, 1, kernel_size=1),
            nn.Tanh()
        )
        self.attention4 = nn.Sequential(
            nn.Conv1d(512, 1, kernel_size=1, stride=1, padding=0),
            nn.Tanh()
        )

    def _make_layer(self, block, planes, blocks, stride=1):
        downsample = None
        if stride != 1 or self.inplanes != planes * block.expansion:
            downsample = nn.Sequential(
                conv1x1(self.inplanes, planes * block.expansion, stride),
                nn.BatchNorm1d(planes * block.expansion),
            )

        layers = []
        layers.append(block(self.inplanes, planes, stride, downsample))
        self.inplanes = planes * block.expansion
        for _ in range(1, blocks):
            layers.append(block(self.inplanes, planes))

        return nn.Sequential(*layers)


    def _upsample_add(self, x, y):
        '''Upsample and add two feature maps.
        Args:
          x: (Variable) top feature map to be upsampled.
          y: (Variable) lateral feature map.
        Returns:
          (Variable) added feature map.
        Note in PyTorch, when input size is odd, the upsampled feature map
        with `F.upsample(..., scale_factor=2, mode='nearest')`
        maybe not equal to the lateral feature map size.
        e.g.
        original input size: [N,_,15,15] ->
        conv2d feature map size: [N,_,8,8] ->
        upsampled feature map size: [N,_,16,16]
        So we choose bilinear upsample which supports arbitrary output sizes.
        '''
        _,_,L = y.size()
        return F.interpolate(x, size=L) + y

    def forward(self, x):
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.maxpool(x)

        c1 = self.layer1(x)
        a1 = self.attention1(c1)
        a1 = a1.expand(c1.shape[0], c1.shape[1], c1.shape[2])
        p1 = self.pool(c1)
        p1 = p1.expand(c1.shape[0], c1.shape[1], c1.shape[2])
        c11 = p1*a1
        c1 = c1 + c11

        c2 = self.layer2(c1)
        a2 = self.attention2(c2)
        a2 = a2.expand(c2.shape[0], c2.shape[1], c2.shape[2])
        p2 = self.pool(c2)
        p2 = p2.expand(c2.shape[0], c2.shape[1], c2.shape[2])
        c22 = p2*a2
        c2 = c2 + c22

        c3 = self.layer3(c2)
        a3 = self.attention3(c3)
        a3 = a3.expand(c3.shape[0], c3.shape[1], c3.shape[2])
        p3 = self.pool(c3)
        p3 = p3.expand(c3.shape[0], c3.shape[1], c3.shape[2])
        c33 = p3*a3
        c3 = c3 + c33

        c4 = self.layer4(c3)
        a4 = self.attention4(c4)
        a4 = a4.expand(c4.shape[0], c4.shape[1], c4.shape[2])
        p4 = self.pool(c4)
        p4 = p4.expand(c4.shape[0], c4.shape[1], c4.shape[2])
        c44 = p4*a4
        c4 = c4 + c44

        #act = self.pool(c4)
        act = self.ACTClassifier(c4)
        act = act.view(act.size(0), -1)
        act1 = self.act_fc(act)

        return act1, x, c1, c2, c3, c4, act

'''
Model 5:
'''
class ResNetANM_5(nn.Module):
    def __init__(self, block, layers,  inchannel=1, num_classes=6):
        super(ResNetANM_5, self).__init__()
        self.inplanes = 128
        self.conv1 = nn.Conv1d(inchannel, 128, kernel_size=7, stride=2, padding=3,
                                 bias=False)

        self.bn1 = nn.BatchNorm1d(128)
        self.relu = nn.ReLU(inplace=True)
        self.maxpool = nn.MaxPool1d(kernel_size=3, stride=2, padding=1)

        self.layer1 = self._make_layer(block, 128, layers[0], stride=1)
        self.layer2 = self._make_layer(block, 128, layers[1], stride=2)
        self.layer3 = self._make_layer(block, 256, layers[2], stride=2)
        self.layer4 = self._make_layer(block, 512, layers[3], stride=2)

        self.pool = nn.AdaptiveAvgPool1d(1)
        self.ACTClassifier = nn.Sequential(
            nn.Conv1d(512 * block.expansion, 512 * block.expansion, kernel_size=3, stride=1, padding=0, bias=False),
            nn.BatchNorm1d(512 * block.expansion),
            nn.ReLU(inplace=True),
            nn.AdaptiveAvgPool1d(1),
        )
        self.act_fc = nn.Linear(512 * block.expansion, num_classes)

        self.attention1 = nn.Sequential(
            nn.Conv1d(128, 1, kernel_size=1),
            nn.Tanh()
        )
        self.attention2 = nn.Sequential(
            nn.Conv1d(128, 1, kernel_size=1),
            nn.Tanh()
        )
        self.attention3 = nn.Sequential(
            nn.Conv1d(256, 1, kernel_size=1),
            nn.Tanh()
        )
        self.attention4 = nn.Sequential(
            nn.Conv1d(512, 1, kernel_size=1, stride=1, padding=0),
            nn.Tanh()
        )

    def _make_layer(self, block, planes, blocks, stride=1):
        downsample = None
        if stride != 1 or self.inplanes != planes * block.expansion:
            downsample = nn.Sequential(
                conv1x1(self.inplanes, planes * block.expansion, stride),
                nn.BatchNorm1d(planes * block.expansion),
            )

        layers = []
        layers.append(block(self.inplanes, planes, stride, downsample))
        self.inplanes = planes * block.expansion
        for _ in range(1, blocks):
            layers.append(block(self.inplanes, planes))

        return nn.Sequential(*layers)


    def _upsample_add(self, x, y):
        '''Upsample and add two feature maps.
        Args:
          x: (Variable) top feature map to be upsampled.
          y: (Variable) lateral feature map.
        Returns:
          (Variable) added feature map.
        Note in PyTorch, when input size is odd, the upsampled feature map
        with `F.upsample(..., scale_factor=2, mode='nearest')`
        maybe not equal to the lateral feature map size.
        e.g.
        original input size: [N,_,15,15] ->
        conv2d feature map size: [N,_,8,8] ->
        upsampled feature map size: [N,_,16,16]
        So we choose bilinear upsample which supports arbitrary output sizes.
        '''
        _,_,L = y.size()
        return F.interpolate(x, size=L) + y

    def forward(self, x):
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.maxpool(x)

        c1 = self.layer1(x)
        # a1 = self.attention1(c1)
        # a1 = a1.expand(c1.shape[0], c1.shape[1], c1.shape[2])
        # p1 = self.pool(c1)
        # p1 = p1.expand(c1.shape[0], c1.shape[1], c1.shape[2])
        # c11 = p1*a1
        # c1 = c1 + c11

        c2 = self.layer2(c1)
        a2 = self.attention2(c2)
        a2 = a2.expand(c2.shape[0], c2.shape[1], c2.shape[2])
        p2 = self.pool(c2)
        p2 = p2.expand(c2.shape[0], c2.shape[1], c2.shape[2])
        c22 = p2*a2
        c2 = c2 + c22

        c3 = self.layer3(c2)
        a3 = self.attention3(c3)
        a3 = a3.expand(c3.shape[0], c3.shape[1], c3.shape[2])
        p3 = self.pool(c3)
        p3 = p3.expand(c3.shape[0], c3.shape[1], c3.shape[2])
        c33 = p3*a3
        c3 = c3 + c33

        c4 = self.layer4(c3)
        a4 = self.attention4(c4)
        a4 = a4.expand(c4.shape[0], c4.shape[1], c4.shape[2])
        p4 = self.pool(c4)
        p4 = p4.expand(c4.shape[0], c4.shape[1], c4.shape[2])
        c44 = p4*a4
        c4 = c4 + c44

        #act = self.pool(c4)
        act = self.ACTClassifier(c4)
        act = act.view(act.size(0), -1)
        act1 = self.act_fc(act)

        return act1, x, c1, c2, c3, c4, act

'''
Model 6:
'''
class ResNetANM_6(nn.Module):
    def __init__(self, block, layers,  inchannel=1, num_classes=6):
        super(ResNetANM_6, self).__init__()
        self.inplanes = 128
        self.conv1 = nn.Conv1d(inchannel, 128, kernel_size=7, stride=2, padding=3,
                                 bias=False)

        self.bn1 = nn.BatchNorm1d(128)
        self.relu = nn.ReLU(inplace=True)
        self.maxpool = nn.MaxPool1d(kernel_size=3, stride=2, padding=1)

        self.layer1 = self._make_layer(block, 128, layers[0], stride=1)
        self.layer2 = self._make_layer(block, 128, layers[1], stride=2)
        self.layer3 = self._make_layer(block, 256, layers[2], stride=2)
        self.layer4 = self._make_layer(block, 512, layers[3], stride=2)

        self.pool = nn.AdaptiveAvgPool1d(1)
        self.ACTClassifier = nn.Sequential(
            nn.Conv1d(512 * block.expansion, 512 * block.expansion, kernel_size=3, stride=1, padding=0, bias=False),
            nn.BatchNorm1d(512 * block.expansion),
            nn.ReLU(inplace=True),
            nn.AdaptiveAvgPool1d(1),
        )
        self.act_fc = nn.Linear(512 * block.expansion, num_classes)

        self.attention1 = nn.Sequential(
            nn.Conv1d(128, 1, kernel_size=1),
            nn.Tanh()
        )
        self.attention2 = nn.Sequential(
            nn.Conv1d(128, 1, kernel_size=1),
            nn.Tanh()
        )
        self.attention3 = nn.Sequential(
            nn.Conv1d(256, 1, kernel_size=1),
            nn.Tanh()
        )
        self.attention4 = nn.Sequential(
            nn.Conv1d(512, 1, kernel_size=1, stride=1, padding=0),
            nn.Tanh()
        )

    def _make_layer(self, block, planes, blocks, stride=1):
        downsample = None
        if stride != 1 or self.inplanes != planes * block.expansion:
            downsample = nn.Sequential(
                conv1x1(self.inplanes, planes * block.expansion, stride),
                nn.BatchNorm1d(planes * block.expansion),
            )

        layers = []
        layers.append(block(self.inplanes, planes, stride, downsample))
        self.inplanes = planes * block.expansion
        for _ in range(1, blocks):
            layers.append(block(self.inplanes, planes))

        return nn.Sequential(*layers)


    def _upsample_add(self, x, y):
        '''Upsample and add two feature maps.
        Args:
          x: (Variable) top feature map to be upsampled.
          y: (Variable) lateral feature map.
        Returns:
          (Variable) added feature map.
        Note in PyTorch, when input size is odd, the upsampled feature map
        with `F.upsample(..., scale_factor=2, mode='nearest')`
        maybe not equal to the lateral feature map size.
        e.g.
        original input size: [N,_,15,15] ->
        conv2d feature map size: [N,_,8,8] ->
        upsampled feature map size: [N,_,16,16]
        So we choose bilinear upsample which supports arbitrary output sizes.
        '''
        _,_,L = y.size()
        return F.interpolate(x, size=L) + y

    def forward(self, x):
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.maxpool(x)

        c1 = self.layer1(x)
        # a1 = self.attention1(c1)
        # a1 = a1.expand(c1.shape[0], c1.shape[1], c1.shape[2])
        # p1 = self.pool(c1)
        # p1 = p1.expand(c1.shape[0], c1.shape[1], c1.shape[2])
        # c11 = p1*a1
        # c1 = c1 + c11

        c2 = self.layer2(c1)
        # a2 = self.attention2(c2)
        # a2 = a2.expand(c2.shape[0], c2.shape[1], c2.shape[2])
        # p2 = self.pool(c2)
        # p2 = p2.expand(c2.shape[0], c2.shape[1], c2.shape[2])
        # c22 = p2*a2
        # c2 = c2 + c22

        c3 = self.layer3(c2)
        a3 = self.attention3(c3)
        a3 = a3.expand(c3.shape[0], c3.shape[1], c3.shape[2])
        p3 = self.pool(c3)
        p3 = p3.expand(c3.shape[0], c3.shape[1], c3.shape[2])
        c33 = p3*a3
        c3 = c3 + c33

        c4 = self.layer4(c3)
        a4 = self.attention4(c4)
        a4 = a4.expand(c4.shape[0], c4.shape[1], c4.shape[2])
        p4 = self.pool(c4)
        p4 = p4.expand(c4.shape[0], c4.shape[1], c4.shape[2])
        c44 = p4*a4
        c4 = c4 + c44

        #act = self.pool(c4)
        act = self.ACTClassifier(c4)
        act = act.view(act.size(0), -1)
        act1 = self.act_fc(act)

        return act1, x, c1, c2, c3, c4, act

'''
Model 7:
'''
class ResNetANM_8(nn.Module):
    def __init__(self, block, layers,  inchannel=1, num_classes=6):
        super(ResNetANM_8, self).__init__()
        self.inplanes = 128
        self.conv1 = nn.Conv1d(inchannel, 128, kernel_size=7, stride=2, padding=3,
                                 bias=False)

        self.bn1 = nn.BatchNorm1d(128)
        self.relu = nn.ReLU(inplace=True)
        self.maxpool = nn.MaxPool1d(kernel_size=3, stride=2, padding=1)

        self.layer1 = self._make_layer(block, 128, layers[0], stride=1)
        self.layer2 = self._make_layer(block, 128, layers[1], stride=2)
        self.layer3 = self._make_layer(block, 256, layers[2], stride=2)
        self.layer4 = self._make_layer(block, 512, layers[3], stride=2)

        self.pool = nn.AdaptiveAvgPool1d(1)
        self.ACTClassifier = nn.Sequential(
            nn.Conv1d(512 * block.expansion, 512 * block.expansion, kernel_size=3, stride=1, padding=0, bias=False),
            nn.BatchNorm1d(512 * block.expansion),
            nn.ReLU(inplace=True),
            nn.AdaptiveAvgPool1d(1),
        )
        self.act_fc = nn.Linear(512 * block.expansion, num_classes)

        self.attention1 = nn.Sequential(
            nn.Conv1d(128, 1, kernel_size=1),
            nn.Tanh()
        )
        self.attention2 = nn.Sequential(
            nn.Conv1d(128, 1, kernel_size=1),
            nn.Tanh()
        )
        self.attention3 = nn.Sequential(
            nn.Conv1d(256, 1, kernel_size=1),
            nn.Tanh()
        )
        self.attention4 = nn.Sequential(
            nn.Conv1d(512, 1, kernel_size=1, stride=1, padding=0),
            nn.Tanh()
        )

    def _make_layer(self, block, planes, blocks, stride=1):
        downsample = None
        if stride != 1 or self.inplanes != planes * block.expansion:
            downsample = nn.Sequential(
                conv1x1(self.inplanes, planes * block.expansion, stride),
                nn.BatchNorm1d(planes * block.expansion),
            )

        layers = []
        layers.append(block(self.inplanes, planes, stride, downsample))
        self.inplanes = planes * block.expansion
        for _ in range(1, blocks):
            layers.append(block(self.inplanes, planes))

        return nn.Sequential(*layers)


    def _upsample_add(self, x, y):
        '''Upsample and add two feature maps.
        Args:
          x: (Variable) top feature map to be upsampled.
          y: (Variable) lateral feature map.
        Returns:
          (Variable) added feature map.
        Note in PyTorch, when input size is odd, the upsampled feature map
        with `F.upsample(..., scale_factor=2, mode='nearest')`
        maybe not equal to the lateral feature map size.
        e.g.
        original input size: [N,_,15,15] ->
        conv2d feature map size: [N,_,8,8] ->
        upsampled feature map size: [N,_,16,16]
        So we choose bilinear upsample which supports arbitrary output sizes.
        '''
        _,_,L = y.size()
        return F.interpolate(x, size=L) + y

    def forward(self, x):
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.maxpool(x)

        c1 = self.layer1(x)
        # a1 = self.attention1(c1)
        # a1 = a1.expand(c1.shape[0], c1.shape[1], c1.shape[2])
        # p1 = self.pool(c1)
        # p1 = p1.expand(c1.shape[0], c1.shape[1], c1.shape[2])
        # c11 = p1*a1
        # c1 = c1 + c11

        c2 = self.layer2(c1)
        a2 = self.attention2(c2)
        a2 = a2.expand(c2.shape[0], c2.shape[1], c2.shape[2])
        p2 = self.pool(c2)
        p2 = p2.expand(c2.shape[0], c2.shape[1], c2.shape[2])
        c22 = p2*a2
        c2 = c2 + c22

        c3 = self.layer3(c2)
        # a3 = self.attention3(c3)
        # a3 = a3.expand(c3.shape[0], c3.shape[1], c3.shape[2])
        # p3 = self.pool(c3)
        # p3 = p3.expand(c3.shape[0], c3.shape[1], c3.shape[2])
        # c33 = p3*a3
        # c3 = c3 + c33

        c4 = self.layer4(c3)
        # a4 = self.attention4(c4)
        # a4 = a4.expand(c4.shape[0], c4.shape[1], c4.shape[2])
        # p4 = self.pool(c4)
        # p4 = p4.expand(c4.shape[0], c4.shape[1], c4.shape[2])
        # c44 = p4*a4
        # c4 = c4 + c44

        #act = self.pool(c4)
        act = self.ACTClassifier(c4)
        act = act.view(act.size(0), -1)
        act1 = self.act_fc(act)

        return act1, x, c1, c2, c3, c4, act


'''
Model 8:
'''
class ResNetANM_9(nn.Module):
    def __init__(self, block, layers,  inchannel=1, num_classes=6):
        super(ResNetANM_9, self).__init__()
        self.inplanes = 128
        self.conv1 = nn.Conv1d(inchannel, 128, kernel_size=7, stride=2, padding=3,
                                 bias=False)

        self.bn1 = nn.BatchNorm1d(128)
        self.relu = nn.ReLU(inplace=True)
        self.maxpool = nn.MaxPool1d(kernel_size=3, stride=2, padding=1)

        self.layer1 = self._make_layer(block, 128, layers[0], stride=1)
        self.layer2 = self._make_layer(block, 128, layers[1], stride=2)
        self.layer3 = self._make_layer(block, 256, layers[2], stride=2)
        self.layer4 = self._make_layer(block, 512, layers[3], stride=2)

        self.pool = nn.AdaptiveAvgPool1d(1)
        self.ACTClassifier = nn.Sequential(
            nn.Conv1d(512 * block.expansion, 512 * block.expansion, kernel_size=3, stride=1, padding=0, bias=False),
            nn.BatchNorm1d(512 * block.expansion),
            nn.ReLU(inplace=True),
            nn.AdaptiveAvgPool1d(1),
        )
        self.act_fc = nn.Linear(512 * block.expansion, num_classes)

        self.attention1 = nn.Sequential(
            nn.Conv1d(128, 1, kernel_size=1),
            nn.Tanh()
        )
        self.attention2 = nn.Sequential(
            nn.Conv1d(128, 1, kernel_size=1),
            nn.Tanh()
        )
        self.attention3 = nn.Sequential(
            nn.Conv1d(256, 1, kernel_size=1),
            nn.Tanh()
        )
        self.attention4 = nn.Sequential(
            nn.Conv1d(512, 1, kernel_size=1, stride=1, padding=0),
            nn.Tanh()
        )

    def _make_layer(self, block, planes, blocks, stride=1):
        downsample = None
        if stride != 1 or self.inplanes != planes * block.expansion:
            downsample = nn.Sequential(
                conv1x1(self.inplanes, planes * block.expansion, stride),
                nn.BatchNorm1d(planes * block.expansion),
            )

        layers = []
        layers.append(block(self.inplanes, planes, stride, downsample))
        self.inplanes = planes * block.expansion
        for _ in range(1, blocks):
            layers.append(block(self.inplanes, planes))

        return nn.Sequential(*layers)


    def _upsample_add(self, x, y):
        '''Upsample and add two feature maps.
        Args:
          x: (Variable) top feature map to be upsampled.
          y: (Variable) lateral feature map.
        Returns:
          (Variable) added feature map.
        Note in PyTorch, when input size is odd, the upsampled feature map
        with `F.upsample(..., scale_factor=2, mode='nearest')`
        maybe not equal to the lateral feature map size.
        e.g.
        original input size: [N,_,15,15] ->
        conv2d feature map size: [N,_,8,8] ->
        upsampled feature map size: [N,_,16,16]
        So we choose bilinear upsample which supports arbitrary output sizes.
        '''
        _,_,L = y.size()
        return F.interpolate(x, size=L) + y

    def forward(self, x):
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.maxpool(x)

        c1 = self.layer1(x)
        # a1 = self.attention1(c1)
        # a1 = a1.expand(c1.shape[0], c1.shape[1], c1.shape[2])
        # p1 = self.pool(c1)
        # p1 = p1.expand(c1.shape[0], c1.shape[1], c1.shape[2])
        # c11 = p1*a1
        # c1 = c1 + c11

        c2 = self.layer2(c1)
        # a2 = self.attention2(c2)
        # a2 = a2.expand(c2.shape[0], c2.shape[1], c2.shape[2])
        # p2 = self.pool(c2)
        # p2 = p2.expand(c2.shape[0], c2.shape[1], c2.shape[2])
        # c22 = p2*a2
        # c2 = c2 + c22

        c3 = self.layer3(c2)
        a3 = self.attention3(c3)
        a3 = a3.expand(c3.shape[0], c3.shape[1], c3.shape[2])
        p3 = self.pool(c3)
        p3 = p3.expand(c3.shape[0], c3.shape[1], c3.shape[2])
        c33 = p3*a3
        c3 = c3 + c33

        c4 = self.layer4(c3)
        # a4 = self.attention4(c4)
        # a4 = a4.expand(c4.shape[0], c4.shape[1], c4.shape[2])
        # p4 = self.pool(c4)
        # p4 = p4.expand(c4.shape[0], c4.shape[1], c4.shape[2])
        # c44 = p4*a4
        # c4 = c4 + c44

        #act = self.pool(c4)
        act = self.ACTClassifier(c4)
        act = act.view(act.size(0), -1)
        act1 = self.act_fc(act)

        return act1, x, c1, c2, c3, c4, act



'''
Model 9:
'''
class ResNetANM_7(nn.Module):
    def __init__(self, block, layers,  inchannel=1, num_classes=6):
        super(ResNetANM_7, self).__init__()
        self.inplanes = 128
        self.conv1 = nn.Conv1d(inchannel, 128, kernel_size=7, stride=2, padding=3,
                                 bias=False)

        self.bn1 = nn.BatchNorm1d(128)
        self.relu = nn.ReLU(inplace=True)
        self.maxpool = nn.MaxPool1d(kernel_size=3, stride=2, padding=1)

        self.layer1 = self._make_layer(block, 128, layers[0], stride=1)
        self.layer2 = self._make_layer(block, 128, layers[1], stride=2)
        self.layer3 = self._make_layer(block, 256, layers[2], stride=2)
        self.layer4 = self._make_layer(block, 512, layers[3], stride=2)

        self.pool = nn.AdaptiveAvgPool1d(1)
        self.ACTClassifier = nn.Sequential(
            nn.Conv1d(512 * block.expansion, 512 * block.expansion, kernel_size=3, stride=1, padding=0, bias=False),
            nn.BatchNorm1d(512 * block.expansion),
            nn.ReLU(inplace=True),
            nn.AdaptiveAvgPool1d(1),
        )
        self.act_fc = nn.Linear(512 * block.expansion, num_classes)

        self.attention1 = nn.Sequential(
            nn.Conv1d(128, 1, kernel_size=1),
            nn.Tanh()
        )
        self.attention2 = nn.Sequential(
            nn.Conv1d(128, 1, kernel_size=1),
            nn.Tanh()
        )
        self.attention3 = nn.Sequential(
            nn.Conv1d(256, 1, kernel_size=1),
            nn.Tanh()
        )
        self.attention4 = nn.Sequential(
            nn.Conv1d(512, 1, kernel_size=1, stride=1, padding=0),
            nn.Tanh()
        )

    def _make_layer(self, block, planes, blocks, stride=1):
        downsample = None
        if stride != 1 or self.inplanes != planes * block.expansion:
            downsample = nn.Sequential(
                conv1x1(self.inplanes, planes * block.expansion, stride),
                nn.BatchNorm1d(planes * block.expansion),
            )

        layers = []
        layers.append(block(self.inplanes, planes, stride, downsample))
        self.inplanes = planes * block.expansion
        for _ in range(1, blocks):
            layers.append(block(self.inplanes, planes))

        return nn.Sequential(*layers)


    def _upsample_add(self, x, y):
        '''Upsample and add two feature maps.
        Args:
          x: (Variable) top feature map to be upsampled.
          y: (Variable) lateral feature map.
        Returns:
          (Variable) added feature map.
        Note in PyTorch, when input size is odd, the upsampled feature map
        with `F.upsample(..., scale_factor=2, mode='nearest')`
        maybe not equal to the lateral feature map size.
        e.g.
        original input size: [N,_,15,15] ->
        conv2d feature map size: [N,_,8,8] ->
        upsampled feature map size: [N,_,16,16]
        So we choose bilinear upsample which supports arbitrary output sizes.
        '''
        _,_,L = y.size()
        return F.interpolate(x, size=L) + y

    def forward(self, x):
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.maxpool(x)

        c1 = self.layer1(x)
        # a1 = self.attention1(c1)
        # a1 = a1.expand(c1.shape[0], c1.shape[1], c1.shape[2])
        # p1 = self.pool(c1)
        # p1 = p1.expand(c1.shape[0], c1.shape[1], c1.shape[2])
        # c11 = p1*a1
        # c1 = c1 + c11

        c2 = self.layer2(c1)
        # a2 = self.attention2(c2)
        # a2 = a2.expand(c2.shape[0], c2.shape[1], c2.shape[2])
        # p2 = self.pool(c2)
        # p2 = p2.expand(c2.shape[0], c2.shape[1], c2.shape[2])
        # c22 = p2*a2
        # c2 = c2 + c22

        c3 = self.layer3(c2)
        # a3 = self.attention3(c3)
        # a3 = a3.expand(c3.shape[0], c3.shape[1], c3.shape[2])
        # p3 = self.pool(c3)
        # p3 = p3.expand(c3.shape[0], c3.shape[1], c3.shape[2])
        # c33 = p3*a3
        # c3 = c3 + c33

        c4 = self.layer4(c3)
        a4 = self.attention4(c4)
        a4 = a4.expand(c4.shape[0], c4.shape[1], c4.shape[2])
        p4 = self.pool(c4)
        p4 = p4.expand(c4.shape[0], c4.shape[1], c4.shape[2])
        c44 = p4*a4
        c4 = c4 + c44

        #act = self.pool(c4)
        act = self.ACTClassifier(c4)
        act = act.view(act.size(0), -1)
        act1 = self.act_fc(act)

        return act1, x, c1, c2, c3, c4, act
