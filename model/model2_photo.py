# reference: https://github.com/thuml/OpenDG-DAML

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.hub import load_state_dict_from_url
from torch.nn.parameter import Parameter

Parameter.fast = None
all = ['ResNet', 'resnet18', 'resnet34', 'resnet50', 'resnet101',
'resnet152']

model_urls = {
'resnet18': 'https://download.pytorch.org/models/resnet18-5c106cde.pth',
'resnet34': 'https://download.pytorch.org/models/resnet34-333f7ec4.pth',
'resnet50': 'https://download.pytorch.org/models/resnet50-19c8e357.pth',
'resnet101': 'https://download.pytorch.org/models/resnet101-5d3b4d8f.pth',
'resnet152': 'https://download.pytorch.org/models/resnet152-b121ed2d.pth',
}



class Linear_fw(nn.Linear):
    def __init__(self, in_features, out_features):
        super(Linear_fw, self).__init__(in_features, out_features)

    def forward(self, x):
        if self.weight.fast is not None and self.bias.fast is not None:
            out = F.linear(x, self.weight.fast,
                           self.bias.fast)
        else:
            out = super(Linear_fw, self).forward(x)
        return out


class Conv2d_fw(nn.Conv2d):
    def __init__(self, in_channels, out_channels, kernel_size, stride=1, padding=0, bias=True):
        super(Conv2d_fw, self).__init__(in_channels, out_channels, kernel_size, stride=stride, padding=padding,
                                        bias=bias)

    def forward(self, x):
        if self.bias is None:
            if self.weight.fast is not None:
                out = F.conv2d(x, self.weight.fast, None, stride=self.stride, padding=self.padding)
            else:
                out = super(Conv2d_fw, self).forward(x)
        else:
            if self.weight.fast is not None and self.bias.fast is not None:
                out = F.conv2d(x, self.weight.fast, self.bias.fast, stride=self.stride, padding=self.padding)
            else:
                out = super(Conv2d_fw, self).forward(x)

        return out


class BatchNorm2d_fw(nn.BatchNorm2d):
    def __init__(self, num_features):
        super(BatchNorm2d_fw, self).__init__(num_features)

    def forward(self, input):
        self._check_input_dim(input)

        if self.momentum is None:
            exponential_average_factor = 0.0
        else:
            exponential_average_factor = self.momentum

        if self.training and self.track_running_stats:
            if self.num_batches_tracked is not None:
                self.num_batches_tracked = self.num_batches_tracked + 1
                if self.momentum is None:
                    exponential_average_factor = 1.0 / float(self.num_batches_tracked)
                else:
                    exponential_average_factor = self.momentum

        """ Decide whether the mini-batch stats should be used for normalization rather than the buffers.
        Mini-batch stats are used in training mode, and in eval mode when buffers are None.
        """
        if self.training:
            bn_training = True
        else:
            bn_training = (self.running_mean is None) and (self.running_var is None)

        """Buffers are only updated if they are to be tracked and we are in training mode. Thus they only need to be
        passed when the update should occur (i.e. in training mode when they are tracked), or when buffer stats are
        used for normalization (i.e. in eval mode when buffers are not None).
        """

        if self.weight.fast is not None and self.bias.fast is not None:
            return F.batch_norm(
            input,
            self.running_mean if not self.training or self.track_running_stats else None,
            self.running_var if not self.training or self.track_running_stats else None,
            self.weight.fast, self.bias.fast, bn_training, exponential_average_factor, self.eps)
        else:
            return F.batch_norm(
                input,
                self.running_mean if not self.training or self.track_running_stats else None,
                self.running_var if not self.training or self.track_running_stats else None,
                self.weight, self.bias, bn_training, exponential_average_factor, self.eps)


class BasicBlock(nn.Module):
    expansion = 1

    def __init__(self, inplanes, planes, stride=1, downsample=None):
        super(BasicBlock, self).__init__()
        self.conv1 = Conv2d_fw(in_channels=inplanes, out_channels=planes, kernel_size=3,
                               stride=stride, padding=1, bias=False)
        self.bn1 = BatchNorm2d_fw(planes)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = Conv2d_fw(in_channels=planes, out_channels=planes, kernel_size=3,
                               stride=1, padding=1, bias=False)
        self.bn2 = BatchNorm2d_fw(planes)
        self.downsample = downsample
        self.stride = stride

    def forward(self, x):
        residual = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)

        if self.downsample is not None:
            residual = self.downsample(x)

        out += residual
        out = self.relu(out)

        return out


class Bottleneck(nn.Module):
    expansion = 4

    def __init__(
        self,
        inplanes,
        planes,
        stride=1,
        downsample=None,
    ):
        super(Bottleneck, self).__init__()
        self.conv1 = Conv2d_fw(in_channels=inplanes, out_channels=planes, kernel_size=1, stride=1, bias=False)
        self.bn1 = BatchNorm2d_fw(planes)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = Conv2d_fw(in_channels=planes, out_channels=planes, kernel_size=3, stride=stride, padding=1, bias=False)
        self.bn2 = BatchNorm2d_fw(planes)
        self.conv3 = Conv2d_fw(in_channels=planes, out_channels=planes*self.expansion, kernel_size=1, stride=1, bias=False)
        self.bn3 = BatchNorm2d_fw(planes*self.expansion)
        self.downsample = downsample
        self.stride = stride

    def forward(self, x):
        residual = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)
        out = self.relu(out)

        out = self.conv3(out)
        out = self.bn3(out)

        if self.downsample is not None:
            residual = self.downsample(x)

        out += residual
        out = self.relu(out)

        return out

class ResNetFast(nn.Module):
    def __init__(self, block, layers):
        self.inplanes = 64
        super(ResNetFast, self).__init__()
        self.conv1 = Conv2d_fw(3, 64, kernel_size=7, stride=2, padding=3,
                               bias=False)
        self.bn1 = BatchNorm2d_fw(64)
        self.relu = nn.ReLU(inplace=True)
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        self.layer1 = self._make_layer(block, 64, layers[0])
        self.layer2 = self._make_layer(block, 128, layers[1], stride=2)
        self.layer3 = self._make_layer(block, 256, layers[2], stride=2)
        self.layer4 = self._make_layer(block, 512, layers[3], stride=2)


        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        self.fc = nn.Linear(512 * block.expansion, 1000)

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)

    def _make_layer(self, block, planes, blocks, stride=1):
        downsample = None
        if stride != 1 or self.inplanes != planes * block.expansion:
            downsample = nn.Sequential(
                Conv2d_fw(self.inplanes, planes * block.expansion,
                          kernel_size=1, stride=stride, bias=False),
                BatchNorm2d_fw(planes * block.expansion),
            )

        layers = []
        layers.append(block(self.inplanes, planes, stride, downsample))
        self.inplanes = planes * block.expansion
        for i in range(1, blocks):
            layers.append(block(self.inplanes, planes))

        return nn.Sequential(*layers)

    def forward(self, x):
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.maxpool(x)

        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)

        x = self.avgpool(x)

        x = torch.flatten(x, 1)   
        return x

class MMD_loss(nn.Module):
    def __init__(self, kernel_mul = 2.0, kernel_num = 5):
        super(MMD_loss, self).__init__()
        self.kernel_num = kernel_num
        self.kernel_mul = kernel_mul
        self.fix_sigma = None
    def guassian_kernel(self, source, target, kernel_mul=2.0, kernel_num=5, fix_sigma=None):
        n_samples = int(source.size()[0])+int(target.size()[0])
        total = torch.cat([source, target], dim=0)

        total0 = total.unsqueeze(0).expand(int(total.size(0)), int(total.size(0)), int(total.size(1)))
        total1 = total.unsqueeze(1).expand(int(total.size(0)), int(total.size(0)), int(total.size(1)))
        L2_distance = ((total0-total1)**2).sum(2) 
        if fix_sigma:
            bandwidth = fix_sigma
        else:
            bandwidth = torch.sum(L2_distance.data) / (n_samples**2-n_samples)
        bandwidth /= kernel_mul ** (kernel_num // 2)
        bandwidth_list = [bandwidth * (kernel_mul**i) for i in range(kernel_num)]
        kernel_val = [torch.exp(-L2_distance / bandwidth_temp) for bandwidth_temp in bandwidth_list]
        return sum(kernel_val)

    def forward(self, source, target):
        batch_size = int(source.size()[0])
        kernels = self.guassian_kernel(source, target, kernel_mul=self.kernel_mul, kernel_num=self.kernel_num, fix_sigma=self.fix_sigma)
        XX = kernels[:batch_size, :batch_size]
        YY = kernels[batch_size:, batch_size:]
        XY = kernels[:batch_size, batch_size:]
        YX = kernels[batch_size:, :batch_size]
        loss = torch.mean(XX + YY - XY -YX)
        return loss

class ResNetFast_eve(nn.Module):
    def __init__(self, block, layers, num_classes, rb1, rb2):
        super(ResNetFast_eve, self).__init__()
        self.inplanes = 64
        self.num_classes = num_classes
        self.dropout_ratio = 0.1
        self.noise_alpha_mu = 1.0
        self.noise_alpha_std = 0.1
        self.noise_beta_mu = 1.0
        self.noise_beta_std = 0.1
        self.rebiased_factor = 0.001
        self.conv1 = Conv2d_fw(3, 64, kernel_size=7, stride=2, padding=3,
                               bias=False)

        self.bn1 = BatchNorm2d_fw(64)
        self.mmd_loss = MMD_loss()

        self.relu = nn.ReLU(inplace=True)
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        self.layer1 = self._make_layer(block, 64, layers[0])
        self.layer2 = self._make_layer(block, 128, layers[1], stride=2)
        self.layer3 = self._make_layer(block, 256, layers[2], stride=2)
        self.layer4 = self._make_layer(block, 512, layers[3], stride=2)
        #self.layer_rebiased_1 = self._make_layer(block,512,2, stride=2)
        #self.layer_rebiased_2 = self._make_layer(block,512,1,stride=1)
        self.layer_rebiased_1 = self._make_layer(block,512,rb1, stride=2)
        self.layer_rebiased_2 = self._make_layer(block,512,rb2,stride=1)
        self.avgpool = nn.AdaptiveAvgPool2d((2, 2))
        self.maxpool = nn.MaxPool2d((2,2))
        self.layer_norm = nn.BatchNorm2d(512*block.expansion)
        self.dropout= nn.Dropout(p=self.dropout_ratio)
        self.classifier_1 = Linear_fw(512*block.expansion, self.num_classes)
        self.classifier_1_b = Linear_fw(512*block.expansion, self.num_classes*2)
        self.classifier_2 = Linear_fw(512*block.expansion, self.num_classes)
        self.classifier_2_b = Linear_fw(512*block.expansion, self.num_classes*2)
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)
        nn.init.xavier_uniform_(self.classifier_1.weight, .1)
        nn.init.constant_(self.classifier_1.bias, 0.)
        nn.init.xavier_uniform_(self.classifier_1_b.weight, .1)
        nn.init.constant_(self.classifier_1_b.bias, 0.)
        nn.init.xavier_uniform_(self.classifier_2.weight, .1)
        nn.init.constant_(self.classifier_2.bias, 0.)
        nn.init.xavier_uniform_(self.classifier_2_b.weight, .1)
        nn.init.constant_(self.classifier_2_b.bias, 0.)
    def exp_evidence(self, y):
        return torch.exp(torch.clamp(y, -10, 10))

    def edl_loss(self, func, alpha, y):
        S = torch.sum(alpha, dim=1, keepdim=True).cuda()
        if len(y.shape) < 2:
            y = torch.nn.functional.one_hot(y, num_classes=self.num_classes).cuda()
        loss = torch.sum(y * (func(S) - func(alpha)), dim=1, keepdim=True)
        return loss
    def _make_layer(self, block, planes, blocks, stride=1):
        downsample = None
        if stride != 1 or self.inplanes != planes * block.expansion:
            downsample = nn.Sequential(
                Conv2d_fw(self.inplanes, planes * block.expansion,
                          kernel_size=1, stride=stride, bias=False),
                BatchNorm2d_fw(planes * block.expansion),
            )

        layers = []
        layers.append(block(self.inplanes, planes, stride, downsample))
        self.inplanes = planes * block.expansion
        for i in range(1, blocks):
            layers.append(block(self.inplanes, planes))
        return nn.Sequential(*layers)

    def kl_divergence(self, alpha, beta):
        S_alpha = torch.sum(alpha, dim=1, keepdim=True)
        S_beta = torch.sum(beta, dim=1, keepdim=True)
        lnA = torch.lgamma(S_alpha) - torch.sum(torch.lgamma(alpha), dim=1, keepdim=True)
        lnB = torch.lgamma(S_beta) - torch.sum(torch.lgamma(beta), dim=1, keepdim=True)
        dg_term = torch.digamma(alpha) - torch.digamma(S_alpha)
        kl = lnA - lnB + torch.sum((alpha - beta) * dg_term, dim=1, keepdim=True)
        return kl

    def _kernel(self, X, sigma):
        X = X.view(len(X), -1)
        XX = X @ X.t()
        X_sqnorms = torch.diag(XX)
        X_L2 = -2 * XX + X_sqnorms.unsqueeze(1) + X_sqnorms.unsqueeze(0)
        gamma = 1 / (2 * sigma ** 2)

        kernel_XX = torch.exp(-gamma * X_L2)
        return kernel_XX

    def hsic_loss(self, input1, input2, unbiased=False):
        N = len(input1)
        if N < 4:
            return torch.tensor(0.0).to(input1.device)
        import numpy as np
        sigma_x = np.sqrt(input1.size()[1])
        sigma_y = np.sqrt(input2.size()[1])
        kernel_XX = self._kernel(input1, sigma_x)
        kernel_YY = self._kernel(input2, sigma_y)

        if unbiased:
            """Unbiased estimator of Hilbert-Schmidt Independence Criterion
            Song, Le, et al. "Feature selection via dependence maximization." 2012.
            """
            tK = kernel_XX - torch.diag(torch.diag(kernel_XX))
            tL = kernel_YY - torch.diag(torch.diag(kernel_YY))
            hsic = (
                torch.trace(tK @ tL)
                + (torch.sum(tK) * torch.sum(tL) / (N - 1) / (N - 2))
                - (2 * torch.sum(tK, 0).dot(torch.sum(tL, 0)) / (N - 2))
            )
            loss = hsic if self.alternative else hsic / (N * (N - 3))
        else:
            """Biased estimator of Hilbert-Schmidt Independence Criterion
            Gretton, Arthur, et al. "Measuring statistical dependence with Hilbert-Schmidt norms." 2005.
            """
            KH = kernel_XX - kernel_XX.mean(0, keepdim=True)
            LH = kernel_YY - kernel_YY.mean(0, keepdim=True)
            loss = torch.trace(KH @ LH / (N - 1) ** 2)
        return loss
    def forward(self, x, y, train=False):
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.maxpool(x)

        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)

        x1 = self.layer_rebiased_1(x)
        x2 = self.layer_rebiased_2(x)
        #print(x1.shape)
        x1 = self.layer_norm(x1)
        x2 = self.layer_norm(x2)
        x1 = self.avgpool(x1).mean(-1).mean(-1)
        x1 = self.dropout(x1)
        x2 = self.avgpool(x2).mean(-1).mean(-1)
        x2 = self.dropout(x2)
        mmd_loss = self.mmd_loss(x1, x2)
        x1_ = torch.flatten(x1, 1)
        x2_ = torch.flatten(x2, 1) 
        x1 = self.classifier_1(x1_)
        x1_b = self.classifier_1_b(x1_)
        x2 = self.classifier_2(x2_)
        x2_b = self.classifier_2_b(x2_)
        alpha_unbias = self.exp_evidence(x1) + 1
        alpha_bias2 = self.exp_evidence(x2) + 1
        #loss_hsic = -1.0 * self.hsic_loss(alpha_unbias, alpha_bias2)
        if train:
            edl_loss = self.edl_loss(torch.log, alpha_unbias, y) + self.edl_loss(torch.log, alpha_bias2, y) - self.rebiased_factor*mmd_loss
        else:
            edl_loss = 0.0
        
        return x1, x2, x1_b, x2_b, edl_loss  

class ConvNet(nn.Module):
    def __init__(self):
        super(ConvNet, self).__init__()
        self.conv1 = Conv2d_fw(3, 64, kernel_size=3, stride=1, padding=1)
        self.relu = nn.ReLU(inplace=True)
        self.maxpool = nn.MaxPool2d(kernel_size=2)
        self.conv2 = Conv2d_fw(64, 64, kernel_size=3, stride=1, padding=1)
        self.conv3 = Conv2d_fw(64, 64, kernel_size=3, stride=1, padding=1)
        self.conv4 = Conv2d_fw(64, 64, kernel_size=3, stride=1, padding=1)
        self._out_features = 256
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)

    def _check_input(self, x):
        H, W = x.shape[2:]
        assert (
            H == 32 and W == 32
        ), "Input to network must be 32x32, " "but got {}x{}".format(H, W)

    def forward(self, x):
        self._check_input(x)
        x = self.conv1(x)
        x = self.relu(x)
        x = self.maxpool(x)

        x = self.conv2(x)
        x = self.relu(x)
        x = self.maxpool(x)

        x = self.conv3(x)
        x = self.relu(x)
        x = self.maxpool(x)

        x = self.conv4(x)
        x = self.relu(x)
        x = self.maxpool(x)
        x = torch.flatten(x, 1)
        return x

class MutiClassifier_eve(nn.Module):
    def __init__(self, net, num_classes, feature_dim=512):
        super(MutiClassifier_eve, self).__init__()
        self.net = net
        self.num_classes = num_classes

    def forward(self, x, y, train=False):
        x1, x2,x1_b, x2_b, loss = self.net(x, y, train)
        return x1
    def b_forward(self, x, y, train=False):
        x1, x2,x1_b, x2_b, loss = self.net(x, y, train)
        return x1
    def c_forward(self, x, y, train=False):
        x1, x2, x1_b, x2_b, loss = self.net(x, y, train)
        #print(x1_b.view(x1.size(0), 2, -1)[:,1,:].shape)
        return x1, x2, x1_b.view(x1.size(0), 2, -1), x2_b.view(x1.size(0), 2, -1), loss



class MutiClassifier(nn.Module):
    def __init__(self, net, num_classes, feature_dim=512):
        super(MutiClassifier, self).__init__()
        self.net = net
        self.num_classes = num_classes
        self.classifier = Linear_fw(feature_dim, self.num_classes)
        self.b_classifier = Linear_fw(feature_dim, self.num_classes*2)
        nn.init.xavier_uniform_(self.classifier.weight, .1)
        nn.init.constant_(self.classifier.bias, 0.)
        nn.init.xavier_uniform_(self.b_classifier.weight, .1)
        nn.init.constant_(self.b_classifier.bias, 0.)

    def forward(self, x):
        x = self.net(x)
        x = self.classifier(x)
        return x

    def b_forward(self, x):
        x = self.net(x)
        x = self.b_classifier(x)
        return x

    def c_forward(self, x):
        x = self.net(x)
        x1 = self.classifier(x)
        x2 = self.b_classifier(x)
        return x1, x2


class DomainIndicator(nn.Module):
    def __init__(self, net, num_domain, feature_dim=512):
        super(DomainIndicator, self).__init__()
        self.net = net
        self.classifier = Linear_fw(feature_dim, num_domain)
        nn.init.xavier_uniform_(self.classifier.weight, .1)
        nn.init.constant_(self.classifier.bias, 0.)
        self.sigmoid = nn.Sigmoid()
        self.bias_factor =  0.00001
    def forward(self, x):
        x = self.net(x)
        x = self.classifier(x)
        return self.sigmoid(x) + self.bias_factor
class MutiClassifier_(nn.Module):
    def __init__(self, net, num_classes, feature_dim=512):
        super(MutiClassifier_, self).__init__()
        self.net = net
        self.num_classes = num_classes
        self.b_classifier = Linear_fw(feature_dim, self.num_classes*2)
        nn.init.xavier_uniform_(self.b_classifier.weight, .1)
        nn.init.constant_(self.b_classifier.bias, 0.)

    def forward(self, x):
        x = self.net(x)
        x = self.b_classifier(x)
        x = x.view(x.size(0), 2, -1)
        x = x[:, 1, :]
            
        return x

    def b_forward(self, x):
        x = self.net(x)
        x = self.b_classifier(x)
        return x

    def c_forward(self, x):
        x = self.net(x)   
        x2 = self.b_classifier(x)
        x1 = x2.view(x.size(0), 2, -1)
        x1 = x1[:, 1, :]
        return x1, x2


def resnet18_fast(progress=True, num_classes=None, rb1=1, rb2=1):
    """ResNet-18 model from
    `"Deep Residual Learning for Image Recognition" <https://arxiv.org/pdf/1512.03385.pdf>`_

    Parameters:
        - **pretrained** (bool): If True, returns a model pre-trained on ImageNet
        - **progress** (bool): If True, displays a progress bar of the download to stderr
    """
    model = ResNetFast_eve(BasicBlock, [2, 2, 2, 2],num_classes=num_classes, rb1=rb1, rb2=rb2)
    state_dict = load_state_dict_from_url(model_urls['resnet18'],
                                          progress=progress)
    model.load_state_dict(state_dict, strict=False)
    #del model.fc

    return model


def resnet18_fast_origin(progress=True):
    """ResNet-18 model from
    `"Deep Residual Learning for Image Recognition" <https://arxiv.org/pdf/1512.03385.pdf>`_

    Parameters:
        - **pretrained** (bool): If True, returns a model pre-trained on ImageNet
        - **progress** (bool): If True, displays a progress bar of the download to stderr
    """
    model = ResNetFast(BasicBlock, [2, 2, 2, 2])
    state_dict = load_state_dict_from_url(model_urls['resnet18'],
                                          progress=progress)
    model.load_state_dict(state_dict, strict=False)
    #del model.fc

    return model


def resnet50_fast(progress=True, num_classes=None):
    model = ResNetFast_eve(Bottleneck, [3, 4, 6, 3],num_classes=num_classes)
    state_dict = load_state_dict_from_url(model_urls['resnet50'],
                                          progress=progress)
    model.load_state_dict(state_dict, strict=False)


    return model


def resnet152(progress=True, num_classes=None):
    model = ResNetFast_eve(Bottleneck, [3, 8, 36, 3],num_classes=num_classes)
    state_dict = load_state_dict_from_url(model_urls['resnet152'],
                                          progress=progress)
    model.load_state_dict(state_dict, strict=False)
    return model




def resnet152(progress=True, num_classes=None):
    model = ResNetFast_eve(Bottleneck, [3, 8, 36, 3],num_classes=num_classes)
    state_dict = load_state_dict_from_url(model_urls['resnet152'],
                                          progress=progress)
    model.load_state_dict(state_dict, strict=False)
    return model


    