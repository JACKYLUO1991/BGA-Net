from networks.backbone import resnet, xception, drn, mobilenet


def build_backbone(backbone, output_stride, bn):
    if backbone == 'resnet':
        return resnet.ResNet101(output_stride, bn)
    elif backbone == 'xception':
        return xception.AlignedXception(output_stride, bn)
    elif backbone == 'drn':
        return drn.drn_d_54(bn)
    elif backbone == 'mobilenet':
        return mobilenet.MobileNetV2(output_stride, bn)
    else:
        raise NotImplementedError
