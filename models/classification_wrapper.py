import random
import torch
import torch.nn as nn


__all__ = [
    'classification_wrapper'
]


class Head(nn.Module):
    def __init__(self, input_dim, proj_dims):
        super(Head, self).__init__()
        if not isinstance(proj_dims, list):
            proj_dims = [proj_dims]

        projection = []
        for i, d in enumerate(proj_dims):
            projection += [nn.Linear(input_dim, d)]
            input_dim = d
            if i < len(proj_dims)-1:
                projection += [nn.ReLU(inplace=True)]
        self.projection = nn.Sequential(*projection)
        self.out_dim = proj_dims[-1]

    def forward(self, x):
        return self.projection(x)


class ClassificationWrapper(nn.Module):
    def __init__(self, model, outp_dim=128):
        super(ClassificationWrapper, self).__init__()
        self.model = model

        if outp_dim is not None:
            self.classifier = Head(model.out_dim, outp_dim)
            self.out_dim = self.classifier.out_dim
        else:
            self.classifier = None
            self.out_dim = model.out_dim

    def forward(self, x):
        x = self.model(x)
        x = x.view(x.shape[0], x.shape[1])
        if self.classifier is not None:
            x = self.classifier(x)

        return x


def classification_wrapper(backbone, backbone_args, outp_dim=128, checkpoint=None):
    import models
    assert backbone in models.__dict__, 'Unknown model architecture'
    model = models.__dict__[backbone](**backbone_args)

    model = ClassificationWrapper(model, outp_dim=outp_dim)
    if checkpoint is not None:
        ckp = torch.load(checkpoint, map_location='cpu')
        if model.out_dim != ckp['model']['module.classifier.projection.0.weight'].shape[0]:
            ckp['model']['module.classifier.projection.0.weight'] = model.classifier.projection[0].weight
            ckp['model']['module.classifier.projection.0.bias'] = model.classifier.projection[0].bias
        nn.DataParallel(model).load_state_dict(ckp['model'])

    return model


def main():
    import utils.main_utils as utils
    import yaml
    import sys
    import GPUtil
    sys.path.insert(0, '.')

    cfg = yaml.safe_load(open('configs/main/mc-avc/l3-mc100-Fcat-C1-100k.yaml'))
    model = classification_wrapper(**cfg['model']['args'])
    model.cuda()
    model.train()
    print(model)
    print(utils.parameter_description(model))

    # Dummy data
    dummy_video = torch.ones((100, 8, 3, 112, 112)).cuda()
    dummy_audio = torch.ones((100, 1, 200, 257)).cuda()

    logits, labels = model(dummy_video, dummy_audio)
    criterion = torch.nn.CrossEntropyLoss()
    loss = criterion(logits, labels)
    print(GPUtil.getGPUs()[0].memoryUsed)
    loss.backward()
    print(logits.shape)
    print(labels.shape)


if __name__ == '__main__':
    main()