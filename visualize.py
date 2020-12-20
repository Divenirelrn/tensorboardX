import torch
import numpy as np
from torchvision import datasets, models, utils
from tensorboardX import SummaryWriter

writer = SummaryWriter()
resnet = models.resnet18(pretrained=False)

for n_iter in range(100):
    dummy_s1 = torch.randn(1)
    dummy_s2 = torch.randn(1)
    writer.add_scalar('scalar1', dummy_s1[0], n_iter)
    writer.add_scalar('scalar2', dummy_s2[0], n_iter)

    writer.add_scalars('group_scalar', {
        'sin': np.sin(n_iter),
        'cos': np.cos(n_iter)
    }, n_iter)

    if n_iter % 10 == 0:
        images = torch.rand(32,3,64,64)
        x = utils.make_grid(images, normalize=True, scale_each=True)
        writer.add_image('Images', x, n_iter)

        writer.add_text('text', 'text in epoch' + str(n_iter), n_iter)

        for name, param in resnet.named_parameters():
            writer.add_histogram(name, param, n_iter)

        writer.add_pr_curve('pr_curve', np.random.randint(2, size=100), np.random.rand(100), n_iter)

writer.add_graph(resnet, torch.randn(64,3,128,128))
dataset = datasets.MNIST('../data', download=True, train=False)
images = dataset.data[:100].float()
labels = dataset.targets[:100]
features = images.view(images.size(0), -1)
writer.add_embedding(features, metadata=labels, label_img=images.unsqueeze(1))

writer.export_scalars_to_json('scalars.json')

writer.close()

