# Data sources and artifact policy

## Public repository

The default demo creates geometric images with Pillow. These assets are
generated locally and carry no third-party dataset restrictions.

## Tiny ImageNet and ImageNet-family data

The historical experiments used a subset of Tiny ImageNet. ImageNet states
that it does not own the copyright to the images and provides access for
non-commercial research and educational use under its stated conditions.
Accordingly, no ImageNet-derived images or archives are redistributed here.

Obtain any research dataset from its official source, review its current terms,
and keep it in an ignored local directory such as `data/`.

## Pretrained model weights

Torchvision VGG16 and LPIPS weights are fetched only when their respective
features are requested. Those artifacts retain their upstream licenses and
terms. The synthetic `avg_rgb` demo requires neither model.
