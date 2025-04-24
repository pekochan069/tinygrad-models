from tinygrad import Tensor, nn


class Cifar10Classifier:
    def __init__(self):
        self.layer1 = nn.Conv2d(3, 32, kernel_size=(3, 3))
        self.layer2 = nn.Conv2d(32, 64, kernel_size=(3, 3))
        self.layer3 = nn.Conv2d(64, 128, kernel_size=(3, 3))
        self.layer4 = nn.Linear(512, 10)

    def __call__(self, x: Tensor) -> Tensor:
        x = self.layer1(x).swish().max_pool2d((2, 2))
        x = self.layer2(x).swish().max_pool2d((2, 2))
        x = self.layer3(x).swish().max_pool2d((2, 2))
        x = self.layer5(x.flatten(1).dropout(0.5))
        return x
