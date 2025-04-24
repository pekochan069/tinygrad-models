from tinygrad import Tensor, nn


class MNISTClassifier:
    def __init__(self):
        self.layer1 = nn.Conv2d(1, 32, kernel_size=(3, 3))
        self.layer2 = nn.Conv2d(32, 64, kernel_size=(3, 3))
        self.layer3 = nn.Linear(1600, 10)

    def __call__(self, x: Tensor) -> Tensor:
        x = self.layer1(x).relu().max_pool2d((2, 2))
        x = self.layer2(x).relu().max_pool2d((2, 2))
        x = self.layer3(x.flatten(1).dropout(0.5))
        return x
