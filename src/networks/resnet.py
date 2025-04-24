from tinygrad import Tensor, nn


class ResBlock:
    def __init__(self, filter: int, use_subsample=True, use_shortcut=True):
        self.use_shortcut = use_shortcut

        m = 0.5 if use_subsample else 1

        self.layer1 = nn.Conv2d(m * filter, filter, 3, int(1 / m), 1, bias=False)
        self.bn1 = nn.BatchNorm2d(filter)
        self.layer2 = nn.Conv2d(filter, filter, 3, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(filter)

    def __call__(self, x: Tensor):
        if self.use_shortcut:
            i = x

        x = self.layer1(x)
        x = self.bn1(x).celu()
        x = self.layer2(x)
        x = self.bn2(x)

        if self.use_shortcut:
            if x.shape != i.shape:
                d = i.avg_pool2d(1, 2)
                x = x + d.cat(d.mul(0), dim=1)
            else:
                x = x + i

        return x.celu()


class Resnet:
    def __init__(self, n: int, use_shortcut=True):
        self.layer1 = nn.Conv2d(3, 16, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(16)

        self.blocks1 = [ResBlock(16, False, use_shortcut) for _ in range(n)]
        self.blocks2 = [ResBlock(32, True, use_shortcut)].extend(
            [ResBlock(32, False, use_shortcut) for _ in range(n - 1)]
        )
        self.blocks3 = [ResBlock(64, True, use_shortcut)].extend(
            [ResBlock(64, False, use_shortcut) for _ in range(n - 1)]
        )

        self.fc = nn.Linear(64, 10)

    def __call__(self, x: Tensor):
        x = self.layer1(x)
        x = self.bn1(x).celu()

        for block in self.blocks1:
            x = block(x)

        for block in self.blocks2:
            x = block(x)

        for block in self.blocks3:
            x = block(x)

        x = x.avg_pool2d((1, 1)).view(x.size(0))
        x = self.fc(x).softmax()

        return x
