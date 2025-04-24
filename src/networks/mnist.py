from tinygrad import Tensor, nn, TinyJit
from tinygrad.nn import datasets


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


def train_mnist(steps=1000):
    print("=== Running MNIST Model ===")
    print("Loading MNIST Dataset...")
    X_train, Y_train, X_test, Y_test = datasets.mnist()
    print("Loaded MNIST Dataset")

    model = MNISTClassifier()

    print("=================")
    optim = nn.optim.Adam(nn.state.get_parameters(model))
    batch_size = 128

    def step_function():
        Tensor.training = True
        samples = Tensor.randint(batch_size, high=X_train.shape[0])
        X, Y = X_train[samples], Y_train[samples]
        optim.zero_grad()
        loss = model(X).sparse_categorical_crossentropy(Y).backward()
        optim.step()
        return loss

    jit_step_function = TinyJit(step_function)

    for step in range(steps):
        loss = jit_step_function()
        if step % 100 == 0:
            Tensor.training = False
            acc = (model(X_test).argmax(axis=1) == Y_test).mean().item()
            print(f"step {step:4d}, loss {loss.item():.2f}, acc {acc * 100.0:.2f}%")
