from tinygrad import Tensor, nn, TinyJit
from lib.datasets import load_dataset
from networks.cifar_classifier import Cifar10Classifier


def train_cifar10_classifier(steps=1000):
    print("=== Running Cifar10 Classifier Model ===")
    print("Loading Cifar10 Dataset...")
    X_train, Y_train, X_test, Y_test = load_dataset("cifar")
    print("Loaded Cifar10 Dataset")

    model = Cifar10Classifier()

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

    loss_final = jit_step_function()
    Tensor.training = False
    acc = (model(X_test).argmax(axis=1) == Y_test).mean().item()
    print(f"step {steps:4d}, loss {loss_final.item():.2f}, acc {acc * 100.0:.2f}%")
