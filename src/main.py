from tinygrad import Device

from networks.mnist import train_mnist


def main():
    print(f"Current Device: {Device.DEFAULT}")
    train_mnist()


if __name__ == "__main__":
    main()
