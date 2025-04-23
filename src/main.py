from tinygrad import Device

from networks.mnist import run_mnist


def main():
    print(f"Current Device: {Device.DEFAULT}")
    run_mnist()


if __name__ == "__main__":
    main()
