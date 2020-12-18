import argparse

parser = argparse.ArgumentParser()
parser.add_argument('--model', type=str, default='custom', choices=['vgg', 'svm', 'ada', 'decision', 'knn'])
parser.add_argument('--gap', type=bool, default=False, help='use Flatten or GAP')
parser.add_argument('--init', type=str, default='he', help='type of initialization', choices=['random', 'he', 'xavier'])
parser.add_argument('--num_epochs', type=int, default=50, help='total epochs')
parser.add_argument('--freeze', type=bool, default=False, help='Freeze or not when using pre-trained models')

config = parser.parse_args()

if __name__ == "__main__":
    print(config)