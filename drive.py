from RoverBrain import *
import argparse

parser = argparse.ArgumentParser(description='human driver')
parser.add_argument('--driver')
args = parser.parse_args()
print(args.driver)

if __name__ == '__main__':
    rover = RoverBrain(args.driver)
