from RoverBrain import *
import argparse

parser = argparse.ArgumentParser(description='Rover Curiosity Parameters')
parser.add_argument('--driver')
parser.add_argument('--save_dict', default=False)
args = parser.parse_args()
print(args.driver)
print(args.save_dict)

if __name__ == '__main__':
    rover = RoverBrain(args.driver, args.save_dict)
