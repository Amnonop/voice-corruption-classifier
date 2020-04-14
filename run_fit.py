from configuration import *

CONFIG_FILENAME = 'configs/config.json'

def main():
    config = Configuration(CONFIG_FILENAME)
    print(config.csv_filename)

if __name__ == '__main__':
    main()