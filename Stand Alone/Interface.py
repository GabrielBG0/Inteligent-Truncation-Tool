from Spliter import spliter

def main():
    print('Welcome to the Spliter tool')
    origin = input('Origin file: ')
    destination = input('Destination folder: ')
    size = int(input('Sub group size: '))
    spliter(origin, destination, size)

if __name__ == '__main__':
    main()