def spliter(origin, destination, size):
    texts = open(origin, 'r', encoding="utf8").read()
    texts = texts.split('\n')

    for i in range(len(texts)):
        words = texts[i].split(' ')
        words = [*words, *words[:size]]
        with open(destination + '/' + str(i) + '.txt', 'w', encoding="utf8") as file:
            for j in range(len(words) - size):
                file.write(' '.join(words[j:j + size]) + '\n')

spliter('fake.txt', 'Results', 50)