def load_own_data(file_name, vulnerability, amount_of_examples):
    with open(file_name, "r") as f:
        temp = f.readlines()
        x = []
        normal_count = 0
        vulner_count = 0
        for xi in temp:
            if (xi.split(',')[-1] == 'normal.\n') and (normal_count <= amount_of_examples):
                x.append(xi.split(','))
                normal_count += 1
            if (xi.split(',')[-1] == (vulnerability + '.\n')) and (vulner_count <= amount_of_examples):
                x.append(xi.split(','))
                vulner_count += 1
            # if (xi.split(',')[-1] == 'normal.\n' or xi.split(',')[-1] == (vulnerability + '.\n')):
            #     x.append(xi.split(','))
        temp = 0
        # x = [if () xi.split(',') for xi in x]

        x_train = x[:int((len(x) / 2))]
        x_test = x[int((len(x) / 2)):]
        y_train = []
        y_test = []

        for line in x_train:
            for i in range(len(line)):
                if i == 1 or i == 2 or i == 3 or i == 41:
                    if i == 41:
                        if (line[i][:-2] == "normal"):
                            y_train.append(True)
                        else:
                            y_train.append(False)
                        # y_train.append(line[i][:-2])
                elif i == 24 or i == 25 or i == 26 or i == 27 or i == 28 or i == 29 or i == 30 or i == 33 or i == 34 or i == 35 or i == 36 or i == 37 or i == 38 or i == 39 or i == 40:
                    line[i] = float(line[i])
                else:
                    line[i] = int(line[i])
            # удаляем последний элемент (метку)
            line.pop()

        for line in x_test:
            for i in range(len(line)):
                if i == 1 or i == 2 or i == 3 or i == 41:
                    if i == 41:
                        if (line[i][:-2] == "normal"):
                            y_test.append(True)
                        else:
                            y_test.append(False)
                        # y_test.append(line[i][:-2])
                elif i == 24 or i == 25 or i == 26 or i == 27 or i == 28 or i == 29 or i == 30 or i == 33 or i == 34 or i == 35 or i == 36 or i == 37 or i == 38 or i == 39 or i == 40:
                    line[i] = float(line[i])
                else:
                    line[i] = int(line[i])
            # удаляем последний элемент (метку)
            line.pop()

    return (x_train, y_train), (x_test, y_test)


(x_train, y_train), (x_test, y_test) = load_own_data(
    "kddcup.data_10_percent_corrected", "neptune", 100)
print(x_train[0], y_train[0], x_test[0], y_test[0])
