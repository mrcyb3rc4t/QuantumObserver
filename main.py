import tensorflow as tf
import tensorflow_quantum as tfq

import cirq
import sympy
import numpy as np
import seaborn as sns
import collections
import bitstring

# visualization tools
# %matplotlib inline
import matplotlib.pyplot as plt
from cirq.contrib.svg import SVGCircuit


def load_own_data(file_name, vulnerability):
    with open(file_name, "r") as f:
        temp = f.readlines()
        x = []
        for xi in temp:
            if xi.split(',')[-1] == 'normal.\n' or xi.split(',')[-1] == (vulnerability + '.\n'):
                x.append(xi.split(','))
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
                        if line[i][:-2] == "normal":
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
                        if line[i][:-2] == "normal":
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


# загружаем датасеты MNIST

(x_train, y_train), (x_test, y_test) = load_own_data("kddcup.data_10_percent_corrected", "neptune")

# Rescale the images from [0,255] to the [0.0,1.0] range.
# x_train, x_test = x_train[..., np.newaxis]/255.0, x_test[..., np.newaxis]/255.0

DATA_SLICE = 10000

x_train = x_train[:DATA_SLICE]
y_train = y_train[:DATA_SLICE]

x_test = x_test[:DATA_SLICE]
y_test = y_test[:DATA_SLICE]

print("Number of original training examples:", len(x_train))
print("Number of original test examples:", len(x_test))


# оставим только 3 и 6


# def filter_36(x, y):
#     keep = (y == 3) | (y == 6)
#     x, y = x[keep], y[keep]
#     y = y == 3
#     return x, y
#
#
# x_train, y_train = filter_36(x_train, y_train)
# x_test, y_test = filter_36(x_test, y_test)
#
# print("Number of filtered training examples:", len(x_train))
# print("Number of filtered test examples:", len(x_test))

# print(y_train[0])

# plt.imshow(x_train[0, :, :, 0])
# plt.colorbar()
# plt.show(block=True)

# Размер изображения 28x28 слишком велик для современных квантовых компьютеров. Измените размер изображения до 4x4:

# x_train_small = tf.image.resize(x_train, (4, 4)).numpy()
# x_test_small = tf.image.resize(x_test, (4, 4)).numpy()

# print(y_train[0])
#
# plt.imshow(x_train_small[0, :, :, 0], vmin=0, vmax=1)
# plt.colorbar()
# plt.show()

# удаление противоречивых примеров согласно какой-то теории


# def remove_contradicting(xs, ys):
#     mapping = collections.defaultdict(set)
#     orig_x = {}
#     # Determine the set of labels for each unique image:
#     for x, y in zip(xs, ys):
#         orig_x[tuple(x.flatten())] = x
#         mapping[tuple(x.flatten())].add(y)
#
#     new_x = []
#     new_y = []
#     for flatten_x in mapping:
#         x = orig_x[flatten_x]
#         labels = mapping[flatten_x]
#         if len(labels) == 1:
#             new_x.append(x)
#             new_y.append(next(iter(labels)))
#         else:
#             # Throw out images that match more than one label.
#             pass
#
#     num_uniq_3 = sum(1 for value in mapping.values()
#                      if len(value) == 1 and True in value)
#     num_uniq_6 = sum(1 for value in mapping.values()
#                      if len(value) == 1 and False in value)
#     num_uniq_both = sum(1 for value in mapping.values() if len(value) == 2)
#
#     print("Number of unique images:", len(mapping.values()))
#     print("Number of unique 3s: ", num_uniq_3)
#     print("Number of unique 6s: ", num_uniq_6)
#     print("Number of unique contradicting labels (both 3 and 6): ", num_uniq_both)
#     print()
#     print("Initial number of images: ", len(xs))
#     print("Remaining non-contradicting unique images: ", len(new_x))
#
#     return np.array(new_x), np.array(new_y)


# x_train_nocon, y_train_nocon = remove_contradicting(x_train_small, y_train)


def convert_data_to_binary(data):
    result = []

    line_size = 0

    for line in data:
        new_line = []
        for i in range(len(line)):
            if i == 1 or i == 2 or i == 3:
                a_byte_array = bytearray(line[i], "utf8")

                byte_list = []

                for byte in a_byte_array:
                    binary_representation = bin(byte)
                    byte_list.append(binary_representation)

                for byte in byte_list:
                    for bit in byte[2:]:
                        line_size += 1
                        new_line.append(int(bit))

            elif i == 24 or i == 25 or i == 26 or i == 27 or i == 28 or i == 29 or i == 30 or i == 33 or i == 34 or i == 35 or i == 36 or i == 37 or i == 38 or i == 39 or i == 40:
                bits = bitstring.BitArray(float=line[i], length=32)
                for bit in bits.bin:
                    line_size += 1
                    new_line.append(int(bit))
            else:
                for bit in bin(line[i])[2:]:
                    line_size += 1
                    new_line.append(int(bit))

        result.append(new_line)

    return result


THRESHOLD = 0.5

x_train_bin = convert_data_to_binary(x_train)
x_test_bin = convert_data_to_binary(x_test)

print("Convert to binary done")


def pack_data(data):
    result = []
    for line in data:
        new_line = []
        for i in range(0, 160, 8):
            new_line.append(line[i])
        result.append(new_line)

    return result


# _ = remove_contradicting(x_train_bin, y_train_nocon)

x_train_bin = pack_data(x_train_bin)
x_test_bin = pack_data(x_test_bin)

# самый примитивный перевод в кубиты


def convert_to_circuit(data):
    qubits = cirq.GridQubit.rect(1, 20)
    circuit_t = cirq.Circuit()
    for i in range(20):
        if data[i]:
            circuit_t.append(cirq.X(qubits[i]))
    return circuit_t


print("Starting convert to circuit...")

# x_train_circ = [convert_to_circuit(x) for x in x_train_bin]
x_train_circ = [convert_to_circuit(x) for x in x_train_bin]
print("train done")
x_test_circ = [convert_to_circuit(x) for x in x_test_bin]
print("test_done")

print("... convert to circuits done")

SVGCircuit(x_train_circ[0])

# bin_img = x_train_bin[0, :, :, 0]
# indices = np.array(np.where(bin_img)).T

x_train_tfcirc = tfq.convert_to_tensor(x_train_circ)
x_test_tfcirc = tfq.convert_to_tensor(x_test_circ)


# построение модельной схемы
# В следующем примере показан многоуровневый подход.
# Каждый уровень использует n экземпляров одного и того же логического элемента,
# причем каждый из кубитов данных действует на считываемый кубит.


class CircuitLayerBuilder:
    def __init__(self, data_qubits, readout):
        self.data_qubits = data_qubits
        self.readout = readout

    def add_layer(self, circuit_t, gate, prefix):
        for i, qubit in enumerate(self.data_qubits):
            symbol = sympy.Symbol(prefix + '-' + str(i))
            circuit_t.append(gate(qubit, self.readout) ** symbol)


demo_builder = CircuitLayerBuilder(data_qubits=cirq.GridQubit.rect(1, len(x_train[0])),
                                   readout=cirq.GridQubit(-1, -1))

circuit = cirq.Circuit()
demo_builder.add_layer(circuit, gate=cirq.XX, prefix='xx')
SVGCircuit(circuit)


# построение двухуровневой модели,
# соответствующей размеру цепи данных,
# и включение в нее операции подготовки и считывания.


def create_quantum_model():
    """Create a QNN model circuit and readout operation to go along with it."""
    data_qubits = cirq.GridQubit.rect(1, 20)  # a 4x4 grid.
    readout = cirq.GridQubit(-1, -1)  # a single qubit at [-1,-1]
    circuit_t = cirq.Circuit()

    # Prepare the readout qubit.
    circuit_t.append(cirq.X(readout))
    circuit_t.append(cirq.H(readout))

    builder = CircuitLayerBuilder(
        data_qubits=data_qubits,
        readout=readout)

    # Then add layers (experiment by adding more).
    builder.add_layer(circuit_t, cirq.XX, "xx1")
    builder.add_layer(circuit_t, cirq.ZZ, "zz1")

    # Finally, prepare the readout qubit.
    circuit_t.append(cirq.H(readout))

    return circuit_t, cirq.Z(readout)


model_circuit, model_readout = create_quantum_model()

# Build the Keras model.
model = tf.keras.Sequential([
    # The input is the data-circuit, encoded as a tf.string
    tf.keras.layers.Input(shape=(), dtype=tf.string),
    # The PQC layer returns the expected value of the readout gate, range [-1,1].
    tfq.layers.PQC(model_circuit, model_readout),
])

y_train_hinge = 2.0 * np.array(y_train) - 1.0
y_test_hinge = 2.0 * np.array(y_test) - 1.0


def hinge_accuracy(y_true, y_pred):
    y_true = tf.squeeze(y_true) > 0.0
    y_pred = tf.squeeze(y_pred) > 0.0
    result = tf.cast(y_true == y_pred, tf.float32)

    return tf.reduce_mean(result)


model.compile(
    loss=tf.keras.losses.Hinge(),
    optimizer=tf.keras.optimizers.Adam(),
    metrics=[hinge_accuracy])

print(model.summary())

EPOCHS = 3
BATCH_SIZE = 32

NUM_EXAMPLES = len(x_train_tfcirc)

x_train_tfcirc_sub = x_train_tfcirc[:NUM_EXAMPLES]
y_train_hinge_sub = y_train_hinge[:NUM_EXAMPLES]

qnn_history = model.fit(
    x_train_tfcirc_sub, y_train_hinge_sub,
    batch_size=32,
    epochs=EPOCHS,
    verbose=1,
    validation_data=(x_test_tfcirc, y_test_hinge))

qnn_results = model.evaluate(x_test_tfcirc, np.array(y_test))
