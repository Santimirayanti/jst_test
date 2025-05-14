# ----------Penggunaan Libaries----------

import numpy as np
import math
from tkinter import *
import tkinter.filedialog as fdialog
import tkinter.messagebox as messagebox
import os

# ----------Tahap Inisialisasi----------

# Inisialisasi matriks 7x9
w = 7
h = 9
input_size = w * h
states = np.zeros((w, h))

# Inisialisasi beban dan bias
weights = {}
bias = {}

# Inisialisasi Threshold (teta) = 0, Learning Rate = 1, dan iterasi maksimum = 1000
threshold = 0
LR = 1
max_iterations = 10000

# ----------Fungsi untuk Transformasi File menjadi Biner (0 & 1)----------

def open_file(file):
    result = np.zeros((w,h)) # inisialisai matriks 0
    lines = [line.rstrip('\n') for line in open(file)] # isi dari tiap (x,y) dari file
    for y, line in enumerate(lines):
        for x, ch in enumerate(line):
            result[x, y] = 1 if ch == '*' else 0 # jika '-' maka 1, jika '*' maka 0
    return result

# ----------Algoritma Training----------

def train(input):

    # ----------Step 0----------
    # Inisialisasi beban dan bias
    weights = {}
    bias = {}
    # Atur nilai awal dari beban dan bias = 0
    for key in input:
        weights[key] = np.zeros(input_size)
        bias[key] = 0

    # ----------Step 1----------
    trained = False     # Selama stopping condition adalah false, lakukan step 2-6
    for epoch in range(max_iterations):
        trained = True

        # ----------Step 2----------
        for key, samples in input.items():      # Untuk setiap pasangan training S:T, lakukan step 3-5
            target = {}                         # inisialisasi target
            for letter in input:
                target[letter] = 1 if letter == key else -1     # nilai dari target

            # ----------Step 3----------
            for sample_index, sample in enumerate(samples):     # Xi = Si (input units)

                # ----------Step 4----------
                for letter, t in target.items():
                    y_in = bias[letter]
                    for i in range(input_size):
                        y_in += sample[i] * weights[letter][i]  # y_in += b + sum(xi*wi) (output units)
                    if y_in > threshold:        # jika y_in lebih besar dari theta
                        y = 1
                    elif y_in < -threshold:     # jika y_in kurang dari minus theta
                        y = -1
                    else:
                        y = 0                   # jika y_in diantara minus theta dan theta

                    # ----------Step 5----------
                    if y != t:                  # Updating nilai dari beban dan bias jika terdapat error
                        error = t - y
                        bias[letter] = bias[letter] + LR * error
                        for i in range(input_size):
                            weights[letter][i] = weights[letter][i] + LR * sample[i] * error
                        trained = False

        # ----------Step 6----------
        if trained:  # Test kondisi stopping
            break

     # Output dari training
    return (trained, weights, bias, epoch)

# ----------Implementasi fungsi training terhadap file dari folder train----------

def train_folder():
    global weights, bias
    # Sesuiakan dengan data training
    dir_path = os.getcwd() + '\\data\\'
    data = {}
    for file in os.listdir(dir_path):
        if file.endswith('.txt'):   # hanya menerima input dengan format .txt
            ch = file[0].upper()
            if not ch in data:
                data[ch] = []
            matrix = open_file(dir_path + file) # Ubah file kedalam bentuk matriks
            matrix = matrix.reshape(input_size) # Ubah ke 1 dimensi
            matrix[matrix == 0] = -1     # Ubah ke bentuk bipolar
            data[ch].append(matrix)      # insert train data into dictionary
    trained, weights, bias, epoch = train(data)
    return (epoch)

# ----------Algoritma Testing----------

def test(input):
    found = []
    y_vec = []
    bias_final = []
    weight_final = []

    # ----------Step 0----------
    input[input == 0] = -1  # ubah input vector ke bipolar

    # ----------Step 1----------
    for letter, weight in weights.items():      # Beban didapat dari algoritma training
        y_in = bias[letter]
        bias_final = bias[letter]
        weight_final = weight

        # ----------Step 2----------
        for s, w in zip(input, weight):         # Input units (Xi)

            # ----------Step 3----------
            y_in += s * w                       # hitung nila dari y_in = sum(Xi*Wi)
        if y_in > threshold:                    # jika y_in lebih besar dari theta
            found.append(letter)
            y_vec.append(1)
        elif y_in < -threshold:                 # jika y_in lebih kecil dari min theta
            y_vec.append(-1)
        else:                                   # selain dari yang di atas
            y_vec.append(0)
    return (found, y_vec, bias_final, weight_final)

# ----------Design GUI----------

# Inisialisasi root, frame, dan toolbar
root = Tk()
root.configure(background='white')
root.title('Pengenalan Pola Huruf dengan Perceptron')
root.resizable(False, False)
frame = Frame()
frame.configure(background='white')
frame.pack(padx=10, pady=10)
toolbar = Frame(frame, background='white')
toolbar.pack(fill=X)

# Posisi & Ukuran dari GUI
window_height = 700
window_width = 650
screen_width = root.winfo_screenwidth()
screen_height = root.winfo_screenheight()
x_cordinate = int((screen_width/2) - (window_width/2))
y_cordinate = int((screen_height/2) - (window_height/2))
root.geometry("{}x{}+{}+{}".format(window_width, window_height, x_cordinate, y_cordinate))

# Judul GUI
path = "judul-perceptron.png"
img = PhotoImage(file=path)
label = Label(frame, image=img, background='white').pack(side=TOP)

# Tombol Load
def load_callback():
    global states
    file = fdialog.askopenfilename()
    if file != '':
        states = open_file(file)
    print_grid()
Button(toolbar, text="Load", command = load_callback).pack(side=LEFT)

# Tombol Save
def save_callback():
    file = fdialog.asksaveasfile(mode='w', defaultextension=".txt")
    for y in range(h):
        for x in range(w):
            file.write('.' if states[x,y] == 0 else '*')
        file.write('\n')
    file.close()
Button(toolbar, text="Save", command = save_callback).pack(side=LEFT)

# Tombol Clear
def clear_callback():
    np.ndarray.fill(states, 0)
    print_grid()
    test_result_field_value.set('')
    y_result_field_value.set(0)
Button(toolbar, text="Clear", command = clear_callback).pack(side=LEFT)
    
# Entry untuk Learning Rate
Label(toolbar, text='Learning Rate', background='white').pack(side=LEFT, padx = 10)
learning_rate_field = Entry(toolbar, textvariable=StringVar(root, value=LR), width=8)
learning_rate_field.pack(side=LEFT)

# Entry untuk Threshold
Label(toolbar, text='Threshold', background='white').pack(side=LEFT, padx = 10)
threshold_field = Entry(toolbar, textvariable=StringVar(root, value=threshold), width=8)
threshold_field.pack(side=LEFT)

# Entry untuk angka maksimum dari iterasi
Label(toolbar, text='Iterasi Maksimum', background='white').pack(side=LEFT, padx = 10)
max_iterations_field = Entry(toolbar, textvariable=StringVar(root, value=max_iterations), width=8)
max_iterations_field.pack(side=LEFT)

# Tombol Train
def train_callback():
    global weights, bias, threshold, LR, max_iterations
    threshold = float(threshold_field.get())
    LR = float(learning_rate_field.get())
    max_iterations = int(max_iterations_field.get())
    epoch = train_folder()
    messagebox.showinfo('Hasil Training', 'Training selesai dengan %d iterasi' % epoch)
Button(toolbar, text="Train", command = train_callback).pack(side=LEFT)

# Canvas Grid (kotak kosong)
def mouseClick(event):
    x = math.floor(event.x / rect_size)
    y = math.floor(event.y / rect_size)
    if x < w and y < h: states[x, y] = 0 if states[x, y] > 0 else 1 # swap zero & one
    print_grid()
rect_size = 50  # grid rectangles size
canvas = Canvas(frame, width=rect_size*w, height=rect_size*h, background='white')
canvas.bind("<Button-1>", mouseClick)
canvas.pack(side=TOP)

# Draw Grid (Kotak terarsir)
def print_grid():
    for i in range(w):
        for j in range(h):
            color = 'black' if states[i, j] > 0 else 'white'
            canvas.create_rectangle(i * rect_size, j * rect_size, (i + 1) * rect_size, (j + 1) * rect_size, outline="black", fill=color)
print_grid();

# Bottom Bar
bottom_bar = Frame(frame, height=50, background='white')
bottom_bar.pack(fill=X)

# Tombol Test
def test_callback():
    input = states.copy().reshape(input_size)
    (found, y_vec, bias_final, weight_final) = test(input)
    if len(found) > 0:
        test_result_field_value.set(', '.join(found))
        y_result_field_value.set(y_vec)
        print("Bias yang digunakan: %s" % bias_final)
        print("Beban yang digunakan: ")
        print(np.matrix(weight_final))
        print("\n")
    else:
        test_result_field_value.set('???')
Button(bottom_bar, text="Test", command = test_callback).pack(side=LEFT)

# Entry dari Prediksi
Label(bottom_bar, text='Prediksi', background='white').pack(side=LEFT, padx = 10)
test_result_field_value = StringVar()
test_result_field = Entry(bottom_bar, width=20, textvariable=test_result_field_value)
test_result_field.pack(side=LEFT, padx = 10)

# Entry dari nilai y
Label(bottom_bar, text='Kode Target (A, B, C, D, E, J, K)', background='white').pack(side=LEFT, padx = 10)
y_result_field_value = IntVar()
y_result_field = Entry(bottom_bar, width=20, textvariable=y_result_field_value)
y_result_field.pack(side=LEFT, padx = 10)

root.mainloop()