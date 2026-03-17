# Feed-Forward Neural Network

Implementasi **Feed-Forward Neural Network (FFNN)** menggunakan Python dan NumPy. Proyek ini dibangun menggunakan mesin automatic differentiation (autodiff), sehingga proses backpropagation dilakukan secara otomatis melalui komputasi graf.

---

## Struktur Proyek

```
Feed-Forward-Neural-Network_ml-1/
├── data/
│   └── datasetml_2026.csv
├── src/
│   ├── ffnn.py
│   ├── test.py
│   ├── FFNN.ipynb
│   └── module/
│       ├── autodiff.py
│       ├── layer.py
│       ├── activation.py
│       ├── loss_function.py
│       ├── optimizer.py
│       └── normalization.py
├── pyproject.toml
└── README.md
```

---

## Fitur

- **Automatic Differentiation** — backpropagation otomatis dengan komputasi graf
- **Sequential API Layer** — bangun model layer demi layer
- **Weight Initialization**: `zero`, `random_uniform`, `random_normal`, `he`, `xavier`
- **Activation Functions**: `Linear`, `ReLU`, `Sigmoid`, `Tanh`, `Softmax`, `Swish`, `LeakyReLU`
- **Loss Functions**: `MSE`, `BinaryCrossEntropy`, `CategoricalCrossEntropy`
- **Optimizers**: `SGD`, `Adam` (dengan regularisasi L1 & L2)
- **Normalization**: `RMSNorm`
- **Visualisasi**: distribusi bobot dan gradien per layer
- **Save & Load** model ke file `.npy`

---

## Setup & Instalasi

Proyek ini menggunakan **Python ≥ 3.13** dan dikelola dengan [`uv`](https://github.com/astral-sh/uv).

### 1. Clone repository

```bash
git clone https://github.com/ahsuunn/Feed-Forward-Neural-Network_ml-1.git
cd Feed-Forward-Neural-Network_ml-1
```

### 2. Install dependencies

Menggunakan `uv`:

```bash
uv sync
```

### 3. Aktifkan virtual environment

```bash
# Windows
.venv\Scripts\activate

# Linux / macOS
source .venv/bin/activate
```

---

## Cara Menjalankan

### Menjalankan script pengujian

```bash
cd src
uv run test
```

### Menjalankan notebook eksperimen

```bash
cd src
uv run jupyter notebook FFNN.ipynb
```

---

## Contoh Penggunaan

```python
import numpy as np
from ffnn import FeedForwardNeuralNetwork
from module.layer import Layer
from module.activation import ReLU, Softmax
from module.loss_function import CategoricalCrossEntropy
from module.optimizer import Adam

# Buat model
model = FeedForwardNeuralNetwork(verbose=True, seed=42)
model.add(Layer(input_size=4, output_size=16, initialization_type="he"))
model.add(ReLU())
model.add(Layer(input_size=16, output_size=3, initialization_type="he"))
model.add(Softmax())

# Kumpulkan parameter untuk optimizer
params = []
for layer in model.layers:
    if hasattr(layer, "get_parameters"):
        params.extend(layer.get_parameters())

# Compile & train
model.compile(CategoricalCrossEntropy(), optimizer=Adam(params, learning_rate=0.01))
history = model.fit(X_train, y_train, epochs=50, batch_size=32,
                    validation_data=(X_val, y_val))

# Prediksi
y_pred = model.predict(X_test)

# Simpan & muat model
model.save("model.npy")
model.load("model.npy")

# Visualisasi
model.plot_weight_distributions()
model.plot_gradient_distributions()
```

---

## Pembagian Tugas Anggota Kelompok

| NIM        | Nama           | Tugas                                                                                                                                       |
| ---------- | -------------- | ------------------------------------------------------------------------------------------------------------------------------------------- |
| `13523113` | Kefas Kurnia Jonathan | `autodiff.py` — Automatic Differentiation Engine (kelas `Tensor`, operasi forward & backward)                                               |
| `13523074` | Ahsan Malik Al Farisi | `layer.py`, `normalization.py`, `activation.py` — Dense Layer, RMSNorm, inisialisasi bobot, fungsi aktivasi                                 |
| `13622076` | Ziyan Agil Nur Ramadhan | `loss_function.py`, `optimizer.py`, `ffnn.py` — Fungsi loss, Optimizer (SGD, Adam), kelas utama FFNN, training loop, visualisasi, save/load |

---
