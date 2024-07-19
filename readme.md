# Machine Learning for Intrusion Detection in Cybersecurity

## Project Overview

This project focuses on developing machine learning models for intrusion detection in cybersecurity. The models are trained and tested on the KDD99 Cup Dataset, leveraging various neural network architectures.

## Directory Structure
```
Project/
│
├── notebooks/
│ └── Supervised_Training_KDD99.ipynb
│
├── src/
│ ├── data/
│ │ ├── preprocess.py
│ │ └── load_data.py
│ │
│ ├── models/
│ │ ├── cnn_model.py
│ │ ├── mlp_model.py
│ │ └── rnn_model.py
│ │
│ ├── utils/
│ │ ├── optimizers.py
│ │ └── regularizers.py
│ │
│ └── main.py
│
├── tests/
│ ├── test_optimizers.py
│ ├── test_regularizers.py
│ ├── test_cnn_model.py
│ ├── test_mlp_model.py
│ └── test_rnn_model.py
│
├── README.md
├── requirements.txt
└── .gitignore   


## How to Run

1. Clone the repository.
2. Install the required dependencies using `pip install -r requirements.txt`.
3. Run the main script `python src/main.py`.

## References

1. KDD Cup 1999 Dataset: http://kdd.ics.uci.edu/databases/kddcup99/kddcup99.html
2. TensorFlow: https://www.tensorflow.org/

