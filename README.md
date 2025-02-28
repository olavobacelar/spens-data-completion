# Structured Prediction Energy Networks (SPENs) for Data Completion

Code and experiments from my Master’s thesis on [Structured Prediction Energy Networks for Data Completion](https://fenix.tecnico.ulisboa.pt/downloadFile/281870113705561/Deep%20Energy%20Networks%20for%20Data%20Completion%20[submission]%20%20Olavo%20Bacelar.pdf), published in 2021. It features a new framework that adapts [SPENs](https://arxiv.org/pdf/1511.06350) to handle data completion with arbitrary masks.

The proposed framework for data completion performs better than an autoencoder baseline in a font completion task. Also, unlike with previous SPEN experiments, it does not require a separate initialization network to achieve good results and requires only a single step of gradient-based inference.

This code provides an implementation of that framework for completing missing letters in fonts. A summary paper of the thesis can be found [here](https://fenix.tecnico.ulisboa.pt/downloadFile/281870113705562/Deep%20Energy%20Networks%20for%20Data%20Completion%20[submission,%20paper]%20%20Olavo%20Bacelar.pdf).
