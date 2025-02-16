###########################################################
# final_Bayes.py
###########################################################

import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import (
    confusion_matrix,
    ConfusionMatrixDisplay,
    precision_score,
    recall_score,
    f1_score,
    accuracy_score,
    classification_report
)
from sklearn.model_selection import train_test_split, StratifiedKFold
import time

# PyTorch (BiLSTM)
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset

# Gensim for Word2Vec
import gensim.downloader as api

# Keras (IMDB dataset)
from keras.datasets import imdb
from keras.preprocessing.sequence import pad_sequences

###########################################################
#Helper functions & Custom BernoulliNB
###########################################################

def score_accuracy(y_true, y_pred):
    return accuracy_score(y_true, y_pred)

def score_precision(y_true, y_pred):
    return precision_score(y_true, y_pred, average="binary", zero_division=0)

def score_recall(y_true, y_pred):
    return recall_score(y_true, y_pred, average="binary", zero_division=0)

def score_f1(y_true, y_pred):
    return f1_score(y_true, y_pred, average="binary", zero_division=0)

def training(model, X, y, n_splits=5):
    """
    Εκτυπώνει μέσες τιμές Training/Validation F1 για διαφορετικά train sizes.
    """
    skf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=42)
    train_sizes = np.linspace(0.1, 1.0, 5)
    train_scores = []
    val_scores = []

    for train_idx, val_idx in skf.split(X, y):
        X_train_fold, X_val_fold = X[train_idx], X[val_idx]
        y_train_fold, y_val_fold = np.array(y)[train_idx], np.array(y)[val_idx]
        fold_train_scores = []
        fold_val_scores = []
        for train_size in train_sizes:
            subset_size = int(len(X_train_fold) * train_size)
            X_subset = X_train_fold[:subset_size]
            y_subset = y_train_fold[:subset_size]
            model.fit(X_subset, y_subset)
            y_pred_train = model.predict(X_subset)
            y_pred_val   = model.predict(X_val_fold)
            fold_train_scores.append(f1_score(y_subset, y_pred_train, zero_division=0))
            fold_val_scores.append(f1_score(y_val_fold, y_pred_val, zero_division=0))
        train_scores.append(fold_train_scores)
        val_scores.append(fold_val_scores)

    train_scores_mean = np.mean(train_scores, axis=0)
    val_scores_mean   = np.mean(val_scores, axis=0)
    return train_scores_mean, val_scores_mean, train_sizes

def training_metric(model, X, y, scoring_func, n_splits=5, train_sizes=np.linspace(0.1, 1.0, 5)):
    """
    Γενική συνάρτηση για να μετράμε training/validation metrics
    (Accuracy, Precision, Recall, κ.λπ.) για διαφορετικά train sizes.
    """
    skf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=42)
    train_scores = []
    val_scores = []

    for train_idx, val_idx in skf.split(X, y):
        X_train_fold, X_val_fold = X[train_idx], X[val_idx]
        y_train_fold, y_val_fold = np.array(y)[train_idx], np.array(y)[val_idx]
        fold_train_scores = []
        fold_val_scores = []
        for train_size in train_sizes:
            subset_size = int(len(X_train_fold) * train_size)
            X_subset = X_train_fold[:subset_size]
            y_subset = y_train_fold[:subset_size]
            model.fit(X_subset, y_subset)
            y_pred_train = model.predict(X_subset)
            y_pred_val   = model.predict(X_val_fold)
            fold_train_scores.append(scoring_func(y_subset, y_pred_train))
            fold_val_scores.append(scoring_func(y_val_fold, y_pred_val))
        train_scores.append(fold_train_scores)
        val_scores.append(fold_val_scores)

    train_scores_mean = np.mean(train_scores, axis=0)
    val_scores_mean   = np.mean(val_scores, axis=0)
    return train_scores_mean, val_scores_mean, train_sizes

class BernoulliNaiveBayes:
    """
    Custom Bernoulli Naive Bayes (δεν βασίζεται σε scikit-learn)
    """
    def __init__(self, alpha=1.0):
        self.alpha = alpha

    def fit(self, X, y):
        X = np.array(X, dtype=np.float32)
        y = np.array(y, dtype=np.int32)
        self.classes_ = np.unique(y)
        n_classes = len(self.classes_)
        n_features = X.shape[1]

        class_count = np.zeros(n_classes, dtype=np.float64)
        feature_count = np.zeros((n_classes, n_features), dtype=np.float64)

        for i, c in enumerate(self.classes_):
            X_c = X[y == c]
            class_count[i] = X_c.shape[0]
            feature_count[i] = np.sum(X_c, axis=0)

        self.class_log_prior_ = np.log(class_count / class_count.sum())

        # Laplace smoothing
        smoothed_fc = feature_count + self.alpha
        smoothed_denom = class_count.reshape(-1, 1) + 2 * self.alpha
        self.feature_prob_ = smoothed_fc / smoothed_denom
        self.feature_log_prob_ = np.log(self.feature_prob_)
        self.feature_log_prob_neg_ = np.log(1.0 - self.feature_prob_)
        return self

    def predict(self, X):
        X = np.array(X, dtype=np.float32)
        jll = []
        for i in range(len(self.classes_)):
            log_prob = self.class_log_prior_[i] + np.sum(
                X * self.feature_log_prob_[i] + (1 - X) * self.feature_log_prob_neg_[i],
                axis=1
            )
            jll.append(log_prob)
        jll = np.array(jll).T
        return self.classes_[np.argmax(jll, axis=1)]

###########################################################
# 2. Φόρτωση IMDB (0/1) για Naive Bayes + Train/Val/Test
###########################################################

def load_and_preprocess_imdb_nb(num_words=10000, skip_top=20):
    """
    Φορτώνει το IMDB dataset ως 0/1 bag-of-words (μέγεθος = num_words).
    """
    (x_train, y_train), (x_test, y_test) = imdb.load_data(num_words=num_words, skip_top=skip_top)

    def to_binary_vector(sequences, vocab_size):
        bin_matrix = np.zeros((len(sequences), vocab_size), dtype=np.float32)
        for i, seq in enumerate(sequences):
            for idx in seq:
                if 0 <= idx < vocab_size:
                    bin_matrix[i, idx] = 1.0
        return bin_matrix

    x_train = to_binary_vector(x_train, num_words)
    x_test  = to_binary_vector(x_test,  num_words)
    return x_train, y_train, x_test, y_test

###########################################################
# ΜΕΡΟΣ Α κ Β: Bernoulli NB Πλήρης Κώδικας
###########################################################

if __name__ == "__main__":
    # -- Φόρτωση δεδομένων (Naive Bayes) --
    x_data_nb, y_data_nb, x_test_nb, y_test_nb = load_and_preprocess_imdb_nb(num_words=10000, skip_top=20)
    print("Complete data size:", x_data_nb.shape, np.array(y_data_nb).shape)

    # -- Split σε train/val --
    x_train_nb, x_val_nb, y_train_nb, y_val_nb = train_test_split(
        x_data_nb, y_data_nb, test_size=0.1, random_state=42
    )
    print("Training data size:", x_train_nb.shape, y_train_nb.shape)
    print("Validation data size:", x_val_nb.shape, y_val_nb.shape)

    # -- Δημιουργία μοντέλου NB --
    nb_model = BernoulliNaiveBayes(alpha=1.0)

    # -- (A) Καμπύλες μάθησης για F1 --
    train_f1, val_f1, tr_sizes = training(nb_model, x_train_nb, y_train_nb)
    plt.figure(figsize=(6,4))
    plt.plot(tr_sizes, train_f1, marker='o', label="Training F1 Score")
    plt.plot(tr_sizes, val_f1, marker='s', label="Validation F1 Score")
    plt.title("Learning Curve: Custom Bernoulli Naive Bayes (F1 Score)")
    plt.xlabel("Training Size (Proportion)")
    plt.ylabel("F1 Score")
    plt.legend()
    plt.grid()
    plt.show()

    # -- (B) Καμπύλες μάθησης για Accuracy --
    train_acc, val_acc, _ = training_metric(nb_model, x_train_nb, y_train_nb, score_accuracy)
    plt.figure(figsize=(6,4))
    plt.plot(tr_sizes, train_acc, marker='o', label="Training Accuracy")
    plt.plot(tr_sizes, val_acc, marker='s', label="Validation Accuracy")
    plt.title("Learning Curve: Custom Bernoulli Naive Bayes (Accuracy)")
    plt.xlabel("Training Size (Proportion)")
    plt.ylabel("Accuracy")
    plt.legend()
    plt.grid()
    plt.show()

    # -- (C) Καμπύλες μάθησης για Precision --
    train_prec, val_prec, _ = training_metric(nb_model, x_train_nb, y_train_nb, score_precision)
    plt.figure(figsize=(6,4))
    plt.plot(tr_sizes, train_prec, marker='o', label="Training Precision")
    plt.plot(tr_sizes, val_prec, marker='s', label="Validation Precision")
    plt.title("Learning Curve: Custom Bernoulli Naive Bayes (Precision)")
    plt.xlabel("Training Size (Proportion)")
    plt.ylabel("Precision")
    plt.legend()
    plt.grid()
    plt.show()

    # -- (D) Καμπύλες μάθησης για Recall --
    train_rec, val_rec, _ = training_metric(nb_model, x_train_nb, y_train_nb, score_recall)
    plt.figure(figsize=(6,4))
    plt.plot(tr_sizes, train_rec, marker='o', label="Training Recall")
    plt.plot(tr_sizes, val_rec, marker='s', label="Validation Recall")
    plt.title("Learning Curve: Custom Bernoulli Naive Bayes (Recall)")
    plt.xlabel("Training Size (Proportion)")
    plt.ylabel("Recall")
    plt.legend()
    plt.grid()
    plt.show()

    # -- Εκπαίδευση σε train_nb και αξιολόγηση σε val_nb --
    nb_model.fit(x_train_nb, y_train_nb)
    val_preds_custom = nb_model.predict(x_val_nb)

    acc_val = accuracy_score(y_val_nb, val_preds_custom)
    prec_val = precision_score(y_val_nb, val_preds_custom, zero_division=0)
    rec_val = recall_score(y_val_nb, val_preds_custom, zero_division=0)
    f1_val = f1_score(y_val_nb, val_preds_custom, zero_division=0)

    print("\nCustom Bernoulli Naive Bayes Evaluation on Validation Set:")
    print(f"Accuracy: {acc_val:.4f}")
    print(f"Precision: {prec_val:.4f}")
    print(f"Recall: {rec_val:.4f}")
    print(f"F1 Score: {f1_val:.4f}")

    print("\nClassification Report (Validation):\n")
    print(classification_report(y_val_nb, val_preds_custom, target_names=["Negative","Positive"], zero_division=0))

    # -- Confusion Matrix (Validation) για το Custom NB --
    cm_custom_val = confusion_matrix(y_val_nb, val_preds_custom)
    disp_custom_val = ConfusionMatrixDisplay(cm_custom_val, display_labels=["Negative","Positive"])
    disp_custom_val.plot(cmap=plt.cm.Blues)
    plt.title("Confusion Matrix - Custom Bernoulli Naive Bayes (Validation)")
    plt.show()

    # -- Αναζήτηση βέλτιστου alpha --
    alpha_values = [0.1, 0.5, 1.0, 2.0, 5.0]
    results_alpha = []
    for alpha_ in alpha_values:
        nb_temp = BernoulliNaiveBayes(alpha=alpha_)
        nb_temp.fit(x_train_nb, y_train_nb)
        preds_temp = nb_temp.predict(x_val_nb)
        acc_  = accuracy_score(y_val_nb, preds_temp)
        prec_ = precision_score(y_val_nb, preds_temp, zero_division=0)
        rec_  = recall_score(y_val_nb, preds_temp, zero_division=0)
        f1_   = f1_score(y_val_nb, preds_temp, zero_division=0)
        results_alpha.append({
            'alpha': alpha_,
            'accuracy': acc_,
            'precision': prec_,
            'recall': rec_,
            'f1': f1_
        })

    # Εκτύπωση
    print("\n--- Alpha search results ---")
    for r in results_alpha:
        print(f"Alpha: {r['alpha']}, Accuracy: {r['accuracy']:.4f}, "
              f"Precision: {r['precision']:.4f}, Recall: {r['recall']:.4f}, F1: {r['f1']:.4f}")

    # Επιλογή βέλτιστου (με κριτήριο F1)
    best_res = sorted(results_alpha, key=lambda x: x['f1'], reverse=True)[0]
    print("\nBest Hyperparameter Result:")
    print(best_res)

    # -- (Προαιρετικό) Διαγραμμα απόδοσης vs alpha --
    # Παράδειγμα: ένα plot που δείχνει πώς μεταβάλλονται Accuracy, Precision, Recall, F1 vs alpha
    plt.figure(figsize=(6,4))
    x_indices = np.arange(len(alpha_values))  # 0..4
    acc_list  = [d['accuracy'] for d in results_alpha]
    prec_list = [d['precision'] for d in results_alpha]
    rec_list  = [d['recall']   for d in results_alpha]
    f1_list   = [d['f1']       for d in results_alpha]

    plt.plot(x_indices, acc_list,  marker='o', label="Accuracy")
    plt.plot(x_indices, prec_list, marker='^', label="Precision")
    plt.plot(x_indices, rec_list,  marker='s', label="Recall")
    plt.plot(x_indices, f1_list,   marker='D', label="F1")

    plt.xticks(x_indices, [str(a) for a in alpha_values])
    plt.title("Model Performance Metrics vs. α (Laplace Smoothing)")
    plt.xlabel("α")
    plt.ylabel("Score")
    plt.legend()
    plt.grid()
    plt.show()

    # -- Σύγκριση με Scikit-Learn BernoulliNB --
    from sklearn.naive_bayes import BernoulliNB
    sk_nb = BernoulliNB(alpha=best_res['alpha'])
    sk_nb.fit(x_train_nb, y_train_nb)
    sk_preds_val = sk_nb.predict(x_val_nb)

    sk_acc_val  = accuracy_score(y_val_nb, sk_preds_val)
    sk_prec_val = precision_score(y_val_nb, sk_preds_val, zero_division=0)
    sk_rec_val  = recall_score(y_val_nb, sk_preds_val, zero_division=0)
    sk_f1_val   = f1_score(y_val_nb, sk_preds_val, zero_division=0)

    print(f"\nScikit-learn BernoulliNB Evaluation on Validation Set (alpha={best_res['alpha']}):")
    print(f"Accuracy : {sk_acc_val:.4f}")
    print(f"Precision: {sk_prec_val:.4f}")
    print(f"Recall   : {sk_rec_val:.4f}")
    print(f"F1 Score : {sk_f1_val:.4f}")

    print("\nClassification Report (Validation):\n")
    print(classification_report(y_val_nb, sk_preds_val, target_names=["Negative","Positive"], zero_division=0))

    # -- Confusion Matrix (Validation) Scikit NB --
    cm_sk_val = confusion_matrix(y_val_nb, sk_preds_val)
    disp_sk_val = ConfusionMatrixDisplay(cm_sk_val, display_labels=["Negative","Positive"])
    disp_sk_val.plot(cmap=plt.cm.Blues)
    plt.title("Confusion Matrix - Scikit-learn BernoulliNB (Validation)")
    plt.show()

    # ---------------------------------------------------
    # 4. Τελική εκπαίδευση σε TRAIN+VAL, Αξιολόγηση στο TEST
    # ---------------------------------------------------
    print("\n--- Final model on Test Set (using best alpha) ---")

    # Ενώνουμε Train + Val
    X_train_full = np.concatenate([x_train_nb, x_val_nb], axis=0)
    y_train_full = np.concatenate([y_train_nb, y_val_nb], axis=0)

    # Φτιάχνουμε νέο μοντέλο με beta alpha
    nb_final = BernoulliNaiveBayes(alpha=best_res['alpha'])
    nb_final.fit(X_train_full, y_train_full)
    test_preds_final = nb_final.predict(x_test_nb)

    acc_test  = accuracy_score(y_test_nb, test_preds_final)
    prec_test = precision_score(y_test_nb, test_preds_final, zero_division=0)
    rec_test  = recall_score(y_test_nb, test_preds_final, zero_division=0)
    f1_test   = f1_score(y_test_nb, test_preds_final, zero_division=0)

    print("Test metrics (Custom NB):")
    print(f"Accuracy : {acc_test:.4f}")
    print(f"Precision: {prec_test:.4f}")
    print(f"Recall   : {rec_test:.4f}")
    print(f"F1 Score : {f1_test:.4f}")

    # Classification report στο test set
    cr_test = classification_report(y_test_nb, test_preds_final, target_names=["Negative","Positive"], zero_division=0)
    print("\nClassification Report (Test) - Custom NB:\n")
    print(cr_test)

    # Confusion matrix στο test set
    cm_final_test = confusion_matrix(y_test_nb, test_preds_final)
    disp_final_test = ConfusionMatrixDisplay(cm_final_test, display_labels=["Negative","Positive"])
    disp_final_test.plot(cmap=plt.cm.Blues)
    plt.title("Confusion Matrix (Test) - Custom Bernoulli Naive Bayes")
    plt.show()

    # ---------------------------------------------------
    # ΜΕΡΟΣ Γ: BiLSTM RNN
    # ---------------------------------------------------
    # Φορτώνουμε ξανά το IMDB σε μορφή ακολουθιών (για το RNN)
    def load_imdb_for_rnn(num_words=10000, skip_top=20, max_len=200):
        (x_train_seq, y_train_seq), (x_test_seq, y_test_seq) = imdb.load_data(num_words=num_words, skip_top=skip_top)
        x_train_seq, x_val_seq, y_train_seq, y_val_seq = train_test_split(
            x_train_seq, y_train_seq, test_size=0.1, random_state=42
        )
        x_train_seq = pad_sequences(x_train_seq, maxlen=max_len, padding='post', truncating='post')
        x_val_seq   = pad_sequences(x_val_seq,   maxlen=max_len, padding='post', truncating='post')
        x_test_seq  = pad_sequences(x_test_seq,  maxlen=max_len, padding='post', truncating='post')
        return (x_train_seq, y_train_seq), (x_val_seq, y_val_seq), (x_test_seq, y_test_seq)

    (x_train_rnn, y_train_rnn), (x_val_rnn, y_val_rnn), (x_test_rnn, y_test_rnn) = load_imdb_for_rnn()
    print(f"\n[BiLSTM] Training set shape: {x_train_rnn.shape} {len(y_train_rnn)}")
    print(f"[BiLSTM] Validation set shape: {x_val_rnn.shape} {len(y_val_rnn)}")
    print(f"[BiLSTM] Test set shape: {x_test_rnn.shape} {len(y_test_rnn)}")

    print("\nLoading pre-trained Word2Vec embeddings (GoogleNews-vectors-negative300)... This may take a while.")
    word2vec = api.load("word2vec-google-news-300")
    embedding_dim = 300

    # Χτίζουμε τα index_word / word_index
    word_index = imdb.get_word_index()
    word_index = {w: (i+3) for w, i in word_index.items()}
    word_index["<PAD>"] = 0
    word_index["<START>"] = 1
    word_index["<UNK>"] = 2
    word_index["<UNUSED>"] = 3
    index_word = {idx: w for w, idx in word_index.items() if idx < 10000}

    # Embedding matrix
    embedding_matrix = np.zeros((10000, embedding_dim), dtype=np.float32)
    for i in range(10000):
        if i < 4:
            embedding_matrix[i] = np.zeros(embedding_dim)
        else:
            word = index_word.get(i, None)
            if word is not None and word in word2vec:
                embedding_matrix[i] = word2vec[word]
            else:
                embedding_matrix[i] = np.random.normal(scale=0.6, size=(embedding_dim,))

    # Torch Dataset
    batch_size = 32
    x_train_tensor = torch.tensor(x_train_rnn, dtype=torch.long)
    y_train_tensor = torch.tensor(y_train_rnn, dtype=torch.float32)
    x_val_tensor   = torch.tensor(x_val_rnn,   dtype=torch.long)
    y_val_tensor   = torch.tensor(y_val_rnn,   dtype=torch.float32)
    x_test_tensor  = torch.tensor(x_test_rnn,  dtype=torch.long)
    y_test_tensor  = torch.tensor(y_test_rnn,  dtype=torch.float32)

    train_dataset = TensorDataset(x_train_tensor, y_train_tensor)
    val_dataset   = TensorDataset(x_val_tensor,   y_val_tensor)
    test_dataset  = TensorDataset(x_test_tensor,  y_test_tensor)

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_loader   = DataLoader(val_dataset,   batch_size=batch_size)
    test_loader  = DataLoader(test_dataset,  batch_size=batch_size)

    class StackedBiLSTM(nn.Module):
        def __init__(self, num_embeddings, embedding_dim, hidden_dim, num_layers, output_dim, embedding_matrix):
            super(StackedBiLSTM, self).__init__()
            self.embedding = nn.Embedding(num_embeddings, embedding_dim)
            # Φορτώνουμε τα προ-εκπαιδευμένα embeddings
            self.embedding.weight = nn.Parameter(torch.tensor(embedding_matrix, dtype=torch.float32))
            self.embedding.weight.requires_grad = False  # freeze embeddings

            self.lstm = nn.LSTM(
                input_size=embedding_dim,
                hidden_size=hidden_dim,
                num_layers=num_layers,
                bidirectional=True,
                batch_first=True
            )
            self.fc = nn.Linear(hidden_dim * 2, output_dim)

        def forward(self, x):
            embedded = self.embedding(x)  # [batch, seq_len, embedding_dim]
            lstm_out, _ = self.lstm(embedded)
            # Max-pooling over time
            pooled, _ = torch.max(lstm_out, dim=1)
            logits = self.fc(pooled)
            return torch.sigmoid(logits)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("\nUsing device:", device)

    hidden_dim = 64
    num_layers = 2
    output_dim = 1
    model_rnn = StackedBiLSTM(
        num_embeddings=10000,
        embedding_dim=embedding_dim,
        hidden_dim=hidden_dim,
        num_layers=num_layers,
        output_dim=output_dim,
        embedding_matrix=embedding_matrix
    ).to(device)

    criterion = nn.BCELoss()
    optimizer = optim.Adam(model_rnn.parameters(), lr=0.001)
    epochs = 10

    best_val_loss = float('inf')
    best_model_state = None
    train_losses = []
    val_losses = []

    print("\n--- Training BiLSTM ---")
    for epoch in range(epochs):
        model_rnn.train()
        running_train_loss = 0.0
        for x_batch, y_batch in train_loader:
            x_batch, y_batch = x_batch.to(device), y_batch.to(device)
            optimizer.zero_grad()
            outputs = model_rnn(x_batch).squeeze()
            loss = criterion(outputs, y_batch)
            loss.backward()
            optimizer.step()
            running_train_loss += loss.item()

        epoch_train_loss = running_train_loss / len(train_loader)
        train_losses.append(epoch_train_loss)

        # Validation
        model_rnn.eval()
        running_val_loss = 0.0
        with torch.no_grad():
            for x_batch, y_batch in val_loader:
                x_batch, y_batch = x_batch.to(device), y_batch.to(device)
                outputs = model_rnn(x_batch).squeeze()
                loss = criterion(outputs, y_batch)
                running_val_loss += loss.item()

        epoch_val_loss = running_val_loss / len(val_loader)
        val_losses.append(epoch_val_loss)

        print(f"Epoch {epoch+1}/{epochs} - Train Loss: {epoch_train_loss:.4f}, Val Loss: {epoch_val_loss:.4f}")

        # Save best model
        if epoch_val_loss < best_val_loss:
            best_val_loss = epoch_val_loss
            best_model_state = model_rnn.state_dict()

    # Φορτώνουμε το καλύτερο state
    model_rnn.load_state_dict(best_model_state)
    print("Training complete. Best validation loss:", best_val_loss)

    # Plot των loss curves
    plt.figure(figsize=(6,4))
    plt.plot(range(1, epochs+1), train_losses, marker='o', label="Training Loss")
    plt.plot(range(1, epochs+1), val_losses,   marker='s', label="Validation Loss")
    plt.title("Stacked BiLSTM Loss Curves")
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.legend()
    plt.grid()
    plt.show()

    # -- Αξιολόγηση στο test set --
    model_rnn.eval()
    all_preds = []
    all_labels = []
    with torch.no_grad():
        for x_batch, y_batch in test_loader:
            x_batch, y_batch = x_batch.to(device), y_batch.to(device)
            outputs = model_rnn(x_batch).squeeze()
            preds = (outputs > 0.5).float()
            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(y_batch.cpu().numpy())

    acc_rnn = accuracy_score(all_labels, all_preds)
    prec_rnn = precision_score(all_labels, all_preds, zero_division=0)
    rec_rnn  = recall_score(all_labels, all_preds, zero_division=0)
    f1_rnn   = f1_score(all_labels, all_preds, zero_division=0)

    print("\n[BiLSTM] Test Set Metrics:")
    print(f"Accuracy : {acc_rnn:.4f}")
    print(f"Precision: {prec_rnn:.4f}")
    print(f"Recall   : {rec_rnn:.4f}")
    print(f"F1 Score : {f1_rnn:.4f}")

    report_rnn = classification_report(all_labels, all_preds, target_names=["Negative","Positive"], zero_division=0)
    print("\n[BiLSTM] Classification Report (Test):\n", report_rnn)

    # Προαιρετικά μπορούμε να δούμε confusion matrix και για το BiLSTM:
    cm_bilstm = confusion_matrix(all_labels, all_preds)
    disp_bilstm = ConfusionMatrixDisplay(cm_bilstm, display_labels=["Negative","Positive"])
    disp_bilstm.plot(cmap=plt.cm.Blues)
    plt.title("Confusion Matrix - BiLSTM (Test)")
    plt.show()

    # -- Τελική σύγκριση --
    print("\n--- Final Comparison ---")
    print(f"Custom NB (best alpha={best_res['alpha']}) -> Test F1 = {f1_test:.4f}")
    print(f"BiLSTM (test)                              -> F1      = {f1_rnn:.4f}")
    print("\nDone! All parts integrated.\n")

    # (Προαιρετική παύση)
    time.sleep(2)
