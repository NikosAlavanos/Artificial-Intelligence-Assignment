Αλαβάνος Νίκος p3130003
Γιοβανόπουλος Κωνσταντίνος p3190275
Νικηφοράκης Βασίλειος p3160114




________________


Γενική Περιγραφή:
Το παρόν project υλοποιεί 3 διαφορετικές μεθόδους για την κατηγοριοποίηση κειμένων (π.χ., κριτικές ταινιών από το IMDB dataset) σε δύο κατηγορίες (Αρνητική και Θετική):
- Random Forest: (υλοποιήθηκε μέσω Jupyter Notebook)
- AdaBoost με Decision Stumps: (υλοποιήθηκε μέσω Jupyter Notebook)
- Αφελή Ταξινομητή Bayes (Bernoulli) (υλοποιήθηκε μέσω pycharm)


- Οδηγίες Εκτέλεσης για το RandomForest και AdaBoost -

Περιγραφή:
1. Προσαρμοσμένο Random Forest:
   * Τα κείμενα αναπαρίστανται ως δυαδικοί πίνακες bag-of-words.
   * Η μέθοδος υλοποιείται με χειροκίνητο bootstrapping και εκπαίδευση δέντρων απόφασης (χρησιμοποιώντας το κριτήριο "entropy", όπως στο ID3).
   * Οι προβλέψεις συγκεντρώνονται μέσω πλειοψηφίας ψήφων.
2. Προσαρμοσμένη υλοποίηση του AdaBoost με Decision Stumps:
   * Τα κείμενα αναπαρίστανται ως δυαδικοί πίνακες bag-of-words.
   * Ο αλγόριθμος AdaBoost υλοποιείται χειροκίνητα με χρήση decision stumps (δέντρα απόφασης με μέγιστο βάθος 1).
   * Οι ετικέτες μετατρέπονται από {0,1} σε {-1,+1} και εφαρμόζεται επαναβάθμιση (re-weighting) των δειγμάτων για την ενίσχυση της απόδοσης.
Και οι δύο μέθοδοι αξιολογούνται βάσει μετρικών (ακρίβεια, precision, recall, F1 score) και παρουσιάζονται γραφήματα που απεικονίζουν τις καμπύλες μάθησης/απώλειας, πίνακες σύγχυσης και αναλυτικές αναφορές.
________________


Απαιτήσεις:
* Python 3.9 ή νεότερη
* Βιβλιοθήκες: numpy, matplotlib, seaborn, scikit‑learn, keras, torch (PyTorch), gensim, tensorflow (για το IMDB dataset)
* (Προαιρετικά) Jupyter Notebook για διαδραστική εκτέλεση
Για εγκατάσταση των απαραίτητων βιβλιοθηκών, εκτελέστε:


pip install numpy matplotlib seaborn scikit-learn keras torch gensim tensorflow


Οδηγίες Εκτέλεσης:
1. Για την υλοποίηση του Random Forest και του AdaBoost, ανοίξτε το notebook (π.χ., RandomForest.ipynb) και εκτελέστε τα cells σειριακά.
2. Βεβαιωθείτε ότι το περιβάλλον έχει ρυθμιστεί σωστά (π.χ., ενεργοποίηση GPU, εάν είναι διαθέσιμη).


* Οδηγίες Εκτέλεσης για το Bayes.py -
 Περιγραφή
Αυτό το πρόγραμμα υλοποιεί έναν Αφελή Ταξινομητή Bayes (Bernoulli) για την κατάταξη κειμένων σε δύο κατηγορίες (π.χ. θετική/αρνητική γνώμη), χρησιμοποιώντας το IMDB dataset. Περιλαμβάνει επίσης και ένα μοντέλο BiLSTM (με χρήση PyTorch) για σύγκριση.
 Απαιτήσεις / Βιβλιοθήκες
Για να τρέξει το πρόγραμμα, απαιτείται Python 3.8+ και οι παρακάτω βιβλιοθήκες:
·        NumPy: pip install numpy
·        Matplotlib: pip install matplotlib
·        scikit-learn: pip install scikit-learn
·        tqdm: pip install tqdm
·        TensorFlow: pip install tensorflow
·        PyTorch: pip install torch torchvision torchaudio
·        Gensim: pip install gensim
 Τρόπος Εκτέλεσης
1. Κατέβασε το αρχείο Bayes.py.
2. Άνοιξε ένα τερματικό στον φάκελο όπου βρίσκεται το αρχείο.
3. Τρέξε την εντολή:
·           python Bayes.py