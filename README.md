# Ham-And-Spam-Classifier

Build a (binary) spam classifier using a naive Bayes model that classifies each document as “Spam” (1) or “Ham” (0).

# Dataset

We will use the dataset drawn from emails released during the 2001 Enron investigation.
In what follows, we will describe this dataset in detail. However, note that we have provided a
python function that handles reading these files for you.
In the directory enron, you will find
- Word ID to word mapping in ind to vocab.txt. Specifically, each line consists of word
ID and its corresponding actual word (space-separated). There are 158,963 words in the
vocabulary.
- 20,229 training emails in train features.txt and their labels in train labels.txt.
- 6,743 validation emails in val features.txt and their labels in val labels.txt.
- 6,744 sampled test emails in sampled test features.txt and their labels in
sampled test labels.txt.

The input format of *_features.txt is shown in Fig. 2. Each of these features files consists
of multiple documents separated by #. Each document starts with its ID n followed by Dn
lines, where each line consists of word id and its frequency (space-separated). For example, the
word word id 3,5 appears in document 3 for frequency word id 3,5 times. The labels of these
documents are in *_labels.txt, in which each line consists of document ID and its label (0 or 1).

**Understanding Data Format**
In nb spam.py, we have provided the function load docs
which reads documents (words and their corresponding frequencies) for you. This function takes in
two file names (features and labels), and outputs two lists. The first list contains documents, each
of which is a list of tuples of word and its frequency in the document. The second contains labels
of those documents.

# Train And Predict
• nb_train estimates the parameters of the naive Bayes model, including the probability distribution
of class being Ham or Spam a priori as well as the class-specific word probability
distributions.

• nb_predict uses the estimates from nb train to predict which class each new document
belongs to.

Please follow the following implementation details. First, use the log-likelihood instead of the
likelihood to avoid any numerical underflow issues. Moreover, if your model decides that a document
has an equal chance of being Ham (0) or Spam (1), it should predict a Spam (1).
Finally, we will have to take care of unseen words. Unseen words are defined as words that do
not appear in the training documents. In this question, we will ignore all unseen words during
prediction. In addition, if you encounter computing log of a zero, replace the output with -1e50.
This scenario could happen when a word appears in the training documents that belong to one
class but it does not appear in those that belong to the other class.

Run script nb1.sh, which will generate a nb1.json file containing training, validation, and test accuracies. You should see that all accuracies are above 97%.

# Train With Smoothing And Predict
We now take another approach to dealing with unseen words: a smoothing technique. Let α > 0 be a continuous-valued smoothing parameter. For each word in the vocabulary in each class, you will assume that you have seen it α times before even seeing
any training data. That is, you should add α to its count before you train your model. This
assumption will affect class-specific word distribution estimates. (Hint: you should see that α (and
adjusted counts) affects both the numerator and the denominator in the formula of your parameter
estimates).
Please finish the implementation of the function nb_train smoothing in nb_spam.py with the
assumption described above. nb train smoothing estimates the parameters of the naive Bayes
model, similar to nb train. Implementation details in the previous question apply to this question.
Try α ∈ {1e-8, 1e-7, 1e-6, 1e-5, 1e-4}.

Run script nb2.sh, which will generate a nb2.json file containing training,
validation, and test accuracies for all values of α. You should see that, for the best α, all accuracies
are above 98%.
