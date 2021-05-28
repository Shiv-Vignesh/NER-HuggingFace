import transformers
from transformers import BertTokenizer
from keras.preprocessing.sequence import pad_sequences
from utilis.Enums import Parameters

class BertTokenizerClass(object):

    '''
    Create an instance of BertTokenizer & perform tokenization on train & test sentences. 
    1) Tokenize sentences in documents 
    2) Pad the tokenized sequences & trim the length to MAX_LEN

    Parameters 
    ----------------
    sentences : List of sentences present in a document. 
    labels : List of labels for tokens in a document. 
    tag2idx : Convert the entity/tag into numerical representation
    
    '''

    def __init__(self, sentences, labels, tag2idx):
        self.tokenizer = BertTokenizer.from_pretrained('bert-base-cased', do_lower_case=False)
        self.sentences = sentences
        self.labels = labels
        self.tag2idx = tag2idx

        self.tokenized_texts, self.labels = self.tokenize_sentences_labels(self.sentences, self.labels)
        self.input_ids, self.tags = self.pad_tokenized_texts(self.tokenized_texts, self.labels)


    def tokenize_and_preserve_labels(self, sentence, text_labels):
        tokenized_sentence = []
        labels = []

        for word, label in zip(sentence, text_labels):
            tokenized_word = self.tokenizer.tokenize(str(word))
            n_subwords = len(tokenized_word)

            tokenized_sentence.extend(tokenized_word)
            labels.extend([label]*n_subwords)

        return tokenized_sentence, labels

    def tokenize_sentences_labels(self, sentences, labels):
        tokenized_texts_and_labels = [
            self.tokenize_and_preserve_labels(sent, labs)
            for sent, labs in zip(sentences, labels)
        ]
        
        tokenized_texts = [token_label_pair[0] for token_label_pair in tokenized_texts_and_labels]
        labels = [token_label_pair[1] for token_label_pair in tokenized_texts_and_labels]

        return tokenized_texts, labels

    def pad_tokenized_texts(self, tokenized_texts, labels):

        input_ids = pad_sequences([self.tokenizer.convert_tokens_to_ids(txt) for txt in tokenized_texts],
                                            maxlen=Parameters.MAX_LEN, 
                                            truncating="post", 
                                            padding="post", 
                                            dtype="long")
        tags = pad_sequences([[self.tag2idx.get(l) for l in label] for label in labels], 
                                            maxlen=Parameters.MAX_LEN, 
                                            truncating="post", 
                                            padding="post", 
                                            dtype="long")


        return input_ids, tags
