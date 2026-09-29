import random
from collections import defaultdict, Counter
import json
import glob

class BigramcharModel:
    def __init__(self):
        self.counts = defaultdict(Counter)
        self.vocab = set()

    def train(self, text):
        text_lower = text.lower()
        self.vocab.update(text_lower)
        for c1, c2 in zip(text_lower, text_lower[1:]):
            self.counts[c1][c2]+=1


    def probability(self, c1, c2):
        total = sum(self.counts[c1].values())
        if total == 0:
            return 0.0
        return self.counts[c1][c2]/total
    
    def next_char(self, c1):
        if c1 not in self.counts or not self.counts[c1]:
            return random.choice(list(self.vocab))
        chars, weights=zip(*self.counts[c1].items())
        return random.choices(chars, weights=weights, k=1)[0]

    def generate(self, start_char, len_max=100):
        text = start_char
        for _ in range(len_max-1):
            text+=self.next_char(text[-1])
        return text
    
class BigramwordModel:
    def __init__(self):
        self.counts = defaultdict(Counter)
        self.vocab = set()

    def train(self, text):
        text_lower_split   = text.lower().split()
        self.vocab.update(text_lower_split)
        for w1, w2 in zip(text_lower_split, text_lower_split[1:]):
            self.counts[w1][w2]+=1

    def next_word(self, w1):
        if w1 not in self.counts or not self.counts[w1]:
            return random.choice(list(self.vocab))
        words, weights=zip(*self.counts[w1].items())
        return random.choices(words, weights=weights, k=1)[0]

    def generate(self, start_word, len_max=100):
        text = start_word
        for _ in range(len_max-1):
            text+=self.next_word(text[-1])
        return text


class TrigramModel:
    def __init__(self, alpha=1.0):
        self.counts=defaultdict(Counter)
        self.vocab=set()
        self.alpha=alpha

    def train(self, text):
        words = text.lower().split()
        self.vocab.update(words)
        for w1, w2, w3 in zip(words, words[1:], words[2:]):
            self.counts[(w1, w2)][w3]+=1

    def probability(self, w1, w2, w3):
        """P(w3 | w1, w2) avec lissage de Laplace, même formule que le
        bigramme mais conditionnée sur la PAIRE (w1, w2)."""
        V = len(self.vocab)
        total = sum(self.counts[(w1, w2)].values())
        count = self.counts[(w1, w2)][w3]
        return (count + self.alpha) / (total + self.alpha * V)

    def next_word(self, w1, w2):
            vocab_list = list(self.vocab)
            weights=[]
            for w3 in vocab_list:
                poids=self.counts[(w1,w2)][w3]+self.alpha
                weights.append(poids)
            return random.choices(vocab_list, weights=weights, k=1)[0]

    def generate(self, start_tuple, len_max=100):
        w1= start_tuple[0]
        w2= start_tuple[1]
        text=[w1, w2]

        for _ in range(len_max-1):
            w3=self.next_word(w1, w2)
            text.append(w3)
            w1,w2=w2,w3 #on fait glisser la fenetrede contexte
        return " ".join(text)


def load_data(path):
    real_path=path+"""/*.txt"""
    fichiers = glob.glob(real_path, recursive=True)
    real_corpus=''
    for f in fichiers:
        name_livre=f.split('\\')[-1]
        with open(f, encoding="utf-8") as f:        
            real_corpus+=f'''############## DEBUT DE {name_livre}'''
            real_corpus += f.read()
            real_corpus+=f'''############## FIN DE {name_livre}'''
    return real_corpus


if __name__ == "__main__":
    
    data_train = """
    le chat mange le poisson. le chien mange la viande.
    le chat dort sur le tapis. le chien dort sur le lit.
    le petit chat joue avec la petite souris.
    le grand chien joue avec le grand chat.
    """

    #use real 3 books for training
    #data_train = load_data("data_gutenberg")

    #train bigram model on letter
    # model = BigramcharModel()
    # model.train(corpus)
    #print(model.generate('l'))
    #print("counts:", json.dumps(model.counts,  indent=1, ensure_ascii=False))

    #train bigram model on words
    # model2 = BigramwordModel()
    # model2.train(corpus)
    # print(model2.generate('je'))

    #trigram model
    model_trigram=TrigramModel()
    model_trigram.train(data_train)
    # print("counts:", model_trigram.counts)
    print(model_trigram.generate(('the', 'cat')))
    print("P('chat', 'mange' -> 'le') =",round(model_trigram.probability("chat", "mange", "le"), 3))
