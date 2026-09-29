import torch
import torch.nn as nn
import torch.nn.functional as F
from bigram_trigram import load_data

#print(torch.__version__)

train_data = load_data('data_gutenberg').lower().split()
train_data=train_data[:10000]
#construis le vocab
vocab=sorted(set(train_data))
#idx veut dire index donc la position du mot dans la liste
#enumerate crée une liste de tuples (mot,index)
word_to_idx={w: i for i,w in enumerate(vocab)}
idx_to_word={i: w for i,w in enumerate(vocab)}
V=len(vocab)

print(f'Vocabulaire de {V} mots : {vocab}')

#construire les PAIRES
#zip() s'arrete a la taille de la liste la plus courte
paires= [(word_to_idx[w1], word_to_idx[w2]) for w1,w2 in zip(train_data, train_data[1:])]

# "Un tensor, c'est un tableau de nombres à N dimensions (comme une liste, une matrice, ou plus), "
# "utilisé par PyTorch parce qu'il peut suivre automatiquement les calculs pour permettre l'entraînement "
# "par descente de gradient — et il est optimisé pour tourner rapidement sur GPU."
inputs=torch.tensor([p[0] for p in paires])
targets=torch.tensor([p[1] for p in paires])

print(f"Nombre de bigrammes d'entraînement : {len(paires)}")

#Création du modèle
class NeuronalBigram(nn.Module):
    def __init__(self, vocab_size, embed_dim=8):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, embed_dim)
        self.linear = nn.Linear(embed_dim, vocab_size)

    def forward(self, word_idx):
        emb=self.embedding(word_idx)
        logits=self.linear(emb)
        return logits 

    def probability_distribution(self, word_idx):
        with torch.no_grad():
            logits=self.forward(torch.tensor([word_idx]))
            return F.softmax(logits, dim=1).squeeze()
        
    def next_word(self, word_idx):
        probs=self.probability_distribution(word_idx)
        return torch.multinomial(probs, num_samples=1).item()

    def generate(self, start_word, lenght_max=15):
        idx=word_to_idx[start_word]
        text=[start_word]
        for _ in range(lenght_max-1):
            idx_next_word = self.next_word(idx)
            newt_word=idx_to_word[idx_next_word]
            text.append(newt_word)
            idx=idx_next_word
        return " ".join(text)

#Training du modèle
model=NeuronalBigram(vocab_size=V, embed_dim=8)
optimizer=torch.optim.Adam(model.parameters(), lr=0.1)

print()
print("=" * 60)
print("ENTRAINEMENT")
print("=" * 60)

for epoch in range(1000):
    logits=model(inputs)
    loss=F.cross_entropy(logits, targets)

    optimizer.zero_grad()
    loss.backward()
    optimizer.step()

    if epoch % 40 == 0:
        print(f"epoch : {epoch} | loss = {loss.item()}")
print(f'epoch final | loss = {loss.item()}')


#check proba
probs=model.probability_distribution(word_to_idx["he"])
print(f"P_neuronale('il' -> 'est') = {round(probs[word_to_idx['is']].item(), 3)}")

#use the model
print(f"texte généré : {model.generate(start_word='he', lenght_max=30)}")