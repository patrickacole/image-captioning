import torchvision.datasets as dset
import torchvision.transforms as transforms
cap = dset.CocoCaptions(root = 'train2017',
                        annFile = 'annotations/captions_train2017.json',
                        transform=transforms.ToTensor())

print('Number of samples: ', len(cap))
img, target = cap[3] # load 4th sample

print("Image Size: ", img.size())
print(target)

from sklearn.feature_extraction.text import CountVectorizer
# list of text documents
text = ["The quick brown fox jumped over the lazy dog."]
# create the transform
vectorizer = CountVectorizer(token_pattern=r"\b\w+\b", max_features=None)
# tokenize and build vocab
vectorizer.fit(target)
# summarize
print(vectorizer.vocabulary_)
print(vectorizer.get_feature_names(), len(vectorizer.get_feature_names()))
# encode document
vector = vectorizer.transform(["a little little boy girl girl girl"])
# summarize encoded vector
print(vector.shape)
print(type(vector))
print(vector.toarray())

## Now get dictionary for entire train set
# Resource links: https://machinelearningmastery.com/prepare-text-data-machine-learning-scikit-learn/
# Resource links: https://scikit-learn.org/stable/modules/generated/sklearn.feature_extraction.text.CountVectorizer.html
print("")
print(len(cap))
print("")
from tqdm import tqdm

all_captions = []
i = 0
for img, captions in tqdm(cap):
    for caption in captions:
        all_captions.append(caption)


vectorizer = CountVectorizer(token_pattern=r"\b\w+\b", max_features=12000)
vectorizer.fit(all_captions)
print(len(vectorizer.get_feature_names()))

# save vectorizor thingy
import pickle

vocab_filename = "vocab.pkl"
with open(vocab_filename, 'wb') as file:
    pickle.dump(vectorizer, file)