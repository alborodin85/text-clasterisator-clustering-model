import sklearn
from sklearn import datasets

dataHome = r'C:\borodin_admin\Институт\_ВКР\2022-06-14 Приложение\clustering-model\fetch_20newsgroups'
subset = 'all'
randomState = 1
categories = ['alt.atheism', 'comp.graphics', 'comp.os.ms-windows.misc', 'comp.sys.ibm.pc.hardware', 'comp.sys.mac.hardware', 'comp.windows.x', 'misc.forsale', 'rec.autos', 'rec.motorcycles', 'rec.sport.baseball', 'rec.sport.hockey', 'sci.crypt', 'sci.electronics', 'sci.med', 'sci.space', 'soc.religion.christian', 'talk.politics.guns', 'talk.politics.mideast', 'talk.politics.misc', 'talk.religion.misc']
# categories = ['alt.atheism']

fetch20Newsgroups = datasets.fetch_20newsgroups(data_home=dataHome, subset=subset, random_state=randomState, categories=categories)

fileName = fetch20Newsgroups.filenames[5]

print(fetch20Newsgroups.__class__)
print(fetch20Newsgroups.data[0])
print(fetch20Newsgroups.filenames)
