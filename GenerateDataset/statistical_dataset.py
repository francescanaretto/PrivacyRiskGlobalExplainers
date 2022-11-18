import pickle
import pandas as pd
from sklearn.mixture import GaussianMixture
from sklearn.model_selection import train_test_split

HOMEDIR = "../Diva/"
classifier = "dt"
mode = "diva"
class_name = "y"
filename: str = f"{HOMEDIR}data/diva_original.sav"
dataset = pd.read_csv(filename)
#dataset.pop(class_name)
labels=dataset[class_name]
#dataset.pop('UserID')
print('dataset da usare ', dataset.head())
X_train,X_test,y_train,y_test= train_test_split(dataset.drop([class_name],axis=1), labels, test_size=0.6, random_state=101, stratify=labels)
print('gm ', X_train.shape)
gm = GaussianMixture(n_components=3, random_state=0).fit(X_train)
stat_dataset = gm.sample(n_samples=40000)
print('finito la creazione del stat dataset', stat_dataset[0].shape)
filename: str = f"{HOMEDIR}data/{mode}_stat_shadow.csv"
f = open(filename,'wb')
pickle.dump(stat_dataset,f)
f.close()
