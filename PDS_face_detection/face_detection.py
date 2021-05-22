from hog import hog
import numpy as np
from skimage import data, color
from sklearn.datasets import fetch_lfw_people
import matplotlib.pyplot as plt
import seaborn as sns; sns.set()
import pickle
import time
from skimage import transform
from sklearn.feature_extraction.image import PatchExtractor
from itertools import chain
from sklearn.naive_bayes import GaussianNB
from sklearn.model_selection import cross_val_score

from multiprocessing import Pool, freeze_support, cpu_count
import argparse


ap = argparse.ArgumentParser()
ap.add_argument("-p", "--procs", type=int, default=-1, help="# of processes to spin up")
args = vars(ap.parse_args())

process = args["procs"] if args["procs"] > 0 else cpu_count()

realtime = time.time()

results = []
def get_hog(inputImg):
    fd=hog(inputImg)
    return fd

def poolCallback(returnDataFromPool):
    global results
    results.append(returnDataFromPool)




faces = fetch_lfw_people()
positive_patches = faces.images
positive_patches.shape

imgs_to_use = ['camera', 'text', 'coins', 'moon',
               'page', 'clock', 'immunohistochemistry',
               'chelsea', 'coffee', 'hubble_deep_field']
images = [color.rgb2gray(getattr(data, name)())
          for name in imgs_to_use]



def extract_patches(img, N, scale=1.0, patch_size=positive_patches[0].shape):
    extracted_patch_size = tuple((scale * np.array(patch_size)).astype(int))
    extractor = PatchExtractor(patch_size=extracted_patch_size,
                               max_patches=N, random_state=0)
    patches = extractor.transform(img[np.newaxis])
    if scale != 1:
        patches = np.array([transform.resize(patch, patch_size)
                            for patch in patches])
    return patches

negative_patches = np.vstack([extract_patches(im, 1000, scale)
                              for im in images for scale in [0.5, 1.0, 2.0]])

negative_patches.shape
fig, ax = plt.subplots(6, 10)
for i, axi in enumerate(ax.flat):
    axi.imshow(negative_patches[500 * i], cmap='gray')
    axi.axis('off')


if __name__ == "__main__":
    freeze_support()
    start = time.time()
    p = Pool(process)
    for im in chain(positive_patches,negative_patches):
        p.apply_async(get_hog, args=(im,), callback=poolCallback)
    p.close()
    p.join()


    end = time.time()
    X_train = np.array(results)
    print("[INFO] X_train hog took {} seconds".format((end - start)))
    # close the pool and wait for all processes to finish
    print("[INFO] waiting for processes to finish...")

    y_train = np.zeros(X_train.shape[0])
    y_train[:positive_patches.shape[0]] = 1

    print(X_train.shape)



    ss = cross_val_score(GaussianNB(), X_train, y_train)

    from sklearn.svm import LinearSVC
    from sklearn.model_selection  import GridSearchCV
    grid = GridSearchCV(LinearSVC(), {'C': [1.0, 2.0, 4.0, 8.0]})
    grid.fit(X_train, y_train)
    grid.best_score_

    grid.best_params_

    model = grid.best_estimator_
    model.fit(X_train, y_train)

    # save the model to disk
    filename = 'face_model.sav'
    pickle.dump(model, open(filename, 'wb'))

    realtime_end = time.time()

    print("[INFO] Real time run {} seconds".format((realtime_end - realtime)))


