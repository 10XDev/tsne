def test_seed():
    from tsne import bh_sne
    from sklearn.datasets import load_iris
    import numpy as np
    import os.path
    from os import environ

    if 'CI' in environ:
        # too many different compiler differences causing numeric instability.
        return

    iris = load_iris()

    X = iris.data
    y = iris.target

    t1 = bh_sne(X, random_state=np.random.RandomState(0), copy_data=True)
    t2 = np.load(os.path.join(os.path.dirname(__file__), "test_stability.npy"))

    assert np.all(t1 == t2)
