# ML Ops Repo

# Assigment 6: Running Tests for SVM model

----------------------

### TODO: write  a test case to check if model is successfully getting created or not?
```
def test_model_writing():

    1. create some data

    2. run_classification_experiment(data, expeted-model-file)

    assert os.path.isfile(expected-model-file)
```

### TODO: write a test case to check fitting on training -- litmus test.

```
def test_small_data_overfit_checking():

    1. create a small amount of data / (digits / subsampling)

    2. train_metrics = run_classification_experiment(train=train, valid=train)

    assert train_metrics['acc']  > some threshold

    assert train_metrics['f1'] > some other threshold
```
---------------------------

```python
======================================= test session starts ========================================
platform linux -- Python 3.6.9, pytest-6.2.5, py-1.10.0, pluggy-1.0.0
rootdir: /media/vinitgore/Workplace/MTech/Developer Notes/MTechYear2Sem1/MLOps/mnist_example
collected 6 items                                                                                  

tests/model_writing_test.py ...                                                              [ 50%]
tests/sample_test.py ..F                                                                     [100%]

============================================= FAILURES =============================================
____________________________________________ test_sqrt _____________________________________________

    def test_sqrt():
        num=1
>       assert(num==2*2)
E       assert 1 == (2 * 2)

tests/sample_test.py:12: AssertionError
========================================= warnings summary =========================================
tests/model_writing_test.py: 5391 warnings
  /home/vinitgore/miniconda3/envs/mlops/lib/python3.6/site-packages/skimage/util/dtype.py:226: DeprecationWarning: Converting `np.inexact` or `np.floating` to a dtype is deprecated. The current result is `float64` which is not strictly correct.
    dtypeobj_out = np.dtype(dtype)

-- Docs: https://docs.pytest.org/en/stable/warnings.html
===================================== short test summary info ======================================
FAILED tests/sample_test.py::test_sqrt - assert 1 == (2 * 2)
=========================== 1 failed, 5 passed, 5391 warnings in 15.38s ============================
```