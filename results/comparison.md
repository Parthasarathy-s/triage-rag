# Model comparison (test split)

```
                          model    n  accuracy  macro_f1  ms_per_ticket
Zero-shot BART-MNLI (subsample)  200    0.5350    0.4842        63.5426
                TF-IDF + LogReg 2428    0.9967    0.9965            NaN
          Fine-tuned DistilBERT 2428    0.9992    0.9991         2.2415
```


## Fine-tuned confusion matrix (rows = true, cols = predicted)

```
          account  billing  general  shipping
account       536        0        0         0
billing         0      731        0         0
general         0        0      723         0
shipping        0        0        2       436
```


8 test tickets are misclassified by at least one model (see errors.csv).


## Retrieval (top-1, 500 test tickets)

same intent: 0.994, same label: 0.996
