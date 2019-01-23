# Sequence-To-Sequence-RNN-TensorFlow-Frame
A Simple Sequence-to-Sequence Framework. It is based on TensorFlow's RRN and tf.contrib.seq2seq.Basic Decoder modules.


# Usage method

## 1. Preparation data
One data per row, two sequential file data correspond one by one. Refer specifically to the contents under the data file.

Sequnce file A like:

```
new jersey is sometimes quiet during autumn , and it is snowy in april .
the united states is usually chilly during july , and it is usually freezing in november .
california is usually quiet during march , and it is usually hot in june .
the united states is sometimes mild during june , and it is cold in september .
your least liked fruit is the grape , but my least liked is the apple .
his favorite fruit is the orange , but my favorite is the grape .
paris is relaxing during december , but it is usually chilly in july .
```

Sequnce file B like:

```
new jersey est parfois calme pendant l' automne , et il est neigeux en avril .
les états-unis est généralement froid en juillet , et il gèle habituellement en novembre .
california est généralement calme en mars , et il est généralement chaud en juin .
les états-unis est parfois légère en juin , et il fait froid en septembre .
votre moins aimé fruit est le raisin , mais mon moins aimé est la pomme .
son fruit préféré est l'orange , mais mon préféré est le raisin .
paris est relaxant en décembre , mais il est généralement froid en juillet .
```
## 2. ```python data_process.py```

## 3. ```python run_model_batch_size.py```


## Explain
seq2seq_RNN_model.py and seq2seq_RNN_model_batch_size.py work the same way, but seq2seq_RNN_model_batch_size.py needs to determine the batch size first, seq2seq_RNN_model.py can dynamically adjust the batch size. But at present, they are quite different in the validation set. I have recorded them in file analysis_log.

TODO: Find out the reason why seq2seq_RNN_model.py does not work well on the verification set, and fix the code BUG.


