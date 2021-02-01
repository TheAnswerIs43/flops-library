Per buildare la libreria 

- python setup.py bdist_wheel

Si crea una cartella "dist", il file wheelfile.whl si trova in quella cartella, spostati nella cartella

- pip install /path/to/wheelfile.whl

Una volta installata la libreria può essere utilizzata

- import myflopslib
- from myflopslib.profiler import Profiler

L'oggetto Profiler calcolerà il numero complessivo di FLOPs di un modello

- profiler = Profiler()
- profiler.compute_flops(NOME_MODELLO)