#!/bin/bash

python create_db_300W_AFLW.py --db '../../data/300W_LP/AFW' --output './AFW.npz' --img_size 256 --ad 0.6
python create_db_300W_AFLW.py --db '../../data/300W_LP/AFW_Flip' --output './AFW_Flip.npz' --img_size 256 --ad 0.6
python create_db_300W_AFLW.py --db '../../data/300W_LP/HELEN' --output './HELEN.npz' --img_size 256 --ad 0.6
python create_db_300W_AFLW.py --db '../../data/300W_LP/HELEN_Flip' --output './HELEN_Flip.npz' --img_size 256 --ad 0.6
python create_db_300W_AFLW.py --db '../../data/300W_LP/IBUG' --output './IBUG.npz' --img_size 256 --ad 0.6
python create_db_300W_AFLW.py --db '../../data/300W_LP/IBUG_Flip' --output './IBUG_Flip.npz' --img_size 256 --ad 0.6
python create_db_300W_AFLW.py --db '../../data/300W_LP/LFPW' --output './LFPW.npz' --img_size 256 --ad 0.6
python create_db_300W_AFLW.py --db '../../data/300W_LP/LFPW_Flip' --output './LFPW_Flip.npz' --img_size 256 --ad 0.6


python create_db_300W_AFLW.py --db '../../data/AFLW2000' --output './AFLW2000.npz' --img_size 256 --ad 0.6