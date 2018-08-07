import os
import shutil

in_dir = '/home/xamyzhao/datasets/watercolors_v2/demasiadomar-Night_Sky'

frames = [os.path.join(in_dir, f) for f in os.listdir(in_dir)]

for f in frames:
	shutil.move(f, f.replace('Nigth','Night'))
