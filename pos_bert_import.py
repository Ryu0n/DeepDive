import os
import shutil
import subprocess
from glob import glob


if not os.path.exists('POSBert'):
    subprocess.run(['git', 'clone', 'https://github.com/Ryu0nPrivateProject/POSBert.git'])

for full_file_name in glob('POSBert/*'):
    file_name = './' + os.path.basename(full_file_name)
    if 'README.md' in file_name:
        continue
    if not os.path.exists(file_name):
        shutil.move(full_file_name, file_name)

shutil.rmtree('POSBert')
