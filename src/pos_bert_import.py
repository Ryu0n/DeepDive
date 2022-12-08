import os
import subprocess


if not os.path.exists('POSBert'):
    subprocess.run(['git', 'clone', 'https://github.com/Ryu0nPrivateProject/POSBert.git'])
