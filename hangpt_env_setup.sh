conda create -n fmfl python=3.8 -y
conda activate fmfl
pip install torch==2.1.0+cu121 torchvision==0.16.0+cu121 torchaudio==2.1.0+cu121 -f https://download.pytorch.org/whl/cu121/torch_stable.html
pip install -r requirements.txt