apt-get update
apt-get install -y libosmesa6 libosmesa6-dev libgl1 libglib2.0-0
export PYOPENGL_PLATFORM=osmesa
apt-get install -y libx11-6 libxext6 libxrender1 libxi6 libxrandr2 libxinerama1
ssh-keygen -t ed25519 -C "kenanblair2@gmail.com"
eval "$(ssh-agent -s)"
cat ~/.ssh/id_ed25519.pub