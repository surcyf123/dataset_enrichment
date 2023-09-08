#!/bin/bash


# curl -sL https://deb.nodesource.com/setup_14.x | sudo -E bash - && \
# sudo apt-get install -y nodejs && \
# sudo npm install -g npm && \
# npm install -g pm2

# Removing existing nodejs and npm
sudo apt-get purge nodejs npm -y

# Installing nvm
curl -o- https://raw.githubusercontent.com/nvm-sh/nvm/v0.38.0/install.sh | bash

# Activating nvm
source ~/.bashrc

# Installing specific node version
nvm install 14.21.3

# Installing pm2 globally
npm install -g pm2

pip install flask