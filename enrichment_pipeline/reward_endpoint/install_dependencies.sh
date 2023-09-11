#!/bin/bash

##RUN SOURCE ~/.bashrc AFTER THIS!, if having problems, make sure to check ~/.bashrc for path resets or anything else messing it up

# if not broken packages- basically impossible with vast
# curl -sL https://deb.nodesource.com/setup_14.x | sudo -E bash - && \
# sudo apt-get install -y nodejs && \
# sudo npm install -g npm && \
# npm install -g pm2

# if broken packages, probably will need to do this
# Removing existing nodejs and npm
sudo apt-get purge nodejs npm -y

# Installing nvm
curl -o- https://raw.githubusercontent.com/nvm-sh/nvm/v0.38.0/install.sh | bash

# Initialize NVM for the current session
export NVM_DIR="$HOME/.nvm"
[ -s "$NVM_DIR/nvm.sh" ] && \. "$NVM_DIR/nvm.sh"
[ -s "$NVM_DIR/bash_completion" ] && \. "$NVM_DIR/bash_completion"

# Installing specific node version
nvm install 14.21.3
nvm alias default 14.21.3
source ~/.bashrc
# Installing pm2 globally
npm install -g pm2
