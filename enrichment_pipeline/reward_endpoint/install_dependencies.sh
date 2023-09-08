#!/bin/bash

# curl -sL https://deb.nodesource.com/setup_14.x | sudo -E bash - && \
# sudo apt-get install -y nodejs && \
# sudo npm install -g npm && \
# npm install -g pm2


# Removing existing nodejs and npm
sudo apt-get purge nodejs npm -y

# Installing nvm
curl -o- https://raw.githubusercontent.com/nvm-sh/nvm/v0.38.0/install.sh | bash

# Append nvm initialization to .bashrc
echo -e "\n# NVM initialization" >> ~/.bashrc
echo 'export NVM_DIR="$HOME/.nvm"' >> ~/.bashrc
echo '[ -s "$NVM_DIR/nvm.sh" ] && \. "$NVM_DIR/nvm.sh"  # This loads nvm' >> ~/.bashrc
echo '[ -s "$NVM_DIR/bash_completion" ] && \. "$NVM_DIR/bash_completion"  # This loads nvm bash_completion' >> ~/.bashrc
echo 'nvm use 14.21.3 > /dev/null 2>&1' >> ~/.bashrc

# Installing specific node version
nvm install 14.21.3
nvm alias default 14.21.3

# Source .bashrc to apply changes in this session
source ~/.bashrc

# Installing pm2 globally
npm install -g pm2

pip install flask
pip install nvitop
pip install torchmetrics
