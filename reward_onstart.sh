#!/bin/bash
# Run source ~/.bashrc after this
# git clone https://github.com/surcyf123/dataset_enrichment && bash dataset_enrichment/reward_onstart.sh && source ~/.bashrc
cd dataset_enrichment
git pull
git checkout pierre/reward_endpoint_v1
pip3 install --upgrade pip
pip3 install flask nvitop torchmetrics transformers
sed -i "/^PATH='\/opt\/conda\/bin:\/usr\/local\/nvidia\/bin:\/usr\/local\/cuda\/bin:\/usr\/local\/sbin:\/usr\/local\/bin:\/usr\/sbin:\/usr\/bin:\/sbin:\/bin'$/s/^/#/" ~/.bashrc
curl -o- https://raw.githubusercontent.com/nvm-sh/nvm/v0.38.0/install.sh | bash
export NVM_DIR="$HOME/.nvm"
[ -s "$NVM_DIR/nvm.sh" ] && \. "$NVM_DIR/nvm.sh"
[ -s "$NVM_DIR/bash_completion" ] && \. "$NVM_DIR/bash_completion"
nvm install 14.21.3
nvm alias default 14.21.3
source ~/.bashrc
npm install -g pm2
chmod +x /root/dataset_enrichment/enrichment_pipeline/reward_endpoint/launch_reward_endpoints.sh
/root/dataset_enrichment/enrichment_pipeline/reward_endpoint/launch_reward_endpoints.sh
