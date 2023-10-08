
# One Click Pipeline to Deploy a Model on VastAI and Run Experiments
This is a module that contains everything needed to run big experiments


### Install Requirements
```bash
pip3 install -r requirements.txt
```

### Set the keys and such

```python
active_branch = "ethan/command-line-prompt-formatting" # Sets the active branch
VAST_API_KEY = "dd582e01b1712f13d7da8dd6463551029b33cff6373de8497f25a2a03ec813ad" # Your Vast API key
pkey = paramiko.RSAKey.from_private_key_file("../../credentials/autovastai") # This key should be able to both SSH into the instance and have access to the github repos too
```

### Launch Args
```python
use_fmt_file = bool(sys.argv[1])
fmt_file_path = sys.argv[2] # This is optional, it will use fmtEXAMPLE.json instead if no path is provided
```