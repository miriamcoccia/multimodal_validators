export OLLAMA_FLASH_ATTENTION=1
export OLLAMA_KV_CACHE_TYPE="q8_0"
export OLLAMA_KEEP_ALIVE=-1
export OLLAMA_LOAD_TIMEOUT=600
export OLLAMA_DIR=/home/ldap/coccia@private.list.lu/oat_2024/libs/ollama_v09
export PATH=$PATH:$OLLAMA_DIR/bin/:$OLLAMA_LOAD_TIMEOUT

#systemctl --user daemon-reload
#systemctl --user enable ollama
#systemctl --user start ollama
