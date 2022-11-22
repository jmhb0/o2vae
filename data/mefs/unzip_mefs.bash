# move directory
SCRIPT_DIR=$( cd -- "$( dirname -- "${BASH_SOURCE[0]}" )" &> /dev/null && pwd )
cd ${SCRIPT_DIR}/mefs

pwd

tar xvzf mefs_not_scaled.tar.gz
tar xvzf mefs_scaled.tar.gz
