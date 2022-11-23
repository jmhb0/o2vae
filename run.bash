SCRIPT_DIR=$( cd -- "$( dirname -- "${BASH_SOURCE[0]}" )" &> /dev/null && pwd )
SAVE_DIR=${SCRIPT_DIR}/logfiles/

# get the experiment number based on what logfiles are in the directory
EXP_NUM=$(ls ${SAVE_DIR} | wc -l)
EXP_NUM=$((${EXP_NUM}+1))
OUT_FILE=${SAVE_DIR}logfile_${EXP_NUM}.out
echo "logging to " $OUT_FILE
echo 

# send the stdout and stderr to the log file
python run.py configs.config_o2mnist 2>&1 | tee $OUT_FILE
