#!/bin/bash

# Function to get GPU ID
get_gpu() {
    # echo $((RANDOM % 8))
    echo 0
}


FET_HOME="."

# Main execution
datasets=("gisette" "mnist")
n_classes=(2 10)


for ((i=0; i<${#datasets[@]}; i++)); do
    dataset=${datasets[i]}
    n_class=${n_classes[i]}
    
    LOG_DIR="${FET_HOME}/out/scale/${dataset}"
    mkdir -p "${LOG_DIR}"
    
    for seed in {0..4}; do
        for party in 10 20 30 40 50; do
            noise=0.05
            
            # Solo
            gpu=$(get_gpu)
            python "${FET_HOME}/src/script/train_solo.py" \
                -d "${dataset}" -c "${n_class}" -m "acc" -p "${party}" \
                -s "${seed}" -w 100 --key-noise "${noise}" \
                -g "${gpu}" \
                > "${LOG_DIR}/scaletest-solo_${dataset}_c${n_class}-noise${noise}_macc_party${party}_seed${seed}.txt" 

            # FeT
            gpu=$(get_gpu)
            python "${FET_HOME}/src/script/train_fet.py" \
                -d "${dataset}" -m "acc" -c "${n_class}" -p "${party}" \
                -s "${seed}" --knn-k 100 -w 100 -nh 1 -ded 200 -ked 200 \
                -nlb 1 -nab 1 --dropout 0.0 --key-noise "${noise}" --party-dropout 0.6 -paf 1 \
                -g "${gpu}" \
                > "${LOG_DIR}/scaletest-fet_${dataset}_c${n_class}-noise${noise}_macc_party${party}_seed${seed}_k100_nh1_ded200_ked200_nAb1_nLb1_dropOut0.0.txt" 
        done
    done
done

echo "All tasks completed"
