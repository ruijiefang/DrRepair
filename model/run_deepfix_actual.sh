### Base + graph + pretrain ###
#first, pretrain
name="code-compiler--2l-graph--pretrain"
mkdir -p out_deepfix/${name}
python3 -u main_deepfix.py -o ${name} train \
    configs/base.yml  configs/data-deepfix/err-data-extra.yml \
    configs/model-code-compiler/2l-graph--dec-attn-all.yml \
    > out_deepfix/${name}/log.txt 2>&1
#then fine tune
name="code-compiler--2l-graph--finetune"
mkdir -p out_deepfix/${name}
python3 -u main_deepfix.py -o ${name} train \
    -l out_deepfix/code-compiler--2l-graph--pretrain/400000 \
    configs/base.yml  configs/data-deepfix/err-data-orig-finetune.yml \
    configs/model-code-compiler/2l-graph--dec-attn-all.yml \
    > out_deepfix/${name}/log.txt 2>&1