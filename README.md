# Omega-Net

step1: config.yaml更改硬件配置，重点注意gpu_ids, num_processes

step2: 运行data/generator_raw_data.py产生train,valid,test数据集，默认放置同级目录的raw_data下

step3: 调整scripts下的shell脚本，调整model的超参数，详情见run.py

step4: bash {mode}.sh运行

step5: main编写了predict示例，建议服务器训练完，在本机上绘制查看效果

目前仅TEMDnet效果可以，SFSDSA训练效果很差，全力赶工TEMSGnet中...