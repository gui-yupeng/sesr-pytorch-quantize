MFLAG     =     3
TEST_RAW_ADD_NOISE = False

#目前只支持4个PE
PE          =      4

QUAN_BIT    =      8

BIAS_BIT            =      16           #对量化后的bias(int)和bias(int)-Xzero*sum(Wint)同时做clamp
PE_ACC_BIT          =      18
PE_ADD_BIT          =      20

REQUAN_BIT          =      16
REQUAN_N_MAX        =      32

# 舍弃KL
# 是否使用KL熵校准，这个函数应该仅在执行KL校准脚本时为True，其余任何时间都应该为False
# USE_KL_CALI = False
# BINS_NUM = 2048
# TGT_BINS_NUM = 128


w_flg_c = True

WEIGHT_W_FLG            =   w_flg_c and True                #输出weight的txt
INPUT_W_FLG             =   w_flg_c and True                #输出input的Tensor
BIAS_W_FLG              =   w_flg_c and True                #输出bias量化后的pt文件
BIAS_QUAN_W_FLG         =   w_flg_c and True                #输出bias(int)-Xzero*sum(Wint)的pt文件
OUTPUT_PE_W_FLG         =   w_flg_c and True                #输出一个PE卷积后的输出Tensor
OUTPUT_PE_ADD_W_FLG     =   w_flg_c and True                #输出4个PE结果相加后的Tensor
REQUAN_FACTOR_W_FLG     =   w_flg_c and True                #重量化系数输出int


# whether write histogram png files
WEIGHT_W_HIST_PNG       =   w_flg_c and False
INPUT_W_HIST_PNG        =   w_flg_c and False
