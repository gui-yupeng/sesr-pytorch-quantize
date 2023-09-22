#目前只支持4个PE
PE          =      4

QUAN_BIT    =      8

BIAS_BIT            =      19           #对量化后的bias(int)和bias(int)-Xzero*sum(Wint)同时做clamp
PE_ACC_BIT          =      18
PE_ADD_BIT          =      20

REQUAN_BIT          =      16
REQUAN_N_MAX        =      32




w_flg_c = True
# whether write weight txt and pt files 

WEIGHT_W_FLG            =   w_flg_c and True                #输出weight的txt
INPUT_W_FLG             =   w_flg_c and True                #输出input的pt文件
BIAS_W_FLG              =   w_flg_c and True                #输出bias量化后的pt文件
BIAS_QUAN_W_FLG         =   w_flg_c and True                #输出bias(int)-Xzero*sum(Wint)的pt文件
OUTPUT_PE_W_FLG         =   w_flg_c and True                #输出一个PE卷积后的输出pt文件
OUTPUT_PE_ADD_W_FLG     =   w_flg_c and True                #输出4个PE结果相加后的pt文件
REQUAN_FACTOR_W_FLG     =   w_flg_c and True                #重量化系数输出


# whether write histogram png files
WEIGHT_W_HIST_PNG       =   w_flg_c and True
INPUT_W_HIST_PNG        =   w_flg_c and True