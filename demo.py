import os
os.system('python demoMUUFL.py')
#####################convlution kernel test
# BATCH_SIZES=[]
# num_result=1
# for BATCH in BATCH_SIZES:
#     os.system('python MUUFL-20200601.py --conv_size {}'\
#     .format(BATCH))
#     num_result+=1

#####################fractional order test
# BATCH_SIZES=[5,10,15,20,25,30,35,40,45,50]
# num_result=1
# for BATCH in BATCH_SIZES:
#     os.system('python MUUFL-20200601.py --order {}'\
#     .format(BATCH))
#     num_result+=1

################learning rate test
# BATCH_SIZES=[1,5,10,50,100,500,1000,5000]###*0.00001
# num_result=1
# for BATCH in BATCH_SIZES:
#     os.system('python MUUFL-20200601.py --lr {}'\
#     .format(BATCH))
#     num_result+=1

################alpha test
# BATCH_SIZES=[1,2,3,4,5,6,7,8,9]###*0.1
# num_result=1
# for BATCH in BATCH_SIZES:
#     os.system('python MUUFL-20200601.py --alpha {}'\
#     .format(BATCH))
#     num_result+=1

################sample number test
# BATCH_SIZES=[20,40,60,80]###*0.1
# num_result=1
# for BATCH in BATCH_SIZES:
#     os.system('python MUUFL-20200601.py --numsample {}'\
#     .format(BATCH))
#     num_result+=1