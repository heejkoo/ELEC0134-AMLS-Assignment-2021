import os
from utils import data_preprocessing
from A1 import A1_DL
from A2 import A2_DL
from B1 import B1_DL
from B2 import B2_DL

# Only the best performing models are trained again for reproduce #
# You can run the files in each directory to derive results on each model #
# ======================================================================================================================
# Task A1
os.chdir('./A1/')

a1_train, a1_val, a1_test = data_preprocessing('celeba', 'gender')
model_A1 = A1_DL.A1(a1_train, a1_val, a1_test, model='custom', gap=False, init='xavier', num_epochs=50)
acc_A1_train = model_A1.train()
acc_A1_test = model_A1.test()

# ======================================================================================================================
# Task A2
os.chdir('..')
os.chdir('./A2/')

a2_train, a2_val, a2_test = data_preprocessing('celeba', 'smiling')

model_A2 = A2_DL.A2(a2_train, a2_val, a2_test, model='custom', gap=False, init='he', num_epochs=50)
acc_A2_train = model_A2.train()
acc_A2_test = model_A2.test()

# ======================================================================================================================
# Task B1
os.chdir('..')
os.chdir('./B1/')

b1_train, b1_val, b1_test = data_preprocessing('cartoon_set', 'face_shape')

model_B1 = B1_DL.B1(b1_train, b1_val, b1_test, model='custom', gap=False, init='he', num_epochs=50)
acc_B1_train = model_B1.train()
acc_B1_test = model_B1.test()

# ======================================================================================================================
# Task B2
os.chdir('..')
os.chdir('./B2/')

b2_train, b2_val, b2_test = data_preprocessing('cartoon_set', 'eye_color')

model_B2 = B2_DL.B2(b2_train, b2_val, b2_test, model='custom', gap=False, init='he', num_epochs=50)
acc_B2_train = model_B2.train()
acc_B2_test = model_B2.test()

# ======================================================================================================================

print('TA1:{},{};TA2:{},{};TB1:{},{};TB2:{},{};'.format(acc_A1_train, acc_A1_test,
                                                        acc_A2_train, acc_A2_test,
                                                        acc_B1_train, acc_B1_test,
                                                        acc_B2_train, acc_B2_test))