import numpy as np
from tqdm import tqdm

def data_loader_synthetic(p=.2, q=.2, r=.2, s=.2, rank=10, seed=42):
    '''
    ground truth matrix Y
    '''
    num_users, num_items = 600, 400
    n1, n2 = num_users // 2, num_items // 2
    np.random.seed(42)
    Y_1 = np.where(np.random.random((rank, n2)) < p, 1, -1)
    np.random.seed(42)
    Y_2 = np.where(np.random.random((rank, n2)) < q, 1, -1)
    Y_rank = np.concatenate((Y_1, Y_2), axis = 1)

    Y_m = Y_rank.copy()

    for i in range(num_users // (rank * 2) -1):
        Y_m = np.concatenate((Y_m, Y_rank))
    np.random.seed(43)
    Y_1 = np.where(np.random.random((rank, n2)) < q, 1, -1)
    np.random.seed(43)
    Y_2 = np.where(np.random.random((rank, n2)) < p, 1, -1)
    Y_rank = np.concatenate((Y_1, Y_2), axis = 1)

    Y_f = Y_rank.copy()

    for i in range(num_users // (rank * 2) -1):
        Y_f = np.concatenate((Y_f, Y_rank))
    
    np.random.shuffle(Y_m)
    np.random.shuffle(Y_f)
    Y = np.concatenate((Y_m, Y_f))
    
    I_obs_mm = np.where(np.random.random((n1, n2)) < r, 1, 0)
    I_obs_mf = np.where(np.random.random((n1, n2)) < s, 1, 0)
    I_obs_fm = np.where(np.random.random((n1, n2)) < s, 1, 0)
    I_obs_ff = np.where(np.random.random((n1, n2)) < r, 1, 0)

    I_obs_m = np.concatenate((I_obs_mm, I_obs_mf), axis = 1)
    I_obs_f = np.concatenate((I_obs_fm, I_obs_ff), axis = 1)

    I_obs = np.concatenate((I_obs_m, I_obs_f))
    
    Y_obs = Y * I_obs
    
    Y_train, Y_test = Y_obs, (Y-Y_obs)
#     Y_train, Y_test = np.zeros((num_users, num_items)), np.zeros((num_users, num_items))

#     for i in tqdm(range(num_users)):
#         for j in range(num_items):
#             if Y_obs[i, j] != 0:
#                 k = np.random.random()
#                 if k > 0.9:
#                     Y_test[i, j] = Y_obs[i, j]
#                 else:
#                     Y_train[i, j] = Y_obs[i, j]
                    
                    
    user, item = {}, {}
    user['M'] = [x for x in range(n1)]
    user['F'] = [x for x in range(n1, n1*2)]
    item['M'] = [x for x in range(n2)]
    item['F'] = [x for x in range(n2, n2*2)]
                    
    return (Y_train, Y_test), user, item
    
    
    
    
    
    
    
    
    
    
    