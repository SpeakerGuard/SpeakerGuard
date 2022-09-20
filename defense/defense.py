
import defense.time_domain as TD
import defense.frequency_domain as FD
import defense.speech_compression as SC
import defense.feature_level as FL

Input_Transformation = [

    'QT', 'AT', 'AS', 'MS', # Time Domain
    'DS', 'LPF', 'BPF', # Frequency Domain
    'OPUS', 'SPEEX', 'AMR', 'AAC_V', 'AAC_C', 'MP3_V', 'MP3_C', # Speech Compression
    'FEATURE_COMPRESSION', # Feature-Level,; Ours
    'FeCo', # Feature-Level,; Ours; abbr,
] 

Robust_Training = [
    'AdvT' # adversarial training
]

def parser_defense(defense, defense_param, defense_flag, defense_order):
    ''' defense: list of str, e.g., ['AT', 'QT', 'FeCo']
        defense_param: list of str, e.g., ['16', '512', "kmeans 0.2 L2"]
        defense_param: list of int, e.g., [0, 0, 1]
    '''
    # print(defense, defense_param, defense_flag, defense_order) 
    if defense is not None:
        if defense_param is None:
            defense_param = [None] * len(defense)
        assert len(defense) == len(defense_param)
        assert len(defense_param) == len(defense_flag)
        my_defense = []
        defense_name = ""
        for x, y, z in zip(defense, defense_param, defense_flag):
            if y is not None:
                f = lambda_defense(x, y.split(' '))
                my_defense.append([z, f])
                defense_name = defense_name + '{}&{}@{}+'.format(x, y.replace(' ', '#'), z) if defense_order == 'sequential' else \
                                defense_name + '{}&{}@{}$'.format(x, y.replace(' ', '#'), z)
            else:
                f = lambda_defense(x, None)
                my_defense.append([z, f])
                defense_name = defense_name + '{}&{}@{}+'.format(x, 'DEFAULT', z) if defense_order == 'sequential' else \
                                defense_name + '{}&{}@{}$'.format(x, 'DEFAULT', z)
        defense_name = defense_name[:-1]
        defense_name = defense_name.replace('.', '_')
        # print(defense_name)
    else:
        my_defense = None
        defense_name = None
    return my_defense, defense_name


def lambda_defense(defense, defense_param):
    ''' defense: str
        defense_param: list
    '''
    if defense is None:
        return lambda x: x
    
    if hasattr(TD, defense):
        ori_f_source = TD
    elif hasattr(FD, defense):
        ori_f_source = FD
    elif hasattr(SC, defense):
        ori_f_source = SC
    elif hasattr(FL, defense):
        ori_f_source = FL
    else:
        raise NotImplementedError('Upsupported Defense Method')
    ori_f = getattr(ori_f_source, defense)
    
    if defense == 'FeCo' or defense == 'FEATURE_COMPRESSION': # cl_m, point, ratio, other_param (L2, cos, ts, random) 
        cl_m = defense_param[0] 
        cl_r = float(defense_param[1])
        other_param = defense_param[2]
        f = lambda x: ori_f(x, method=cl_m, param=cl_r, other_param=other_param)
    else:
        if defense == 'BPF':
            defense_param = [float(defense_param[0]), float(defense_param[1])]
        elif defense == 'DS':
            defense_param = float(defense_param[0])
        else:
            defense_param = int(defense_param[0])
        f = lambda x: ori_f(x, param=defense_param) 
    return f
