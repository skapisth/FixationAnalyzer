import yaml
from collections import OrderedDict
import itertools
import fixationanalyzer as fa


def main():
    with open('config.yaml','r') as f:
        CONFIGS = yaml.load(f)

    sorted_keys = sorted( CONFIGS.keys() )
    ordered_configs = OrderedDict()
    for key in sorted_keys:
        ordered_configs[key] = CONFIGS[key]

    arguments_list = ordered_configs.values()

    all_permutations = list(itertools.product(*arguments_list))
    num_perm = len(all_permutations)
    print("{} Permutations from configs generated!".format(num_perm))


    for perm_idx,perumation in enumerate(all_permutations):
        perm_idx += 1
        kwargs = dict( zip(sorted_keys,perumation) )
        kwargs['DECISION_FUNCTION_TYPE'] = 'ovo' if CONFIGS['CLASSIFICATION_TYPE'] == 'four' else 'ovr'
        kwargs['PERM_IDX'] = perm_idx
        kwargs['NUM_PERM'] = num_perm
        print("----------------BEGINNING NEW TEST WITH THE FOLLOWING KWARGS------------------")
        [print('\t',k,' : ',v) for k,v in kwargs.items()]
        fa.test(kwargs)

if __name__ == '__main__':
    main()
