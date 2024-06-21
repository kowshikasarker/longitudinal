import os

met_path = '/shared/nas/data/m1/ksarker2/Embedding/Data/Metabolome/base_subtracted_from_end_normalized.tsv'

react_to_sub_map = '/shared/nas/data/m1/ksarker2/Embedding/Data/Human-GEM-1.18.0/human_gem_reaction_subsystem_mapping.tsv'

script = '/shared/nas/data/m1/ksarker2/Embedding/Code/construct-homogenous-network.py'

for react_set in range(1, 10):
    react_set_path = '/shared/nas/data/m1/ksarker2/Embedding/Data/Human-GEM-1.18.0/human_gem_reaction_set_' + str(react_set) + '.tsv'
    
    out_dir = '/shared/nas/data/m1/ksarker2/Embedding/Results/homogenous/reaction_set_' + str(react_set)
    
    command = 'python3 ' + script + ' -m ' + met_path + ' -m2r ' + react_set_path + ' -r2s ' + react_to_sub_map + ' -o ' + out_dir
    
    print(command)
    exit_code = os.system(command)
    if(exit_code != 0):
        print('ERROR!', command)
    
    
    