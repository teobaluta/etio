import os
import sys
import csv
import statistics

def parse_results_ml_leaks(logfile):
    results_dict = {
        'target': -1,
        'shadow': -1,
        'nm': -1,
        'train-attack-acc': -1,
        'overall-attack-acc': -1,
        'mem-attack-acc': -1,
        'nm-attack-acc': -1,
        'ntrain': 1,
        'w_label': False,
    }
    with open(logfile, 'r') as f:
        for line in f:
            if line.startswith('Namespace'):
                line = line.strip()
                tokens = line.split(',')
                for token in tokens:
                    if 'attack_type' in token:
                        attack_type = token.split('=')[1]
                        attack_type = attack_type[1:][:-1]
                        if attack_type == 'ml_leaks_label':
                            results_dict['w_label'] = True
                        elif attack_type == 'ml_leaks':
                            results_dict['w_label'] = False
                        else:
                            print('Not supported!')
                            break
                    if 'ntrain=' in token:
                        ntrain = token.split('=')[1].strip()
                        results_dict['ntrain'] = ntrain[:-2][1:]
                    if 'shadow_model_id' in token:
                        shadow_id = token.split('=')[1]
                        results_dict['shadow']=shadow_id
                    if 'target_model_id' in token:
                        target_id = token.split('=')[1]
                        results_dict['target'] = target_id
                    if 'nm_model_id' in token:
                        nm_id = token.split('=')[1]
                        results_dict['nm'] = nm_id
            if 'Overall Testing accuracy' in line:
                acc = line.split(':')[1]
                results_dict['overall-attack-acc'] = float(acc.strip())
            if 'Member Testing accuracy' in line:
                acc = line.split(':')[1]
                results_dict['mem-attack-acc'] = float(acc.strip())
            if 'Non-member Testing accuracy' in line:
                acc = line.split(':')[1]
                results_dict['nm-attack-acc'] = float(acc.strip())
            if '[epoch' in line:
                acc = line.split(' ')[3]
                results_dict['train-attack-acc'] = float(acc.strip()) / 100


    return results_dict

def main(logdir, outname):
    f = open(outname, 'w')
    all_results = {}
    csvwriter = csv.writer(f, delimiter=',')
    csvwriter.writerow(['Dataset', 'Model', 'Training Size', 'Target', 'Shadow', 'Non-member', 'W_label',
                        'Attack Train Acc', 'Overall Attack Acc', 'Mem Attack Acc',
                        'Non-mem Attack Acc'])
    for filename in os.listdir(logdir):
        tokens = filename.split('-')
        model_name = tokens[1]
        width = tokens[2]
        dataset = tokens[0]
        filename = os.path.join(logdir, filename)
        results_dict = parse_results_ml_leaks(filename)
        #print('{} - {}'.format(filename, results_dict))
        mykey = model_name + '-' + width + '-' + str(results_dict['ntrain']) + '-' + str(results_dict['w_label'])
        if mykey not in all_results:
            all_results[mykey] = {}
            all_results[mykey] = {
                'train-avg-attack-acc': [],
                'avg-overall-attack-acc': [],
                'avg-mem-attack-acc': [],
                'avg-nm-attack-acc': [],
                'stdev-overall-attack-acc': -1,
                'diff-overall-attack-acc': -1,
            }
        else:
            if results_dict['train-attack-acc'] >= 0:
                all_results[mykey]['train-avg-attack-acc'].append(float(results_dict['train-attack-acc']))
            if results_dict['overall-attack-acc'] >= 0:
                all_results[mykey]['avg-overall-attack-acc'].append(float(results_dict['overall-attack-acc']))
            if results_dict['mem-attack-acc'] >= 0:
                all_results[mykey]['avg-mem-attack-acc'].append(float(results_dict['mem-attack-acc']))
            if results_dict['nm-attack-acc'] >= 0:
                all_results[mykey]['avg-nm-attack-acc'].append(float(results_dict['nm-attack-acc']))

        csvwriter.writerow([dataset, model_name, width, results_dict['ntrain'], results_dict['target'],
                            results_dict['shadow'], results_dict['nm'], results_dict['w_label'],
                            results_dict['train-attack-acc'], results_dict['overall-attack-acc']])


    csvwriter.writerow(['Dataset', 'Model', 'Width', 'Train Size', 'w_label', 'Avg Train Attack Acc',
                        'Avg Overall Attack Acc', 'Avg Mem Attack Acc',
                        'Avg Non-mem Attack Acc', 'Attack Stdev', 'Attack Diff'])

    for mykey in all_results:
        train_list = all_results[mykey]['train-avg-attack-acc']
        test_list = all_results[mykey]['avg-overall-attack-acc']
        if len(train_list) == 0:
            print('Missing {}'.format(mykey))
            continue
        avg_train_acc = sum(train_list) / len(train_list)
        avg_overall_acc = sum(test_list) / len(test_list)
        mem_list = all_results[mykey]['avg-mem-attack-acc']
        if len(mem_list):
            avg_mem_acc = sum(mem_list) / len(mem_list)
        else:
            print('Missing {}'.format(mykey))
            continue
        nm_list = all_results[mykey]['avg-nm-attack-acc']
        if len(nm_list):
            avg_nm_acc = sum(nm_list) / len(nm_list)
        else:
            print('Missing {}'.format(mykey))
            continue
        try:
            train_stdev = statistics.stdev(train_list)
            test_stdev = statistics.stdev(test_list)
        except:
            # gives error when only one point is there as it is not enough to compute variance
            train_stdev = "-"
            test_stdev = "-"
        test_diff = max(test_list) - min(test_list)

        all_results[mykey]['train-avg-attack-acc'] = avg_train_acc
        all_results[mykey]['avg-overall-attack-acc'] = avg_overall_acc
        all_results[mykey]['avg-mem-attack-acc'] = avg_mem_acc
        all_results[mykey]['avg-nm-attack-acc'] = avg_nm_acc
        all_results[mykey]['stdev-overall-attack-acc'] = test_stdev
        all_results[mykey]['diff-overall-attack-acc'] = test_diff

        tokens = mykey.split('-')
        csvwriter.writerow([dataset, tokens[0], tokens[1], tokens[2], tokens[3], avg_train_acc,
                            avg_overall_acc, avg_mem_acc, avg_nm_acc,
                            test_stdev, test_diff])
    f.close()


if __name__ == "__main__":
    main(sys.argv[1], sys.argv[2])
