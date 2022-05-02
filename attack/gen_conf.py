import argparse
import random

parser = argparse.ArgumentParser()
parser.add_argument('num_samples', type=int, help='Number of samples of (shaodw,target,nonm)')
parser.add_argument('num_models', type=int, help='Number of models per trial')
parser.add_argument('num_trials', type=int, help='Number of trials')
parser.add_argument('--outfile', type=str, default='shadow_tuples.conf', help='Out file')
parser.add_argument('--seed', type=int, default=0, help='Set the seed')

args = parser.parse_args()

def generate_samples(num_samples, num_models, num_trials, outfile, seed=0):
    """
    Generate a number of tuples (shadow, target, nonmember) per trial

    num_samples: take this many samples from num_models
    num_models: number of models per trial
    num_trials: number of trials, i.e., we take "trials" times num_models
    outfile: file to write to
    seed: seed of random
    """
    random.seed(seed)
    start = 1
    end = num_models + 1
    all_samples = []
    for trial in range(num_trials):
        samples = set()
        models = list(range(start, end))
        print(models)
        while len(samples) < num_samples:
            id1 = random.choice(models)
            id2 = random.choice(models)
            while id2 == id1:
                id2 = random.choice(models)

            id3 = random.choice(models)
            while id3 == id1 or id3 == id2:
                id3 = random.choice(models)

            samples.add((id1, id2, id3))
            print(id1,id2,id3)

            #print('samples {} < num_samples {}'.format(len(samples),num_samples))

        start = (trial + 1 )* num_models + 1
        end = (trial + 2) * num_models + 1
        print('start {} end {}'.format(start, end))
        all_samples.extend(samples)

    with open(outfile, 'w') as f:
        for sample in all_samples:
            t = ' '.join([str(x) for x in sample])
            f.write(t + '\n')

def main():
    generate_samples(args.num_samples, args.num_models, args.num_trials, args.outfile, args.seed)

if __name__ == "__main__":
    main()
