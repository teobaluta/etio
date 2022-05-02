import os
import shutil
import torch
import numpy as np
import ml_privacy_meter
import pytorch2keras
from tensorflow import keras

ATTACK_DATA = '../attack-data'

def extract(filepath):
    """
    """
    with open(filepath, "r") as f:
        dataset = f.readlines()
    dataset = map(lambda i: i.strip('\n').split(';'), dataset)
    dataset = np.array(list(dataset)) 
    return dataset

def prep_dataset(loader, model_path, start_idx=-1, end_idx=100000000, as_npy=False):
    """
    Save in the ml_privacy_meter format
    Use start_idx and end_idx to select parts of the dataset from the loader
    """
    f = open(model_path, 'w+')
    idx = 0
    wrote_records = 0
    for batch_idx, (inputs, labels) in enumerate(loader):
        inputs_np = inputs.numpy()
        labels = labels.numpy()
        for i, sample in enumerate(inputs_np):
            sample = sample.reshape(sample.shape[0], sample.shape[1] * sample.shape[2])

            final_str = ','.join([str(c) for c in sample[0][:1024]]) + ';' + \
                ','.join([str(c) for c in sample[1][:1024]]) + ';' + \
                ','.join([str(c) for c in sample[2][:1024]]) + ';' + str(labels[i])
            if idx >= start_idx and idx < end_idx:
                f.write(final_str + '\n')
                wrote_records += 1
            i += 1
            idx += 1

    print('Wrote {} records at {}'.format(wrote_records, model_path))
    f.close()
    if as_npy:
        dataset = extract(model_path)
        model_path += '.npy'
        np.save(model_path, dataset)

def convert_to_keras(model):
    # Dummy input image
    # Works with "channels_last" in ~/.keras/keras.json config file and ordering=False
    input_np = np.random.choice([-1, 1], (1, 3, 32, 32))
    input_var = torch.autograd.Variable(torch.FloatTensor(input_np)).cuda()
    keras_model = pytorch2keras.pytorch_to_keras(model, input_var, (3, 32, 32, ), verbose=False,
                                                 change_ordering=False)
    return keras_model

def pre_attack(input_shape, member_path, dataset_path, attack_percentage, batch_size=50):
    """
    saved_path (str):
        Path to a .npy file representing the member training set (training data that was used to
        train the target model) out of which [attack_percentage] of it is used to train the attack
        model. The rest is used for testing the attack model.
    dataset_path (str):
        The member dataset + non-member dataset. An [attack_percentage] of non-members is used to
        train the attack model. The rest is used for testing the attack model.
    """
    datahandlerA = ml_privacy_meter.utils.attack_data.attack_data(dataset_path=dataset_path,
                                                                  member_dataset_path=member_path,
                                                                  batch_size=batch_size,
                                                                  attack_percentage=attack_percentage,
                                                                  input_shape=input_shape,
                                                                  normalization=False)
    return datahandlerA

def blackbox(shadow_model, target_model, train_datahandler, eval_datahandler, layers_to_exploit,
             attack_model_name):
    attackobj = ml_privacy_meter.attack.meminf.initialize(target_train_model=shadow_model,
                                                          target_attack_model=target_model,
                                                          train_datahandler=train_datahandler,
                                                          attack_datahandler=eval_datahandler,
                                                          layers_to_exploit=layers_to_exploit,
                                                          device='/GPU:0', epochs=100,
                                                          exploit_loss=True,
                                                          exploit_label=True,
                                                          model_name=attack_model_name)

    attackobj.train_attack()
    #attackobj.test_attack()


def blackbox_attack(target_model, shadow_model, member_shadow_loader, non_member_shadow_loader,
                    member_target_loader, non_member_target_loader, attack_model_name,
                    eval_nm_size):
    """
    Perform blackbox attack
        - (1 split of training) the member_shadow_loader is a subset of the whole training set. For
          example, we can use the existing splits of 1k, 5k, 10k and their corresponding models.
        - (1 split of training) the non_member_shadow_loader is another split of the whole training
          set. Since there is no intersection between the two and we are taking 100% of the
          datapoints from member_shadow_loader, we will take all datapoints at
          non_member_shadow_loader
    If the target model is trained on the whole dataset.
        - (1 split of training) the member_target_loader is one split of the dataset different from
          the member split. It is just going to be a part of the whole dataset we trained the model
          on, rather than all its training set.
        - (test set) the non_member_target_loader is max the testing set
    """
    if not os.path.exists(ATTACK_DATA):
        os.mkdir(ATTACK_DATA)

    ATTACK_SETS = os.path.join(ATTACK_DATA, attack_model_name)

    target_keras_model = convert_to_keras(target_model)
    shadow_keras_model = convert_to_keras(shadow_model)

    if os.path.exists(ATTACK_SETS):
        try:
            shutil.rmtree(ATTACK_SETS)
        except OSError as e:
            print("Error: %s - %s." % (e.filename, e.strerror))
    os.mkdir(ATTACK_SETS)
    member_shadow_path = os.path.join(ATTACK_SETS, 'shadow_model_mem_path.txt')
    non_member_shadow_path = os.path.join(ATTACK_SETS, 'train_dataset.txt')

    # Prepare the member shadow dataset; the file is saved as .npy
    prep_dataset(member_shadow_loader, member_shadow_path, as_npy=True)
    member_shadow_path += '.npy'
    # load another model's data as non-members
    prep_dataset(non_member_shadow_loader, non_member_shadow_path)

    print('Members for shadow model at {}'.format(member_shadow_path))
    print('Non-members for shadow model at {}'.format(non_member_shadow_path))

    # We are using two different handlers for the data preparations, hence the shadow model data
    # handler has an attack percentage of 100. This means that all of the members at the path
    # `member_shadow_path` are used for training the attack model. The datapoints at
    # `non_member_shadow_path` are used to select the non-members. In the tutorials it is called the
    # `dataset_path` because they are using the same handler for attack training and attack
    # evaluation.
    # The size of the members and non-members is equal to the attack_percentage / 100 *
    # len(non_member_shadow_path).
    # To select all data points at `member_shadow_path` path we need to select to set
    # attack_percentage=100.
    # The non-members are obtained by subtracting the member_shadow_path (the set of members) from the
    # non_member_shadow_path (all the datapoints) and take asize number of datapoints from this
    # difference.
    input_shape = (3, 32, 32)
    attack_percentage = 100
    train_datahandler = pre_attack(input_shape, member_shadow_path, non_member_shadow_path,
                                   attack_percentage)

    member_target_path = os.path.join(ATTACK_SETS, 'target_model_mem_path.txt')
    non_member_target_path = os.path.join(ATTACK_SETS, 'eval_dataset.txt')
    prep_dataset(member_target_loader, member_target_path, start_idx=0, end_idx=eval_nm_size,
                 as_npy=True)
    member_target_path += '.npy'
    prep_dataset(non_member_target_loader, non_member_target_path, start_idx=0, end_idx=eval_nm_size)
    # The members of used for evaluating the attack are constructed from the datapoints at the path
    # member_target_path.
    # The privacy_meter code assumes having the same handler for training and testing. Thus, the
    # members used for testing start from asize index from the set at member_target_path, where size
    # asize (where asize=attack_percentage / 100 * len(member_target_path)) 
    # We directly give training of the target model and testing
    # eval_dataset = target_member + the_other_half_of_test
    # The target_model_member_path is where the test members are. The code in utils/attack_data
    # takes the member_test as from the target_model_member_path starting at asize (so
    # self.train_data[asize:]).
    # Then non-member_test is taken from the: [whole_dataset - (member_test and member_train and
    # nonmember_train)]
    attack_percentage = 0
    eval_datahandler = pre_attack(input_shape, member_target_path, non_member_target_path,
                                  attack_percentage)

    layers_to_exploit = [len(shadow_keras_model.layers) - 1]

    blackbox(shadow_keras_model, target_keras_model, train_datahandler, eval_datahandler,
             layers_to_exploit, attack_model_name)

