import os
import sys
import csv
import numpy as np

def get_oak17_summary_dict_for_model(summary_file_csv):
    # attack_model_dir = os.path.join(attack_storage_base_dir, "shadow_attack_{0}-target_{1}-shadow_{2}-seeds_{3}".format(model, target, shadow, seeds))
    # summary_base = os.path.dirname(summary_file)

    if not os.path.isfile(summary_file_csv):
        # no such file
        # model_title = "shadow_attack_{0}-target_{1}-shadow_{2}-seeds_{3}".format(model, target, shadow, seeds)
        line = seperator.join(["-" for i in range(6)])
        return {}
    lst = []
    with open(summary_file_csv, 'r') as file:
        reader = csv.reader(file)
        header_pass = False
        for row in reader:
            if row[0].lower().startswith("s"):
                row=row[0].split(" ")
            else:
                row = row[0].split("\t")
            if not header_pass:
                header_pass = True
                header = row
                continue
            if not row[0].isnumeric():
                continue
            # row = row[0].split("\t")# tab seperated values file
            rrow = []
            for i in row:
                if i.strip() != "":
                    rrow.append(float(i))
            lst.append(rrow)

    lst_np = np.array(lst)
    avg  = lst_np.sum(axis=0)
    if len(lst) != 0:
        avg = list(avg / len(lst))
    if type(avg) == type([]):
        data_dict = {}
        for i in range(len(header)):
            field = header[i]
            data_dict[field] = avg[i]
        return data_dict

    return {}
