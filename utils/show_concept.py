import numpy as np
def categorize(task_size,attributes):
    att_dict = {}
    for i in range(task_size):att_dict[i] = []
    for att in attributes:
        att_dict[int(att/50)].append(att)
    return att_dict

def locate_concepts(attributes,cls):
    attr = categorize(attributes)
    idx = [np.where(attributes == attr[cls][i])[0][0] for i in range(len(attr[cls]))]
    return idx

def hits_cls(idx,score):
    count = 0
    for i in score:
        if i in idx: count += 1
        
def hits_all_cls(self,scores):
    count_list = []
    for i in range(10):
        score_0 = [np.where(scores[k] == scores[k].max())[0][0] for k in range(i*100,(i+1)*100)]
        count = 0
        idx = [np.where(self.attr_dict[0] ==  self.attr[i][j])[0][0] for j in range(len(self.attr[i]))]
        for i in score_0:
            if i in idx: count += 1
        count_list.append(count)