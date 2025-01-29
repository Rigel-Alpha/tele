from model.classical_methods.base import classical_methods
from copy import deepcopy
import os.path as ops
import pickle
import time
from sklearn.metrics import accuracy_score, mean_squared_error, f1_score
from sklearn.model_selection import cross_validate, StratifiedKFold
import numpy as np
import copy
def cal_binary_f1(y_true, y_pred):
    y_pred = np.round(y_pred)
    return f1_score(y_true, y_pred)
class XGBoostMethod(classical_methods):
    def __init__(self, args, is_regression):
        super().__init__(args, is_regression)
        assert(args.cat_policy != 'indices')

    def construct_model(self, model_config = None):
        if model_config is None:
            model_config = self.args.config['model']
        from xgboost import XGBClassifier, XGBRegressor
        if self.is_regression:
            self.model = XGBRegressor(**model_config,random_state=self.args.seed)
        else:
            self.model = XGBClassifier(**model_config,random_state=self.args.seed,scale_pos_weight=4,eval_metric=cal_binary_f1)

    def fit(self, N, C, y, info, train=True, config=None):
        super().fit(N, C, y, info, train, config)
        # if not train, skip the training process. such as load the checkpoint and directly predict the results        
        if not train:
            return
        # using stratified kfold to get the best model
        results=[]
        kfold=StratifiedKFold(n_splits=self.args.nfold, random_state=self.args.seed, shuffle=True)
        self.model_list=[]
        tic = time.time()
        for i,(train_index, val_index) in enumerate(kfold.split(self.N['train'], self.y['train'])):
            model= copy.deepcopy(self.model)
            N,y={},{}
            N['train'], N['val'] = copy.deepcopy(self.N['train'][train_index]), copy.deepcopy(self.N['train'][val_index])
            y['train'], y['val'] = copy.deepcopy(self.y['train'][train_index]), copy.deepcopy(self.y['train'][val_index])
            fit_config = deepcopy(self.args.config['fit'])
            fit_config['eval_set'] = [(N['val'], y['val'])]

            model.fit(N['train'], y['train'],**fit_config)
            if not self.is_regression:
                y_val_pred = model.predict(N['val'])
                result = f1_score(y['val'], y_val_pred, average='binary')
            else:
                y_val_pred = model.predict(N['val'])
                result = mean_squared_error(y['val'], y_val_pred, squared=False)*self.y_info['std']
            results.append(result)
            self.model_list.append(model)
        self.trlog["best_res"]=np.mean(results)

        time_cost = time.time() - tic
        for i in range (len(self.model_list)):
            with open(ops.join(self.args.save_path , 'best-val-{}-fold-{}.pkl'.format(self.args.seed,i)), 'wb') as f:
                pickle.dump(self.model_list[i], f)
        return time_cost
    
    def predict(self, N, C, y, info, model_name):
        self.model_list=[]
        for i in range(self.args.nfold):
            with open(ops.join(self.args.save_path , 'best-val-{}-fold-{}.pkl'.format(self.args.seed,i)), 'rb') as f:
                model= pickle.load(f)
            self.model_list.append(model)
        self.data_format(False, N, C, y)
        test_label = None
        test_logits=[]
        #calculate feature importance
        # f_im=0
        for model in self.model_list:
            # f_im+=model.feature_importances_
            if self.is_regression:
                test_logit = model.predict(self.N_test)
            else:
                test_logit = model.predict_proba(self.N_test)
            test_logits.append(test_logit)
        test_logit = np.mean(test_logits, axis=0)
        # f_im=f_im/len(self.model_list)
        return test_logit
        # return  test_logit,f_im
    

        